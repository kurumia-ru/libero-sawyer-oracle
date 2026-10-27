#!/usr/bin/env python3
"""Export successful scripted grasp+place demos to LeRobot format.

Sweeps LIBERO init states per task; keeps episodes that pass smoke metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

OPENPI_EVAL = Path(os.environ.get("OPENPI_EVAL_ROOT", str(Path.home() / "openpi_ws" / "openpi_eval")))
DATA_ROOT = Path(os.environ.get("LIBERO_DATA_ROOT", str(Path.home() / "openpi_data")))

# Per-task grasp presets (from smoke tuning).
SCRIPTED_TASK_PRESETS: dict[int, dict] = {
    0: {
        "grasp_style": "cap",
        "cap_drop": 0.023,
        "insert_depth": 0.004,
        "force_gripper_open": True,
    },
    1: {
        "grasp_style": "side",
        "side_x": 0.022,
        "side_z": 0.006,
        "force_gripper_open": False,
    },
    2: {
        "grasp_style": "cap",
        "cap_drop": 0.023,
        "insert_depth": 0.004,
        "force_gripper_open": True,
    },
    3: {
        "grasp_style": "cap",
        "cap_drop": 0.023,
        "insert_depth": 0.004,
        "force_gripper_open": True,
    },
    4: {
        "grasp_style": "cap",
        "cap_drop": 0.023,
        "insert_depth": 0.004,
        "force_gripper_open": True,
    },
    5: {
        "grasp_style": "cap",
        "cap_drop": 0.023,
        "insert_depth": 0.004,
        "force_gripper_open": True,
    },
    6: {
        "grasp_style": "side",
        "side_x": 0.018,
        "side_z": 0.010,
        "force_gripper_open": False,
    },
    7: {
        "grasp_style": "side",
        "side_x": 0.025,
        "side_z": 0.015,
        "force_gripper_open": False,
    },
    8: {
        "grasp_style": "side",
        "side_x": 0.025,
        "side_z": 0.015,
        "force_gripper_open": False,
    },
    9: {
        "grasp_style": "side",
        "side_x": 0.025,
        "side_z": 0.015,
        "force_gripper_open": False,
    },
}


def _setup_paths() -> None:
    openpi_libero = Path(
        os.environ.get("OPENPI_LIBERO_ROOT", str(Path.home() / "openpi_ws" / "openpi" / "third_party" / "libero"))
    )
    sys.path.insert(0, str(openpi_libero))
    sys.path.insert(0, str(OPENPI_EVAL))


def _torch_patch() -> None:
    import torch

    _orig = torch.load

    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)

    torch.load = _load  # type: ignore


def export_task(
    *,
    suite: str,
    task_id: int,
    dst: object,
    init_indices: list[int],
    motion_profile: str,
    seed: int,
    ik_backend: str,
    start_pose_strategy: str,
    min_frames: int,
    grasp_mode: str = "ik_set",
    require_open_loop: bool = True,
    retrack_osc: bool = False,
    action_space: str = "abs_eef",
    path_step_m: float = 0.014,
    carry_path_step_m: float | None = None,
    post_grasp_repeat: int = 0,
    grasp_repeat: int = 0,
    grasp_near_m: float = 0.12,
    approach_noise_m: float = 0.0,
    approach_noise_copies: int = 0,
    tip_noise_m: float = 0.0,
    tip_noise_copies: int = 0,
    transport_noise_m: float = 0.0,
    transport_noise_copies: int = 0,
    action_noise_std: float = 0.0,
    action_ori_noise_std: float = 0.0,
    action_noise_copies: int = 0,
    action_noise_only: bool = False,
    place_repeat: int = 0,
    release_repeat: int = 0,
    release_pad_min: int = 25,
    min_lift_dz: float = 0.03,
    preclose_repeat: int = 0,
    preclose_frames: int = 12,
) -> list[dict]:
    """Export demos. Default: ik_set live success -> abs_eef (next-state) + IK open-loop."""
    from libero_grasp_smoke import _phases_for_mode, run_grasp_attempt
    from scripted_action_spaces import reencode_abs_eef_from_qpos

    preset = SCRIPTED_TASK_PRESETS[task_id]
    grasp_style = preset["grasp_style"]
    place_to_basket = suite == "object"
    # Live success uses teleport-capable controller for ik_set; honor explicit osc.
    # Cap-grasp cans fail open-loop after ik_set→retrack (OSC misses rim insert).
    if action_space == "abs_eef":
        live_mode = "ik_set"
    elif retrack_osc and grasp_mode == "ik_set":
        live_mode = "ik_set"
    else:
        live_mode = grasp_mode
    live_profile = motion_profile
    records: list[dict] = []

    def _finalize_and_save(
        *,
        out_frames: list[dict],
        demo_frames: list[dict],
        res: dict,
        init_index: int,
        replay_ok: bool | None,
        variant: str,
        approach_noise_m: float,
        tip_noise_m: float,
        transport_noise_m: float,
        action_noise_std: float,
        action_ori_noise_std: float,
        noise_seed: int,
        ok: bool,
    ) -> bool:
        if ok and preclose_repeat > 0:
            out_frames = _oversample_preclose_window(
                out_frames, preclose_repeat, n_pre=preclose_frames
            )
        if ok and grasp_repeat > 0:
            out_frames = _oversample_grasp_window(
                out_frames, grasp_repeat, near_m=grasp_near_m
            )
        if ok and post_grasp_repeat > 0:
            n_before = len(out_frames)
            out_frames = _oversample_closed_carry(out_frames, post_grasp_repeat)
            print(
                f"  [oversample] closed-carry x{1 + post_grasp_repeat}: "
                f"{n_before} -> {len(out_frames)} frames",
                flush=True,
            )
        # Place/release thicken only for episodes that already lifted (contact OK).
        lift_ok = float(res.get("lift_dz") or 0.0) >= float(min_lift_dz)
        if ok and lift_ok and (place_repeat > 0 or release_repeat > 0):
            n_before = len(out_frames)
            out_frames = _oversample_place_release(
                out_frames, place_repeat=place_repeat, release_repeat=release_repeat
            )
            print(
                f"  [oversample] place/release place_x{1 + place_repeat} "
                f"release_x{1 + release_repeat}: {n_before} -> {len(out_frames)} frames",
                flush=True,
            )
        rec = {
            "task_id": task_id,
            "init_index": init_index,
            "passed": res.get("passed"),
            "lifted": res.get("lifted"),
            "placed": res.get("placed"),
            "lift_dz": res.get("lift_dz"),
            "basket_xy_dist": res.get("basket_xy_dist"),
            "n_frames": len(out_frames),
            "n_frames_live": len(demo_frames),
            "exported": bool(ok),
            "grasp_style": grasp_style,
            "grasp_mode": live_mode,
            "action_space": action_space,
            "retrack_osc": retrack_osc,
            "motion_profile": live_profile,
            "path_step_m": path_step_m,
            "carry_path_step_m": carry_path_step_m,
            "post_grasp_repeat": post_grasp_repeat,
            "grasp_repeat": grasp_repeat,
            "grasp_near_m": grasp_near_m,
            "place_repeat": place_repeat,
            "release_repeat": release_repeat,
            "release_pad_min": release_pad_min,
            "min_lift_dz": min_lift_dz,
            "preclose_repeat": preclose_repeat,
            "preclose_frames": preclose_frames,
            "approach_noise_m": approach_noise_m,
            "tip_noise_m": tip_noise_m,
            "transport_noise_m": transport_noise_m,
            "action_noise_std": action_noise_std,
            "action_ori_noise_std": action_ori_noise_std,
            "approach_noise_seed": noise_seed,
            "variant": variant,
            "open_loop_ok": replay_ok,
        }
        records.append(rec)
        status = "SAVE" if ok else "skip"
        release_ok = _has_clear_release(out_frames) if out_frames else False
        print(
            f"  {status}[{variant}]: passed={res.get('passed')} lift_dz={res.get('lift_dz'):+.3f} "
            f"placed={res.get('placed')} frames={len(out_frames)} "
            f"(live={len(demo_frames)}) open_loop={replay_ok} release_ok={release_ok}",
            flush=True,
        )
        if not ok:
            return False
        for fr in out_frames:
            dst.add_frame(
                {
                    "image": fr["image"],
                    "wrist_image": fr["wrist_image"],
                    "state": fr["state"],
                    "actions": fr["actions"],
                    "task": res["task"],
                }
            )
        dst.save_episode()
        return True

    for init_index in init_indices:
        print(
            f"[export] task={task_id} init={init_index} style={grasp_style} "
            f"live={live_mode}/{live_profile} action_space={action_space} "
            f"retrack_osc={retrack_osc} place_to_basket={place_to_basket} "
            f"path_step={path_step_m} carry_step={carry_path_step_m} "
            f"post_grasp_repeat={post_grasp_repeat} grasp_repeat={grasp_repeat} "
            f"preclose_repeat={preclose_repeat} place_repeat={place_repeat} "
            f"release_repeat={release_repeat} min_lift_dz={min_lift_dz} "
            f"approach_noise_m={approach_noise_m}x{approach_noise_copies} "
            f"tip_noise_m={tip_noise_m}x{tip_noise_copies} "
            f"transport_noise_m={transport_noise_m}x{transport_noise_copies}",
            flush=True,
        )
        res = run_grasp_attempt(
            robot="sawyer",
            suite=suite,
            task_id=task_id,
            init_index=init_index,
            start_pose_strategy=start_pose_strategy,
            phases=_phases_for_mode(
                live_mode,
                0.05,
                grasp_style,
                place_to_basket=place_to_basket,
                motion_profile=live_profile,
            ),
            record_video=False,
            record_demo=True,
            ik_backend=ik_backend,
            grasp_mode=live_mode,
            grasp_style=grasp_style,
            grasp_z=0.05,
            rim_inset=0.004,
            bite_depth=0.003,
            insert_depth=preset.get("insert_depth", 0.004),
            cap_drop=preset.get("cap_drop", 0.023),
            force_gripper_open=preset.get("force_gripper_open", False),
            side_x=preset.get("side_x"),
            side_z=preset.get("side_z"),
            place_to_basket=place_to_basket,
            motion_profile=live_profile,
        )
        demo_frames = res.get("demo_frames") or []
        lift_dz = float(res.get("lift_dz") or 0.0)
        grip_ok = bool(res.get("gripper_closed", True))
        live_ok = (
            bool(res.get("passed"))
            and len(demo_frames) >= min_frames
            and lift_dz >= float(min_lift_dz)
            and grip_ok
        )
        if not live_ok:
            records.append(
                {
                    "task_id": task_id,
                    "init_index": init_index,
                    "passed": res.get("passed"),
                    "lift_dz": lift_dz,
                    "gripper_closed": grip_ok,
                    "exported": False,
                    "variant": "live_fail",
                    "n_frames_live": len(demo_frames),
                    "open_loop_ok": None,
                }
            )
            print(
                f"  skip: passed={res.get('passed')} lift_dz={lift_dz:+.3f} "
                f"(min={min_lift_dz:.3f}) grip_closed={grip_ok} "
                f"frames_live={len(demo_frames)}",
                flush=True,
            )
            continue

        # Clean + optional approach / tip / transport / action-noise variants.
        # Variants are separate (not a cartesian product) to keep episode counts sane.
        # action_noise: MimicGen-style Gaussian on executed OSC actions; only
        # open-loop-successful rollouts are saved (--require-open-loop).
        variants: list[dict] = []
        if not (action_noise_only and action_noise_std > 0.0 and action_noise_copies > 0):
            variants.append(
                {
                    "variant": "clean",
                    "apr": 0.0,
                    "tip": 0.0,
                    "tr": 0.0,
                    "act": 0.0,
                    "act_ori": 0.0,
                    "seed": 0,
                }
            )
        if approach_noise_m > 0.0 and approach_noise_copies > 0:
            for k in range(approach_noise_copies):
                variants.append(
                    {
                        "variant": f"apr{k}",
                        "apr": float(approach_noise_m),
                        "tip": 0.0,
                        "tr": 0.0,
                        "act": 0.0,
                        "act_ori": 0.0,
                        "seed": int(seed + 1009 * (init_index + 1) + 17 * (k + 1)),
                    }
                )
        if tip_noise_m > 0.0 and tip_noise_copies > 0:
            for k in range(tip_noise_copies):
                variants.append(
                    {
                        "variant": f"tip{k}",
                        "apr": 0.0,
                        "tip": float(tip_noise_m),
                        "tr": 0.0,
                        "act": 0.0,
                        "act_ori": 0.0,
                        "seed": int(seed + 3011 * (init_index + 1) + 41 * (k + 1)),
                    }
                )
        if transport_noise_m > 0.0 and transport_noise_copies > 0:
            for k in range(transport_noise_copies):
                variants.append(
                    {
                        "variant": f"tr{k}",
                        "apr": 0.0,
                        "tip": 0.0,
                        "tr": float(transport_noise_m),
                        "act": 0.0,
                        "act_ori": 0.0,
                        "seed": int(seed + 2039 * (init_index + 1) + 31 * (k + 1)),
                    }
                )
        if action_noise_std > 0.0 and action_noise_copies > 0:
            for k in range(action_noise_copies):
                variants.append(
                    {
                        "variant": f"act{k}",
                        "apr": 0.0,
                        "tip": 0.0,
                        "tr": 0.0,
                        "act": float(action_noise_std),
                        "act_ori": float(action_ori_noise_std),
                        "seed": int(seed + 4099 * (init_index + 1) + 53 * (k + 1)),
                    }
                )
        if not variants:
            variants.append(
                {
                    "variant": "clean",
                    "apr": 0.0,
                    "tip": 0.0,
                    "tr": 0.0,
                    "act": 0.0,
                    "act_ori": 0.0,
                    "seed": 0,
                }
            )

        for v in variants:
            variant = str(v["variant"])
            apr_m = float(v["apr"])
            tip_m = float(v["tip"])
            tr_m = float(v["tr"])
            act_std = float(v["act"])
            act_ori_std = float(v["act_ori"])
            noise_seed = int(v["seed"])
            replay_ok = None
            out_frames = demo_frames
            if action_space == "abs_eef":
                cont = _retrack_osc_frames(
                    demo_frames,
                    suite=suite,
                    task_id=task_id,
                    init_index=init_index,
                    ik_backend=ik_backend,
                    start_pose_strategy=start_pose_strategy,
                    pad_open=True,
                    path_step_m=path_step_m,
                    carry_path_step_m=carry_path_step_m,
                    approach_noise_m=apr_m,
                    approach_noise_seed=noise_seed,
                    tip_noise_m=tip_m,
                    tip_noise_seed=noise_seed + 3,
                    transport_noise_m=tr_m,
                    transport_noise_seed=noise_seed + 7,
                    action_noise_std=act_std,
                    action_ori_noise_std=act_ori_std,
                    action_noise_seed=noise_seed + 11,
                    release_pad_min=release_pad_min,
                )
                out_frames = reencode_abs_eef_from_qpos(cont)
                ok = len(out_frames) >= min_frames and _has_clear_release(
                    out_frames, min_open=max(25, release_pad_min)
                )
                if ok and require_open_loop:
                    replay_ok = _open_loop_replay_ok(
                        out_frames,
                        suite=suite,
                        task_id=task_id,
                        init_index=init_index,
                        start_pose_strategy=start_pose_strategy,
                        ik_backend=ik_backend,
                        action_space="abs_eef",
                    )
                    ok = replay_ok is not False
                _finalize_and_save(
                    out_frames=out_frames,
                    demo_frames=demo_frames,
                    res=res,
                    init_index=init_index,
                    replay_ok=replay_ok,
                    variant=variant,
                    approach_noise_m=apr_m,
                    tip_noise_m=tip_m,
                    transport_noise_m=tr_m,
                    action_noise_std=act_std,
                    action_ori_noise_std=act_ori_std,
                    noise_seed=noise_seed,
                    ok=bool(ok),
                )
            elif retrack_osc:
                out_frames = _retrack_osc_frames(
                    demo_frames,
                    suite=suite,
                    task_id=task_id,
                    init_index=init_index,
                    ik_backend=ik_backend,
                    start_pose_strategy=start_pose_strategy,
                    pad_open=True,
                    path_step_m=path_step_m,
                    carry_path_step_m=carry_path_step_m,
                    approach_noise_m=apr_m,
                    approach_noise_seed=noise_seed,
                    tip_noise_m=tip_m,
                    tip_noise_seed=noise_seed + 3,
                    transport_noise_m=tr_m,
                    transport_noise_seed=noise_seed + 7,
                    action_noise_std=act_std,
                    action_ori_noise_std=act_ori_std,
                    action_noise_seed=noise_seed + 11,
                    release_pad_min=release_pad_min,
                )
                ok = len(out_frames) >= min_frames and _has_clear_release(
                    out_frames, min_open=max(25, release_pad_min)
                )
                if ok and require_open_loop:
                    replay_ok = _open_loop_replay_ok(
                        out_frames,
                        suite=suite,
                        task_id=task_id,
                        init_index=init_index,
                        start_pose_strategy=start_pose_strategy,
                        ik_backend=ik_backend,
                        action_space="delta_osc",
                    )
                    ok = replay_ok is not False
                _finalize_and_save(
                    out_frames=out_frames,
                    demo_frames=demo_frames,
                    res=res,
                    init_index=init_index,
                    replay_ok=replay_ok,
                    variant=variant,
                    approach_noise_m=apr_m,
                    tip_noise_m=tip_m,
                    transport_noise_m=tr_m,
                    action_noise_std=act_std,
                    action_ori_noise_std=act_ori_std,
                    noise_seed=noise_seed,
                    ok=bool(ok),
                )
            else:
                if variant != "clean":
                    continue
                ok = True
                if require_open_loop:
                    replay_ok = _open_loop_replay_ok(
                        out_frames,
                        suite=suite,
                        task_id=task_id,
                        init_index=init_index,
                        start_pose_strategy=start_pose_strategy,
                        ik_backend=ik_backend,
                        action_space=action_space,
                    )
                    ok = replay_ok is not False
                _finalize_and_save(
                    out_frames=out_frames,
                    demo_frames=demo_frames,
                    res=res,
                    init_index=init_index,
                    replay_ok=replay_ok,
                    variant=variant,
                    approach_noise_m=0.0,
                    tip_noise_m=0.0,
                    transport_noise_m=0.0,
                    action_noise_std=0.0,
                    action_ori_noise_std=0.0,
                    noise_seed=0,
                    ok=bool(ok),
                )
    return records


def _pose_track_action_scaled(
    obs: dict,
    target_state: np.ndarray,
    gripper: float,
    *,
    pos_scale: float = 0.05,
    rot_scale: float = 0.5,
) -> np.ndarray:
    """Like generate_b_demos._pose_track_action but with configurable scales."""
    from generate_b_demos import _sim_proprio_state

    cur = _sim_proprio_state(obs)
    target_state = np.asarray(target_state, dtype=np.float32)
    scale_p = np.array([pos_scale, pos_scale, pos_scale], dtype=np.float32)
    scale_r = np.array([rot_scale, rot_scale, rot_scale], dtype=np.float32)
    dpos = (target_state[:3] - cur[:3]) / scale_p
    drot = (target_state[3:6] - cur[3:6]) / scale_r
    return np.concatenate(
        [
            np.clip(dpos, -1.0, 1.0),
            np.clip(drot, -1.0, 1.0),
            np.array([gripper], dtype=np.float32),
        ]
    ).astype(np.float32)


def _oversample_closed_carry(frames: list[dict], repeat: int) -> list[dict]:
    """Duplicate closed-gripper frames to upweight post-grasp / transport for BC.

    Each closed frame is emitted (1 + repeat) times. Open-gripper approach/release
    frames are left unchanged. Safe after open-loop validation because each
    (obs, action) pair remains independently valid.
    """
    if repeat <= 0 or not frames:
        return frames
    out: list[dict] = []
    for fr in frames:
        out.append(fr)
        grip = float(np.asarray(fr["actions"], dtype=np.float32).reshape(-1)[-1])
        if grip > 0.0:
            for _ in range(repeat):
                out.append(fr)
    return out


def _oversample_preclose_window(
    frames: list[dict],
    repeat: int,
    *,
    n_pre: int = 12,
) -> list[dict]:
    """Upsample open-gripper frames just before the first close (approach tip).

    Helps BC learn the last few cm / timing before contact without boosting carry.
    """
    if repeat <= 0 or not frames or n_pre <= 0:
        return frames
    first_close = None
    for i, fr in enumerate(frames):
        grip = float(np.asarray(fr["actions"], dtype=np.float32).reshape(-1)[-1])
        if grip > 0.0:
            first_close = i
            break
    if first_close is None or first_close == 0:
        return frames
    lo = max(0, first_close - int(n_pre))
    out: list[dict] = []
    n_boost = 0
    for i, fr in enumerate(frames):
        out.append(fr)
        if lo <= i < first_close:
            n_boost += 1
            for _ in range(repeat):
                out.append(fr)
    print(
        f"  [oversample] preclose x{1 + repeat} n_pre={n_pre} "
        f"boosted={n_boost} frames: {len(frames)} -> {len(out)}",
        flush=True,
    )
    return out


def _oversample_grasp_window(
    frames: list[dict],
    repeat: int,
    *,
    near_m: float = 0.12,
) -> list[dict]:
    """Upsample closed-gripper frames still near the first-close EEF pose.

    Targets close/squeeze/contact/lift without upweighting basket transport
    (which is far from the grasp pose in XY/Z).
    """
    if repeat <= 0 or not frames:
        return frames
    ref_xyz: np.ndarray | None = None
    for fr in frames:
        grip = float(np.asarray(fr["actions"], dtype=np.float32).reshape(-1)[-1])
        if grip > 0.0:
            ref_xyz = np.asarray(fr["state"], dtype=np.float64)[:3].copy()
            break
    if ref_xyz is None:
        return frames
    out: list[dict] = []
    n_boost = 0
    for fr in frames:
        out.append(fr)
        grip = float(np.asarray(fr["actions"], dtype=np.float32).reshape(-1)[-1])
        xyz = np.asarray(fr["state"], dtype=np.float64)[:3]
        if grip > 0.0 and float(np.linalg.norm(xyz - ref_xyz)) <= near_m:
            n_boost += 1
            for _ in range(repeat):
                out.append(fr)
    print(
        f"  [oversample] grasp-window x{1 + repeat} near={near_m:.2f}m "
        f"boosted={n_boost} frames: {len(frames)} -> {len(out)}",
        flush=True,
    )
    return out


def _oversample_place_release(
    frames: list[dict],
    *,
    place_repeat: int = 0,
    release_repeat: int = 0,
) -> list[dict]:
    """Upsample hold-over-basket (last closed stretch) and the open release tail.

    Finds the last closed->open transition and boosts:
    - closed frames in a window before that transition (place hold)
    - open frames after that transition (release)
    """
    if not frames or (place_repeat <= 0 and release_repeat <= 0):
        return frames
    grips = [float(np.asarray(fr["actions"], dtype=np.float32).reshape(-1)[-1]) for fr in frames]
    release_i = None
    for i in range(1, len(grips)):
        if grips[i - 1] > 0.0 and grips[i] < -0.5:
            release_i = i
    if release_i is None:
        # Fallback: first sustained open in the last half.
        half = len(grips) // 2
        for i in range(half, len(grips)):
            if grips[i] < -0.5:
                release_i = i
                break
    if release_i is None:
        return frames
    hold_lo = max(0, release_i - 40)
    out: list[dict] = []
    for i, fr in enumerate(frames):
        out.append(fr)
        if place_repeat > 0 and hold_lo <= i < release_i and grips[i] > 0.0:
            for _ in range(place_repeat):
                out.append(fr)
        if release_repeat > 0 and i >= release_i and grips[i] < -0.5:
            for _ in range(release_repeat):
                out.append(fr)
    return out


def _inject_approach_detour(
    waypoints: list[tuple[np.ndarray, float]],
    *,
    noise_m: float,
    seed: int,
) -> list[tuple[np.ndarray, float]]:
    """Bend the open-gripper approach off-path then back (recovery demos for BC).

    Only touches frames before the first gripper-close. Closed/carry/release are
    unchanged so grasp geometry stays intact while approach learns return-to-path.
    """
    if noise_m <= 0.0 or len(waypoints) < 12:
        return waypoints
    first_close = next((i for i, (_xyz, g) in enumerate(waypoints) if float(g) > 0.0), len(waypoints))
    if first_close < 12:
        return waypoints
    rng = np.random.default_rng(seed)
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    amp = float(rng.uniform(0.5 * noise_m, noise_m))
    delta = np.array(
        [amp * np.cos(ang), amp * np.sin(ang), float(rng.uniform(-0.3, 0.3)) * amp],
        dtype=np.float64,
    )
    i0 = max(2, int(0.20 * first_close))
    i1 = min(first_close - 2, int(0.75 * first_close))
    if i1 - i0 < 6:
        return waypoints
    mid = 0.5 * (i0 + i1)
    half = max(1.0, 0.5 * (i1 - i0))
    out: list[tuple[np.ndarray, float]] = []
    for i, (xyz, g) in enumerate(waypoints):
        p = np.asarray(xyz, dtype=np.float64).copy()
        if i0 <= i <= i1 and float(g) <= 0.0:
            w = max(0.0, 1.0 - abs(i - mid) / half)
            p = p + delta * w
        out.append((p.astype(np.float32), float(g)))
    print(
        f"  [noise] approach detour amp={amp:.3f}m "
        f"dxyz=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f}) i={i0}-{i1}/{first_close}",
        flush=True,
    )
    return out


def _inject_tip_detour(
    waypoints: list[tuple[np.ndarray, float]],
    *,
    noise_m: float,
    seed: int,
) -> list[tuple[np.ndarray, float]]:
    """Bend only the late open-gripper tip (near → close) off-path then back.

    Complements mid-approach detours: smaller amplitude, window just before first
    close so BC sees near-miss / re-seat recovery without changing grasp geometry.
    """
    if noise_m <= 0.0 or len(waypoints) < 16:
        return waypoints
    first_close = next((i for i, (_xyz, g) in enumerate(waypoints) if float(g) > 0.0), len(waypoints))
    if first_close < 16:
        return waypoints
    rng = np.random.default_rng(seed)
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    amp = float(rng.uniform(0.5 * noise_m, noise_m))
    # Tip needs mild Z too (undershoot/overshoot height), but keep it smaller.
    delta = np.array(
        [amp * np.cos(ang), amp * np.sin(ang), float(rng.uniform(-0.45, 0.45)) * amp],
        dtype=np.float64,
    )
    i0 = max(2, int(0.70 * first_close))
    i1 = min(first_close - 2, int(0.97 * first_close))
    if i1 - i0 < 4:
        return waypoints
    mid = 0.5 * (i0 + i1)
    half = max(1.0, 0.5 * (i1 - i0))
    out: list[tuple[np.ndarray, float]] = []
    for i, (xyz, g) in enumerate(waypoints):
        p = np.asarray(xyz, dtype=np.float64).copy()
        if i0 <= i <= i1 and float(g) <= 0.0:
            w = max(0.0, 1.0 - abs(i - mid) / half)
            p = p + delta * w
        out.append((p.astype(np.float32), float(g)))
    print(
        f"  [noise] tip detour amp={amp:.3f}m "
        f"dxyz=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f}) i={i0}-{i1}/{first_close}",
        flush=True,
    )
    return out


def _inject_transport_detour(
    waypoints: list[tuple[np.ndarray, float]],
    *,
    noise_m: float,
    seed: int,
) -> list[tuple[np.ndarray, float]]:
    """Bend the closed-gripper carry (post-grasp → near-basket) off-path then back.

    Leaves the grasp contact window and final place/release mostly untouched so
    contact geometry stays stable while transport learns lateral recovery.
    """
    if noise_m <= 0.0 or len(waypoints) < 20:
        return waypoints
    first_close = next((i for i, (_xyz, g) in enumerate(waypoints) if float(g) > 0.0), None)
    if first_close is None:
        return waypoints
    last_closed = first_close
    for i in range(first_close, len(waypoints)):
        if float(waypoints[i][1]) > 0.0:
            last_closed = i
        elif i > first_close + 5:
            # Sustained open after carry => release started.
            break
    span = last_closed - first_close
    if span < 16:
        return waypoints

    # Skip early lift / late place-hold; perturb the mid carry only.
    i0 = first_close + max(6, int(0.20 * span))
    i1 = first_close + min(span - 4, int(0.80 * span))
    if i1 - i0 < 8:
        return waypoints

    rng = np.random.default_rng(seed)
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    amp = float(rng.uniform(0.5 * noise_m, noise_m))
    # Mostly XY; mild Z so carried object height stays plausible.
    delta = np.array(
        [amp * np.cos(ang), amp * np.sin(ang), float(rng.uniform(-0.2, 0.35)) * amp],
        dtype=np.float64,
    )
    mid = 0.5 * (i0 + i1)
    half = max(1.0, 0.5 * (i1 - i0))
    out: list[tuple[np.ndarray, float]] = []
    for i, (xyz, g) in enumerate(waypoints):
        p = np.asarray(xyz, dtype=np.float64).copy()
        if i0 <= i <= i1 and float(g) > 0.0:
            w = max(0.0, 1.0 - abs(i - mid) / half)
            p = p + delta * w
        out.append((p.astype(np.float32), float(g)))
    print(
        f"  [noise] transport detour amp={amp:.3f}m "
        f"dxyz=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f}) "
        f"i={i0}-{i1} close={first_close} last_closed={last_closed}",
        flush=True,
    )
    return out


def _smooth_xyz_waypoints(
    live_frames: list[dict],
    *,
    step_m: float = 0.005,
    carry_step_m: float | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Resample live EEF XYZ to near-constant speed (removes ik_set teleport jerks).

    Returns (xyz3, gripper) waypoints spaced ~step_m apart along the polyline.
    When carry_step_m is set, closed-gripper segments use that finer spacing.

    Large ik_set teleports are routed via a raised via-point so linear
    interpolation does not cut through the object (critical for cap-grasp cans).
    """
    if not live_frames:
        return []
    pts: list[tuple[np.ndarray, float]] = []
    jump_m = 0.04
    jump_clear_z = 0.10
    for fr in live_frames:
        xyz = np.asarray(fr["state"], dtype=np.float64)[:3].copy()
        grip = float(np.asarray(fr["actions"], dtype=np.float32).reshape(-1)[-1])
        if not pts:
            pts.append((xyz, grip))
            continue
        prev, _ = pts[-1]
        # Keep gripper changes even if EEF barely moves (close/open holds).
        if abs(grip - pts[-1][1]) > 0.05 and float(np.linalg.norm(xyz - prev)) < 1e-4:
            pts.append((xyz.copy(), grip))
            continue
        if float(np.linalg.norm(xyz - prev)) < 1e-5:
            pts[-1] = (prev, grip)
            continue
        jump = float(np.linalg.norm(xyz - prev))
        if jump >= jump_m:
            z_via = max(float(prev[2]), float(xyz[2]), jump_clear_z) + 0.05
            via1 = np.array([prev[0], prev[1], z_via], dtype=np.float64)
            via2 = np.array([xyz[0], xyz[1], z_via], dtype=np.float64)
            open_g = -1.0 if (float(grip) <= 0.0 or float(pts[-1][1]) <= 0.0) else float(pts[-1][1])
            pts.append((via1, open_g))
            pts.append((via2, open_g))
            print(
                f"  [smooth] retract jump={jump:.3f}m via_z={z_via:.3f}",
                flush=True,
            )
        pts.append((xyz, grip))

    out: list[tuple[np.ndarray, float]] = []
    for i, (xyz, grip) in enumerate(pts):
        if i == 0:
            out.append((xyz.astype(np.float32), grip))
            continue
        prev_xyz, prev_grip = out[-1]
        delta = xyz - np.asarray(prev_xyz, dtype=np.float64)
        dist = float(np.linalg.norm(delta))
        use_step = step_m
        if (
            carry_step_m is not None
            and carry_step_m > 0
            and float(prev_grip) > 0.0
            and float(grip) > 0.0
        ):
            use_step = float(carry_step_m)
        if dist < use_step * 0.5:
            out.append((xyz.astype(np.float32), grip))
            continue
        n = max(1, int(np.ceil(dist / use_step)))
        for k in range(1, n + 1):
            t = k / n
            p = (1.0 - t) * np.asarray(prev_xyz, dtype=np.float64) + t * xyz
            g = prev_grip if t < 0.5 else grip
            out.append((p.astype(np.float32), float(g)))
    return out


def _retrack_osc_frames(
    live_frames: list[dict],
    *,
    suite: str,
    task_id: int,
    init_index: int,
    ik_backend: str,
    start_pose_strategy: str | None = None,
    pos_tol_m: float = 0.008,
    pad_open: bool = True,
    path_step_m: float = 0.014,
    carry_path_step_m: float | None = None,
    approach_noise_m: float = 0.0,
    approach_noise_seed: int = 0,
    tip_noise_m: float = 0.0,
    tip_noise_seed: int = 0,
    transport_noise_m: float = 0.0,
    transport_noise_seed: int = 0,
    action_noise_std: float = 0.0,
    action_ori_noise_std: float = 0.0,
    action_noise_seed: int = 0,
    release_pad_min: int = 25,
) -> list[dict]:
    """Rebuild learnable OSC demos by chasing a smoothed live EEF path.

    ik_set live demos teleport between phases (jerky). We first resample XYZ to
    ~14 mm steps, then OSC-chase with a snappier scale so densified abs_eef stays
    smooth and open-loop/eval do not need IK teleports.

    When action_noise_std > 0, MimicGen-style Gaussian noise is added to the
    executed delta-OSC action (xyz/ori; gripper untouched). The noisy action is
    what gets recorded, so image / proprio / action stay consistent.
    """
    from generate_b_demos import (
        LIBERO_DUMMY,
        NUM_STABILIZE,
        _make_env,
        _render_pair,
        _sim_proprio_state,
    )
    from libero.libero import benchmark
    from preview_scripted_demos import _apply_export_setup

    bench_name = {"object": "libero_object", "spatial": "libero_spatial", "long": "libero_10"}[suite]
    task_suite = benchmark.get_benchmark_dict()[bench_name]()
    task = task_suite.get_task(task_id)
    inits = task_suite.get_task_init_states(task_id)
    env = _make_env(task, "sawyer", 7, horizon=4000)
    env.reset()
    obs = env.set_init_state(inits[init_index % len(inits)])
    for _ in range(NUM_STABILIZE):
        obs, _, _, _ = env.step(LIBERO_DUMMY)
    obs, _ = _apply_export_setup(
        env,
        obs,
        task_id=task_id,
        suite=suite,
        ik_backend=ik_backend,
        start_pose_strategy=start_pose_strategy,
    )

    act_rng = (
        np.random.default_rng(int(action_noise_seed))
        if (action_noise_std > 0.0 or action_ori_noise_std > 0.0)
        else None
    )

    waypoints = _smooth_xyz_waypoints(
        live_frames, step_m=path_step_m, carry_step_m=carry_path_step_m
    )
    # The first recorded live frame is after its first env.step. Bridge from the
    # actual replay start pose to that first point; otherwise a mid-task initial
    # pose can leave an unlearned/unreachable jump before the grasp.
    if waypoints:
        start_xyz = np.asarray(_sim_proprio_state(obs)[:3], dtype=np.float64)
        first_xyz = np.asarray(waypoints[0][0], dtype=np.float64)
        first_grip = float(waypoints[0][1])
        start_dist = float(np.linalg.norm(first_xyz - start_xyz))
        if start_dist > path_step_m * 0.5:
            # Initial approach needs finer spacing than free-space carry so the
            # gripper is actually seated before the first close command.
            bridge_step_m = min(path_step_m, 0.006)
            n_bridge = max(1, int(np.ceil(start_dist / bridge_step_m)))
            bridge = [
                (
                    (
                        start_xyz
                        + (first_xyz - start_xyz) * ((k + 1) / n_bridge)
                    ).astype(np.float32),
                    first_grip,
                )
                for k in range(n_bridge)
            ]
            # Brief open hold at the endpoint lets OSC settle without adding the
            # old home-pose warmup.
            bridge.extend([(first_xyz.astype(np.float32), first_grip)] * 6)
            waypoints = bridge + waypoints[1:]
            print(
                f"  [retrack] start bridge dist={start_dist:.3f}m "
                f"step={bridge_step_m:.3f}m n={n_bridge}",
                flush=True,
            )
    # Inject after bridge so early approach (where closed-loop drifts) is covered.
    if approach_noise_m > 0.0:
        waypoints = _inject_approach_detour(
            waypoints, noise_m=approach_noise_m, seed=approach_noise_seed
        )
    if tip_noise_m > 0.0:
        waypoints = _inject_tip_detour(
            waypoints, noise_m=tip_noise_m, seed=tip_noise_seed
        )
    if transport_noise_m > 0.0:
        waypoints = _inject_transport_detour(
            waypoints, noise_m=transport_noise_m, seed=transport_noise_seed
        )
    carry_msg = (
        f" carry_step={carry_path_step_m:.3f}m"
        if carry_path_step_m is not None
        else ""
    )
    print(
        f"  [retrack] smooth waypoints={len(waypoints)} from live={len(live_frames)} "
        f"step={path_step_m:.3f}m{carry_msg}"
        f" act_noise=xyz{action_noise_std:.3f}/ori{action_ori_noise_std:.3f}",
        flush=True,
    )

    out: list[dict] = []
    episode_done = False

    def _append_step(action: np.ndarray) -> tuple[bool, float]:
        nonlocal obs, episode_done
        action = np.asarray(action, dtype=np.float32).copy()
        if act_rng is not None:
            if action_noise_std > 0.0 and action.shape[0] >= 3:
                action[:3] += act_rng.normal(0.0, float(action_noise_std), size=3).astype(
                    np.float32
                )
            if action_ori_noise_std > 0.0 and action.shape[0] >= 6:
                action[3:6] += act_rng.normal(
                    0.0, float(action_ori_noise_std), size=3
                ).astype(np.float32)
            action[:6] = np.clip(action[:6], -1.0, 1.0)
        cur = _sim_proprio_state(obs)
        img, wrist = _render_pair(obs)
        out.append(
            {
                "image": img,
                "wrist_image": wrist,
                "state": cur,
                "actions": action,
            }
        )
        try:
            obs, reward, done, _ = env.step(action.tolist())
        except ValueError:
            episode_done = True
            return True, 0.0
        # LIBERO often sets done=True as soon as the object is in the basket while
        # still gripping. Keep stepping so densify can record release/retreat.
        if bool(done) and not episode_done:
            episode_done = True
            print(
                f"  [retrack] env success/done at out_frames={len(out)}; continue for release",
                flush=True,
            )
        return False, float(reward)

    def _pad_open_release(n: int = 45) -> None:
        """Append hold+open actions even if the env already terminated.

        LIBERO often sets done=True while the gripper is still closed over the
        basket; without this pad, demos never teach release.
        """
        if not out:
            return
        last = out[-1]
        # Also rewrite a short closed tail to open so BC sees open sooner.
        for fr in out[-20:]:
            a = np.asarray(fr["actions"], dtype=np.float32).copy()
            a[-1] = np.float32(-1.0)
            fr["actions"] = a
        act = np.zeros(7, dtype=np.float32)
        act[-1] = np.float32(-1.0)
        state = np.asarray(last["state"], dtype=np.float32).copy()
        for _ in range(n):
            out.append(
                {
                    "image": last["image"],
                    "wrist_image": last["wrist_image"],
                    "state": state.copy(),
                    "actions": act.copy(),
                }
            )

    # Faster OSC chase; still clipped to avoid hard bang-bang.
    # Cap-grasp cans (task0) need more steps/waypoint than side-grasp boxes —
    # otherwise close commands fire ~2–3cm off-axis and open-loop never lifts.
    pos_scale = 0.020
    per_wp_budget = max(8, int(round(path_step_m / 0.003)) + 4)
    close_xy_tol = min(pos_tol_m, 0.006)

    for i, (xyz, grip) in enumerate(waypoints):
        hard_stop = False
        want_close = float(grip) >= 0.0
        budget = per_wp_budget * (3 if want_close else 1)
        for _step_i in range(budget):
            cur = _sim_proprio_state(obs)
            pos_err = float(np.linalg.norm(cur[:3] - np.asarray(xyz, dtype=np.float64)))
            xy_err = float(np.linalg.norm(cur[:2] - np.asarray(xyz[:2], dtype=np.float64)))
            # Do not close until we are seated on the waypoint (esp. tall thin cans).
            cmd_grip = float(grip)
            if want_close and xy_err > close_xy_tol:
                cmd_grip = -1.0
            pos_tgt = cur.copy()
            pos_tgt[:3] = xyz
            action = _pose_track_action_scaled(
                obs, pos_tgt, cmd_grip, pos_scale=pos_scale, rot_scale=0.5
            )
            _done, _reward = _append_step(action)
            if _done:
                print(f"  [retrack] hard stop at wp={i} out_frames={len(out)}", flush=True)
                hard_stop = True
                break
            tol = close_xy_tol if want_close else pos_tol_m
            if float(np.linalg.norm(_sim_proprio_state(obs)[:3] - xyz)) < tol:
                break
        if hard_stop:
            break
        # Brief squeeze settle on close waypoints so contact can form before lift.
        if want_close and not hard_stop:
            for _ in range(4):
                cur = _sim_proprio_state(obs)
                pos_tgt = cur.copy()
                pos_tgt[:3] = xyz
                action = _pose_track_action_scaled(
                    obs, pos_tgt, float(grip), pos_scale=pos_scale, rot_scale=0.5
                )
                _done, _reward = _append_step(action)
                if _done:
                    hard_stop = True
                    break
            if hard_stop:
                break

    # Always pad a clear open-gripper release tail (env often ends while still closed).
    min_open = max(25, int(release_pad_min))
    grip_hist = [float(np.asarray(fr["actions"], dtype=np.float32)[-1]) for fr in out[-60:]]
    n_open = sum(1 for g in grip_hist if g < -0.5)
    if pad_open and n_open < min_open:
        # Only synthesize a tail if the real trajectory has no clear release.
        n_pad = max(min_open, min_open + 5 - n_open)
        print(
            f"  [retrack] pre-pad open in last60={n_open}; padding release n={n_pad} "
            f"(min_open={min_open})",
            flush=True,
        )
        _pad_open_release(n_pad)
    else:
        print(f"  [retrack] skip open-pad (last60 open={n_open}, min={min_open})", flush=True)

    print(f"  [retrack] out_frames={len(out)} episode_done={episode_done}", flush=True)
    env.close()
    return out


def _has_clear_release(demo_frames: list[dict], *, min_open: int = 25) -> bool:
    """Require sustained gripper-open near the end (release teaching signal)."""
    if len(demo_frames) < min_open:
        return False
    grips = [float(np.asarray(fr["actions"], dtype=np.float32)[-1]) for fr in demo_frames]
    return sum(1 for g in grips[-60:] if g < -0.5) >= min_open


def _open_loop_replay_ok(
    demo_frames: list[dict],
    *,
    suite: str,
    task_id: int,
    init_index: int,
    start_pose_strategy: str,
    ik_backend: str,
    min_lift_dz: float = 0.03,
    action_space: str = "delta_osc",
) -> bool:
    """Replay recorded actions; require lift, and basket place for object suite."""
    from generate_b_demos import LIBERO_DUMMY, NUM_STABILIZE, _make_env
    from libero.libero import benchmark
    from libero_grasp_smoke import (
        BASKET_XY_SUCCESS,
        _basket_object_name,
        _object_body_id,
        _object_geom_bounds,
        _target_object_name,
    )
    from preview_scripted_demos import _apply_export_setup
    from sawyer_ik import ik_context_from_env
    from scripted_action_spaces import step_abs_eef_ik, step_env

    bench_name = {"object": "libero_object", "spatial": "libero_spatial", "long": "libero_10"}[suite]
    task_suite = benchmark.get_benchmark_dict()[bench_name]()
    task = task_suite.get_task(task_id)
    inits = task_suite.get_task_init_states(task_id)
    env = _make_env(task, "sawyer", 7, horizon=4000)
    env.reset()
    obs = env.set_init_state(inits[init_index % len(inits)])
    for _ in range(NUM_STABILIZE):
        obs, _, _, _ = env.step(LIBERO_DUMMY)
    # Match live export start pose (CLI strategy), not only the preview preset.
    obs, _ = _apply_export_setup(
        env,
        obs,
        task_id=task_id,
        suite=suite,
        ik_backend=ik_backend,
        start_pose_strategy=start_pose_strategy,
    )
    ik_ctx = ik_context_from_env(env)
    sim = env.env.robots[0].sim
    target_name = _target_object_name(env)
    try:
        target_bid = sim.model.body_name2id(target_name + "_main")
    except Exception:
        target_bid = None
    z0_target = float(sim.data.body_xpos[target_bid][2]) if target_bid is not None else None
    xy0_target = (
        np.asarray(sim.data.body_xpos[target_bid][:2], dtype=np.float64)
        if target_bid is not None
        else None
    )
    basket_bounds = None
    if suite == "object":
        try:
            basket_name = _basket_object_name(env)
            basket_bid = _object_body_id(sim, basket_name)
            basket_bounds = _object_geom_bounds(sim, basket_bid)
        except Exception:
            basket_bounds = None
    done = False
    reward = 0.0
    zmax_target = z0_target
    for fr in demo_frames:
        try:
            if action_space == "abs_eef":
                # Densified abs_eef demos are executed with the same hybrid used at eval
                # (no joint teleport while carrying).
                obs, reward, done = step_abs_eef_ik(
                    env, obs, np.asarray(fr["actions"], dtype=np.float32), ik_ctx
                )
            else:
                obs, reward, done = step_env(
                    env,
                    obs,
                    np.asarray(fr["actions"], dtype=np.float32),
                    action_space="delta_osc",
                    ik_ctx=ik_ctx,
                )
        except Exception:
            break
        if target_bid is not None:
            zmax_target = max(zmax_target, float(sim.data.body_xpos[target_bid][2]))
        if float(reward) > 0:
            env.close()
            return True
        if done:
            break
    lifted = (
        z0_target is not None
        and zmax_target is not None
        and (zmax_target - z0_target) >= min_lift_dz
    )
    placed = False
    if suite == "object" and target_bid is not None and basket_bounds is not None and xy0_target is not None:
        obj_xy = np.asarray(sim.data.body_xpos[target_bid][:2], dtype=np.float64)
        obj_z = float(sim.data.body_xpos[target_bid][2])
        basket_xy_dist = float(np.linalg.norm(obj_xy - basket_bounds["xy"]))
        left_pickup = float(np.linalg.norm(obj_xy - xy0_target)) >= 0.05
        in_basket_xy = basket_xy_dist <= BASKET_XY_SUCCESS
        above_basket_floor = obj_z >= float(basket_bounds["z_min"]) + 0.01
        placed = in_basket_xy and left_pickup and above_basket_floor
    env.close()
    if suite == "object":
        return bool(lifted and placed)
    return bool(lifted)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="object", choices=["object", "spatial", "long"])
    p.add_argument("--task-ids", nargs="+", type=int, default=[0, 1])
    p.add_argument("--max-inits", type=int, default=50, help="Max init states to try per task.")
    p.add_argument("--init-start", type=int, default=0)
    p.add_argument(
        "--motion-profile",
        default="compact",
        choices=[
            "default",
            "compact",
            "compact_osc",
            "compact_transport_heavy",
            "compact_grasp_heavy",
            "compact_place_heavy",
            "compact_grasp_contact",
        ],
    )
    p.add_argument(
        "--grasp-mode",
        default="ik_set",
        choices=["ik_osc", "ik_set", "osc"],
        help="Live controller for successful demos.",
    )
    p.add_argument(
        "--action-space",
        default="abs_eef",
        choices=["abs_eef", "delta_osc"],
        help="Dataset action encoding. abs_eef: next EEF pose (IK at replay/eval). "
        "delta_osc: use with --retrack-osc.",
    )
    p.add_argument(
        "--retrack-osc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Densify ik_set EEF trajectory and re-simulate with pose_track (legacy).",
    )
    p.add_argument(
        "--require-open-loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only save episodes whose recorded actions succeed under open-loop replay.",
    )
    p.add_argument(
        "--path-step-m",
        type=float,
        default=0.014,
        help="Open-gripper densify spacing for OSC retrack (meters).",
    )
    p.add_argument(
        "--carry-path-step-m",
        type=float,
        default=None,
        help="Closed-gripper densify spacing (meters). Default: same as --path-step-m.",
    )
    p.add_argument(
        "--post-grasp-repeat",
        type=int,
        default=0,
        help="Duplicate each closed-gripper frame this many extra times after open-loop gate.",
    )
    p.add_argument(
        "--grasp-repeat",
        type=int,
        default=0,
        help="Duplicate closed frames near the first-close pose (grasp window only).",
    )
    p.add_argument(
        "--grasp-near-m",
        type=float,
        default=0.12,
        help="Radius (m) around first-close EEF for --grasp-repeat.",
    )
    p.add_argument(
        "--approach-noise-m",
        type=float,
        default=0.0,
        help="Peak XY detour (m) on open-gripper approach before returning to path.",
    )
    p.add_argument(
        "--approach-noise-copies",
        type=int,
        default=0,
        help="Extra approach-detour demos per successful init (same live demo, different seeds).",
    )
    p.add_argument(
        "--tip-noise-m",
        type=float,
        default=0.0,
        help="Peak EEF detour (m) on late open-gripper tip (near→close) before returning.",
    )
    p.add_argument(
        "--tip-noise-copies",
        type=int,
        default=0,
        help="Extra tip-detour demos per successful init (same live demo, different seeds).",
    )
    p.add_argument(
        "--transport-noise-m",
        type=float,
        default=0.0,
        help="Peak XY detour (m) on closed-gripper transport before returning to path.",
    )
    p.add_argument(
        "--transport-noise-copies",
        type=int,
        default=0,
        help="Extra transport-detour demos per successful init (same live demo, different seeds).",
    )
    p.add_argument(
        "--action-noise-std",
        type=float,
        default=0.0,
        help="MimicGen-style Gaussian σ on delta-OSC xyz actions during retrack "
        "(normalized [-1,1] space). Gripper untouched. Recorded action is noisy.",
    )
    p.add_argument(
        "--action-ori-noise-std",
        type=float,
        default=0.0,
        help="Gaussian σ on delta-OSC orientation actions (normalized [-1,1]).",
    )
    p.add_argument(
        "--action-noise-copies",
        type=int,
        default=0,
        help="Extra action-noise demos per successful init (different seeds).",
    )
    p.add_argument(
        "--action-noise-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When action-noise copies > 0, skip the clean (no-noise) variant.",
    )
    p.add_argument(
        "--place-repeat",
        type=int,
        default=0,
        help="Duplicate closed frames in the pre-release (over-basket) window.",
    )
    p.add_argument(
        "--release-repeat",
        type=int,
        default=0,
        help="Duplicate open-gripper frames in the release tail.",
    )
    p.add_argument(
        "--release-pad-min",
        type=int,
        default=25,
        help="Minimum open-gripper frames required/padded at end of densified demos.",
    )
    p.add_argument(
        "--min-lift-dz",
        type=float,
        default=0.03,
        help="Require live lift_dz >= this (meters) before accepting a demo.",
    )
    p.add_argument(
        "--preclose-repeat",
        type=int,
        default=0,
        help="Duplicate open-gripper frames just before first close (approach tip).",
    )
    p.add_argument(
        "--preclose-frames",
        type=int,
        default=12,
        help="How many pre-close frames to boost with --preclose-repeat.",
    )
    p.add_argument("--start-pose-strategy", default="home_w30")
    p.add_argument("--ik-backend", default="mujoco")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-frames", type=int, default=80)
    p.add_argument("--output-suffix", default="_scripted_v1")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--append",
        action="store_true",
        help="Append episodes to an existing repo (do not delete task 0/1 data).",
    )
    p.add_argument(
        "--out-repo",
        default=None,
        help="Override output repo path (default: LIBERO_DATA_ROOT/libero_sawyer{suffix}).",
    )
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    _setup_paths()
    _torch_patch()

    from generate_b_demos import CONTROL_FREQ, DATASET_FPS, _setup_libero_config
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    _setup_libero_config()
    print(
        f"[export] clock aligned: dataset_fps={DATASET_FPS} control_freq={CONTROL_FREQ}",
        flush=True,
    )

    out_repo = Path(args.out_repo) if args.out_repo else DATA_ROOT / f"libero_sawyer{args.output_suffix}"
    features = {
        "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }
    if out_repo.exists() and args.append:
        dst = LeRobotDataset(str(out_repo))
        print(f"Appending to {out_repo}", flush=True)
    elif out_repo.exists() and args.overwrite:
        shutil.rmtree(out_repo)
        dst = LeRobotDataset.create(
            repo_id=str(out_repo),
            robot_type="sawyer",
            fps=DATASET_FPS,
            features=features,
            image_writer_threads=4,
            image_writer_processes=2,
        )
    elif out_repo.exists():
        raise SystemExit(f"{out_repo} exists; pass --append or --overwrite")
    else:
        dst = LeRobotDataset.create(
            repo_id=str(out_repo),
            robot_type="sawyer",
            fps=DATASET_FPS,
            features=features,
            image_writer_threads=4,
            image_writer_processes=2,
        )

    all_records: list[dict] = []
    init_indices = list(range(args.init_start, args.init_start + args.max_inits))
    for task_id in args.task_ids:
        if task_id not in SCRIPTED_TASK_PRESETS:
            raise SystemExit(f"No scripted preset for task_id={task_id}; add to SCRIPTED_TASK_PRESETS")
        recs = export_task(
            suite=args.suite,
            task_id=task_id,
            dst=dst,
            init_indices=init_indices,
            motion_profile=args.motion_profile,
            seed=args.seed,
            ik_backend=args.ik_backend,
            start_pose_strategy=args.start_pose_strategy,
            min_frames=args.min_frames,
            grasp_mode=args.grasp_mode,
            require_open_loop=args.require_open_loop,
            retrack_osc=args.retrack_osc,
            action_space=args.action_space,
            path_step_m=args.path_step_m,
            carry_path_step_m=args.carry_path_step_m,
            post_grasp_repeat=args.post_grasp_repeat,
            grasp_repeat=args.grasp_repeat,
            grasp_near_m=args.grasp_near_m,
            approach_noise_m=args.approach_noise_m,
            approach_noise_copies=args.approach_noise_copies,
            tip_noise_m=args.tip_noise_m,
            tip_noise_copies=args.tip_noise_copies,
            transport_noise_m=args.transport_noise_m,
            transport_noise_copies=args.transport_noise_copies,
            action_noise_std=args.action_noise_std,
            action_ori_noise_std=args.action_ori_noise_std,
            action_noise_copies=args.action_noise_copies,
            action_noise_only=args.action_noise_only,
            place_repeat=args.place_repeat,
            release_repeat=args.release_repeat,
            release_pad_min=args.release_pad_min,
            min_lift_dz=args.min_lift_dz,
            preclose_repeat=args.preclose_repeat,
            preclose_frames=args.preclose_frames,
        )
        all_records.extend(recs)

    meta_dir = out_repo / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "suite": args.suite,
        "task_ids": args.task_ids,
        "dataset_fps": DATASET_FPS,
        "control_freq": CONTROL_FREQ,
        "motion_profile": args.motion_profile,
        "grasp_mode": args.grasp_mode,
        "action_space": args.action_space,
        "retrack_osc": args.retrack_osc,
        "require_open_loop": args.require_open_loop,
        "path_step_m": args.path_step_m,
        "carry_path_step_m": args.carry_path_step_m,
        "post_grasp_repeat": args.post_grasp_repeat,
        "grasp_repeat": args.grasp_repeat,
        "grasp_near_m": args.grasp_near_m,
        "approach_noise_m": args.approach_noise_m,
        "approach_noise_copies": args.approach_noise_copies,
        "tip_noise_m": args.tip_noise_m,
        "tip_noise_copies": args.tip_noise_copies,
        "transport_noise_m": args.transport_noise_m,
        "transport_noise_copies": args.transport_noise_copies,
        "action_noise_std": args.action_noise_std,
        "action_ori_noise_std": args.action_ori_noise_std,
        "action_noise_copies": args.action_noise_copies,
        "action_noise_only": args.action_noise_only,
        "place_repeat": args.place_repeat,
        "release_repeat": args.release_repeat,
        "release_pad_min": args.release_pad_min,
        "min_lift_dz": args.min_lift_dz,
        "preclose_repeat": args.preclose_repeat,
        "preclose_frames": args.preclose_frames,
        "init_indices": init_indices,
        "exported_episodes": sum(1 for r in all_records if r["exported"]),
        "attempted": len(all_records),
        "open_loop_ok": sum(1 for r in all_records if r.get("open_loop_ok")),
        "records": all_records,
    }
    summary_path = meta_dir / f"scripted_export_summary_{'-'.join(map(str, args.task_ids))}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        f"\nWROTE {out_repo} episodes={summary['exported_episodes']}/{summary['attempted']}",
        flush=True,
    )
    print(f"SUMMARY {summary_path}", flush=True)


if __name__ == "__main__":
    main()
