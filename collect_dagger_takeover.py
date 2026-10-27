#!/usr/bin/env python3
"""Round-2 DAgger collector: student closed-loop -> OSC expert takeover (no reset).

Student (π0.5 / v8 policy) runs until a grasp-stall trigger, then a scripted OSC
expert continues from the *current* sim state and we record (obs, expert_action)
frames for BC. Student actions are logged for analysis but excluded from the
training export by default.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
from pathlib import Path

import imageio
import numpy as np

_EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

# LIBERO init dumps need weights_only=False on torch>=2.6
import torch

_torch_load = torch.load


def _torch_load_compat(*a, **k):
    k.setdefault("weights_only", False)
    return _torch_load(*a, **k)


torch.load = _torch_load_compat  # type: ignore

from libero.libero import benchmark  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: E402

from eval_libero_sawyer import (  # noqa: E402
    CONTROL_FREQ,
    LIBERO_DUMMY_ACTION,
    _get_libero_env,
    _maybe_sawyer_home_w30,
    _quat2axisangle,
)
from generate_b_demos import (  # noqa: E402
    DATASET_FPS,
    _pose_track_action,
    _render_pair,
    _sim_proprio_state,
)
from libero_grasp_smoke import (  # noqa: E402
    BASKET_PLACE_ABOVE,
    GRIP_CLOSE,
    GRIP_HOLD,
    GRIP_OPEN,
    GRIP_SQUEEZE,
    _basket_object_name,
    _object_body_id,
    _object_geom_bounds,
    _target_object_name,
)

PROMPT_BY_TASK = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
}

# Match export_scripted_demos.SCRIPTED_TASK_PRESETS grasp geometry.
EXPERT_GRASP_PRESETS: dict[int, dict] = {
    0: {"grasp_style": "cap", "cap_drop": 0.023, "insert_depth": 0.004, "side_x": 0.0, "side_z": 0.006},
    1: {"grasp_style": "side", "side_x": 0.022, "side_z": 0.006},
}


def _obj_top_z(env) -> float:
    sim = env.env.robots[0].sim
    bid = _object_body_id(sim, _target_object_name(env))
    return float(_object_geom_bounds(sim, bid)["z_max"])


def _grasp_target(env, *, task_id: int) -> tuple[np.ndarray, float, float]:
    """Return (grasp_xy, contact_z, descend_z)."""
    preset = EXPERT_GRASP_PRESETS.get(int(task_id), EXPERT_GRASP_PRESETS[1])
    style = preset.get("grasp_style", "side")
    obj_xy, obj_z = _obj_xy_z(env)
    if style == "cap":
        # Center grasp on can top (same as export / libero_grasp_smoke).
        grasp_xy = np.asarray(obj_xy, dtype=np.float64).copy()
        top = _obj_top_z(env)
        contact_z = float(top) - float(preset.get("cap_drop", 0.023))
        descend_z = contact_z - float(preset.get("insert_depth", 0.004))
    else:
        grasp_xy = np.asarray(obj_xy, dtype=np.float64) + np.array(
            [float(preset.get("side_x", 0.022)), 0.0], dtype=np.float64
        )
        contact_z = float(obj_z) + float(preset.get("side_z", 0.006))
        descend_z = contact_z
    return grasp_xy, contact_z, descend_z


def _xy_steps(eef_xy: np.ndarray, target_xy: np.ndarray, *, min_steps: int = 20, max_steps: int = 70) -> int:
    dist = float(np.linalg.norm(np.asarray(eef_xy, dtype=np.float64) - np.asarray(target_xy, dtype=np.float64)))
    return int(np.clip(round(dist / 0.008), min_steps, max_steps))


def _policy_element(obs, prompt: str, resize: int = 224) -> dict:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize, resize))
    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": prompt,
    }


def _frame_from_obs_action(obs, action: np.ndarray, prompt: str) -> dict:
    img, wrist = _render_pair(obs)
    return {
        "image": img,
        "wrist_image": wrist,
        "state": np.asarray(_sim_proprio_state(obs), dtype=np.float32),
        "actions": np.asarray(action, dtype=np.float32).reshape(-1),
        "task": prompt,
    }


def _obj_xy_z(env):
    sim = env.env.robots[0].sim
    bid = _object_body_id(sim, _target_object_name(env))
    pos = np.asarray(sim.data.body_xpos[bid], dtype=np.float64)
    return pos[:2].copy(), float(pos[2])


def _basket_xy_zmax(env):
    sim = env.env.robots[0].sim
    bid = _object_body_id(sim, _basket_object_name(env))
    b = _object_geom_bounds(sim, bid)
    return np.asarray(b["xy"], dtype=np.float64), float(b["z_max"])


def _chase_xyz(
    env,
    obs,
    *,
    xyz: np.ndarray,
    grip: float,
    n_steps: int,
    frames: list[dict],
    tag: str,
    prompt: str,
) -> tuple[object, bool, float]:
    """OSC-chase a Cartesian waypoint while recording expert frames."""
    start = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    end = np.asarray(xyz, dtype=np.float64)
    done = False
    reward = 0.0
    for i in range(max(1, n_steps)):
        frac = (i + 1) / max(1, n_steps)
        wp = start + (end - start) * frac
        tgt = _sim_proprio_state(obs).copy()
        tgt[0] = np.float32(wp[0])
        tgt[1] = np.float32(wp[1])
        tgt[2] = np.float32(wp[2])
        # Keep current orientation; only drive gripper separately via grip arg.
        action = _pose_track_action(obs, tgt, grip)
        frames.append(_frame_from_obs_action(obs, action, prompt))
        frames[-1]["dagger_tag"] = tag
        obs, reward, done, _ = env.step(np.asarray(action, dtype=np.float32).tolist())
        if done:
            break
    return obs, bool(done), float(reward)


def _hold_grip(env, obs, *, grip: float, n_steps: int, frames: list[dict], tag: str, prompt: str):
    done = False
    reward = 0.0
    for _ in range(max(1, n_steps)):
        tgt = _sim_proprio_state(obs).copy()
        action = _pose_track_action(obs, tgt, grip)
        frames.append(_frame_from_obs_action(obs, action, prompt))
        frames[-1]["dagger_tag"] = tag
        obs, reward, done, _ = env.step(np.asarray(action, dtype=np.float32).tolist())
        if done:
            break
    return obs, bool(done), float(reward)


def expert_takeover_osc(
    env,
    obs,
    *,
    task_id: int = 1,
    prompt: str | None = None,
    keep_student_pose: bool = False,
    resume_carry: bool = False,
) -> tuple[list[dict], dict]:
    """From the current state, finish grasp -> lift -> basket place via OSC.

    keep_student_pose=True skips the rise/clear retreat so labels stay near the
    student's failed near-contact pose (regrasp from where it got stuck).
    resume_carry=True skips regrasp and continues transport/place (post-lift fails).
    """
    prompt = prompt or PROMPT_BY_TASK.get(int(task_id), PROMPT_BY_TASK[1])
    frames: list[dict] = []
    basket_xy, basket_zmax = _basket_xy_zmax(env)
    grasp_xy, contact_z, descend_z = _grasp_target(env, task_id=task_id)
    hover_z = contact_z + 0.08
    clear_z = max(hover_z + 0.06, contact_z + 0.14)
    lift_z = contact_z + 0.18
    place_z = basket_zmax + BASKET_PLACE_ABOVE
    done = False
    reward = 0.0

    if resume_carry:
        # Already holding / lifted: keep gripper closed and go to basket.
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        carry_z = max(float(eef[2]), lift_z)
        obs, done, reward = _hold_grip(
            env, obs, grip=GRIP_HOLD, n_steps=6, frames=frames, tag="expert_hold", prompt=prompt
        )
        if done:
            return frames, {"done": True, "reward": reward, "phase": "hold", "resume_carry": True}
        basket_xy, basket_zmax = _basket_xy_zmax(env)
        place_z = basket_zmax + BASKET_PLACE_ABOVE
        transport_n = _xy_steps(eef[:2], basket_xy, min_steps=24, max_steps=80)
        obs, done, reward = _chase_xyz(
            env,
            obs,
            xyz=np.array([basket_xy[0], basket_xy[1], carry_z]),
            grip=GRIP_HOLD,
            n_steps=transport_n,
            frames=frames,
            tag="expert_transport",
            prompt=prompt,
        )
        if done:
            return frames, {"done": True, "reward": reward, "phase": "transport", "resume_carry": True}
        obs, done, reward = _chase_xyz(
            env,
            obs,
            xyz=np.array([basket_xy[0], basket_xy[1], place_z]),
            grip=GRIP_HOLD,
            n_steps=18,
            frames=frames,
            tag="expert_place",
            prompt=prompt,
        )
        if done:
            return frames, {"done": True, "reward": reward, "phase": "place", "resume_carry": True}
        obs, done, reward = _hold_grip(
            env, obs, grip=GRIP_OPEN, n_steps=18, frames=frames, tag="expert_release", prompt=prompt
        )
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        obs, done, reward = _chase_xyz(
            env,
            obs,
            xyz=np.array([eef[0], eef[1], float(eef[2]) + 0.08]),
            grip=GRIP_OPEN,
            n_steps=10,
            frames=frames,
            tag="expert_retreat",
            prompt=prompt,
        )
        obj_xy, _ = _obj_xy_z(env)
        basket_xy, _ = _basket_xy_zmax(env)
        placed = float(np.linalg.norm(obj_xy - basket_xy)) < 0.08
        return frames, {
            "done": bool(done),
            "reward": float(reward),
            "phase": "done",
            "placed_xy": placed,
            "obj_basket_xy": float(np.linalg.norm(obj_xy - basket_xy)),
            "n_expert_frames": len(frames),
            "resume_carry": True,
            "task_id": int(task_id),
        }

    if keep_student_pose:
        # Stay near the failed pose: brief reopen, slide to grasp pose, then close.
        obs, done, reward = _hold_grip(
            env, obs, grip=GRIP_OPEN, n_steps=4, frames=frames, tag="expert_reopen", prompt=prompt
        )
        if done:
            return frames, {"done": True, "reward": reward, "phase": "reopen", "keep_student_pose": True}
        grasp_xy, contact_z, descend_z = _grasp_target(env, task_id=task_id)
        lift_z = contact_z + 0.18
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        target_z = float(np.clip(eef[2], descend_z - 0.01, contact_z + 0.04))
        obs, done, reward = _chase_xyz(
            env,
            obs,
            xyz=np.array([grasp_xy[0], grasp_xy[1], target_z]),
            grip=GRIP_OPEN,
            n_steps=16,
            frames=frames,
            tag="expert_slide",
            prompt=prompt,
        )
        if done:
            return frames, {"done": True, "reward": reward, "phase": "slide", "keep_student_pose": True}
        obs, done, reward = _chase_xyz(
            env,
            obs,
            xyz=np.array([grasp_xy[0], grasp_xy[1], descend_z]),
            grip=GRIP_OPEN,
            n_steps=12,
            frames=frames,
            tag="expert_descend",
            prompt=prompt,
        )
        if done:
            return frames, {"done": True, "reward": reward, "phase": "descend", "keep_student_pose": True}
    else:
        # Long-range safe approach: open -> rise high -> XY transit at height -> hover -> descend.
        # Needed for task0 early DAgger where student is often still far from the can.
        obs, done, reward = _hold_grip(
            env, obs, grip=GRIP_OPEN, n_steps=8, frames=frames, tag="expert_open", prompt=prompt
        )
        if done:
            return frames, {"done": True, "reward": reward, "phase": "open"}
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        obs, done, reward = _chase_xyz(
            env,
            obs,
            xyz=np.array([eef[0], eef[1], max(float(eef[2]), clear_z)]),
            grip=GRIP_OPEN,
            n_steps=16,
            frames=frames,
            tag="expert_clear",
            prompt=prompt,
        )
        if done:
            return frames, {"done": True, "reward": reward, "phase": "clear"}

        grasp_xy, contact_z, descend_z = _grasp_target(env, task_id=task_id)
        hover_z = contact_z + 0.08
        clear_z = max(hover_z + 0.06, contact_z + 0.14)
        lift_z = contact_z + 0.18
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        transit_z = max(float(eef[2]), clear_z)
        xy_n = _xy_steps(eef[:2], grasp_xy)
        obs, done, reward = _chase_xyz(
            env,
            obs,
            xyz=np.array([grasp_xy[0], grasp_xy[1], transit_z]),
            grip=GRIP_OPEN,
            n_steps=xy_n,
            frames=frames,
            tag="expert_transit",
            prompt=prompt,
        )
        if done:
            return frames, {"done": True, "reward": reward, "phase": "transit"}
        obs, done, reward = _chase_xyz(
            env,
            obs,
            xyz=np.array([grasp_xy[0], grasp_xy[1], hover_z]),
            grip=GRIP_OPEN,
            n_steps=14,
            frames=frames,
            tag="expert_hover",
            prompt=prompt,
        )
        if done:
            return frames, {"done": True, "reward": reward, "phase": "hover"}
        # Refresh object before final descend (student may have bumped it earlier).
        grasp_xy, contact_z, descend_z = _grasp_target(env, task_id=task_id)
        lift_z = contact_z + 0.18
        obs, done, reward = _chase_xyz(
            env,
            obs,
            xyz=np.array([grasp_xy[0], grasp_xy[1], descend_z]),
            grip=GRIP_OPEN,
            n_steps=18,
            frames=frames,
            tag="expert_descend",
            prompt=prompt,
        )
        if done:
            return frames, {"done": True, "reward": reward, "phase": "descend"}

    # close / squeeze / hold
    obs, done, reward = _hold_grip(
        env, obs, grip=GRIP_SQUEEZE, n_steps=14, frames=frames, tag="expert_squeeze", prompt=prompt
    )
    if done:
        return frames, {"done": True, "reward": reward, "phase": "squeeze", "keep_student_pose": keep_student_pose}
    obs, done, reward = _hold_grip(
        env, obs, grip=GRIP_CLOSE, n_steps=12, frames=frames, tag="expert_close", prompt=prompt
    )
    if done:
        return frames, {"done": True, "reward": reward, "phase": "close", "keep_student_pose": keep_student_pose}
    obs, done, reward = _hold_grip(
        env, obs, grip=GRIP_HOLD, n_steps=8, frames=frames, tag="expert_hold", prompt=prompt
    )
    if done:
        return frames, {"done": True, "reward": reward, "phase": "hold", "keep_student_pose": keep_student_pose}

    # lift
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    obs, done, reward = _chase_xyz(
        env,
        obs,
        xyz=np.array([eef[0], eef[1], lift_z]),
        grip=GRIP_CLOSE,
        n_steps=22,
        frames=frames,
        tag="expert_lift",
        prompt=prompt,
    )
    if done:
        return frames, {"done": True, "reward": reward, "phase": "lift", "keep_student_pose": keep_student_pose}

    # transport + place (refresh basket; scale XY steps)
    basket_xy, basket_zmax = _basket_xy_zmax(env)
    place_z = basket_zmax + BASKET_PLACE_ABOVE
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    carry_z = max(float(eef[2]), lift_z)
    transport_n = _xy_steps(eef[:2], basket_xy, min_steps=24, max_steps=80)
    obs, done, reward = _chase_xyz(
        env,
        obs,
        xyz=np.array([basket_xy[0], basket_xy[1], carry_z]),
        grip=GRIP_CLOSE,
        n_steps=transport_n,
        frames=frames,
        tag="expert_transport",
        prompt=prompt,
    )
    if done:
        return frames, {"done": True, "reward": reward, "phase": "transport", "keep_student_pose": keep_student_pose}
    obs, done, reward = _chase_xyz(
        env,
        obs,
        xyz=np.array([basket_xy[0], basket_xy[1], place_z]),
        grip=GRIP_CLOSE,
        n_steps=18,
        frames=frames,
        tag="expert_place",
        prompt=prompt,
    )
    if done:
        return frames, {"done": True, "reward": reward, "phase": "place", "keep_student_pose": keep_student_pose}

    # release + retreat
    obs, done, reward = _hold_grip(
        env, obs, grip=GRIP_OPEN, n_steps=18, frames=frames, tag="expert_release", prompt=prompt
    )
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    obs, done, reward = _chase_xyz(
        env,
        obs,
        xyz=np.array([eef[0], eef[1], eef[2] + 0.10]),
        grip=GRIP_OPEN,
        n_steps=12,
        frames=frames,
        tag="expert_retreat",
        prompt=prompt,
    )

    obj_xy, _ = _obj_xy_z(env)
    basket_xy, _ = _basket_xy_zmax(env)
    placed = float(np.linalg.norm(obj_xy - basket_xy)) < 0.08
    return frames, {
        "done": bool(done),
        "reward": float(reward),
        "phase": "done",
        "placed_xy": placed,
        "obj_basket_xy": float(np.linalg.norm(obj_xy - basket_xy)),
        "n_expert_frames": len(frames),
        "keep_student_pose": bool(keep_student_pose),
        "task_id": int(task_id),
    }


def should_takeover(
    *,
    t: int,
    min_eef_obj: float | None,
    first_near_t: int | None,
    last_improve_t: int | None,
    first_close_t: int | None,
    grasped_frames: int,
    near_thresh: float,
    linger_steps: int,
    hard_t: int,
    approach_stall_steps: int,
    approach_active_thresh: float,
    require_close_fail: bool,
    allow_near_without_close: bool = False,
    allow_post_grasp_fail: bool = False,
    first_lift_t: int | None = None,
    min_obj_basket: float | None = None,
    last_basket_improve_t: int | None = None,
    place_near_thresh: float = 0.08,
) -> tuple[bool, str | None]:
    """Return (takeover?, reason).

    Stages (optional):
      - close_fail / near_fail_noclose: pre-lift
      - lift_no_transport / place_stall: post-lift (allow_post_grasp_fail)
    """
    # Later-stage: lifted but not progressing toward / into basket.
    if allow_post_grasp_fail and grasped_frames >= 5 and first_lift_t is not None:
        if min_obj_basket is not None and min_obj_basket <= place_near_thresh:
            return False, None  # already near place; let episode finish or succeed
        # Hovering with object, basket distance not improving.
        if (
            last_basket_improve_t is not None
            and (t - first_lift_t) >= linger_steps
            and (t - last_basket_improve_t) >= linger_steps
        ):
            return True, "lift_no_transport"
        if t >= hard_t and (min_obj_basket is None or min_obj_basket > place_near_thresh):
            return True, "place_fail_hard"
        # Still in post-grasp watch window; do not fall through to pre-grasp rules.
        return False, None

    if grasped_frames >= 5:
        return False, None

    if require_close_fail:
        # Only intervene after near + close attempt, still no lift.
        if (
            min_eef_obj is not None
            and min_eef_obj <= near_thresh
            and first_near_t is not None
            and first_close_t is not None
            and (t - first_close_t) >= linger_steps
        ):
            return True, "close_fail"
        if t >= hard_t and first_close_t is not None and min_eef_obj is not None and min_eef_obj <= near_thresh:
            return True, "close_fail_hard"
        if allow_near_without_close:
            # Optional: near linger even if student never closed (not task1-identical).
            if (
                min_eef_obj is not None
                and min_eef_obj <= near_thresh
                and first_near_t is not None
                and first_close_t is None
                and (t - first_near_t) >= linger_steps
            ):
                return True, "near_fail_noclose"
            if t >= hard_t and min_eef_obj is not None and min_eef_obj <= near_thresh and first_close_t is None:
                return True, "near_fail_hard"
        return False, None

    # Early: approached somewhat, then eef-object distance stopped improving.
    if (
        approach_stall_steps > 0
        and min_eef_obj is not None
        and last_improve_t is not None
        and min_eef_obj <= approach_active_thresh
        and (t - last_improve_t) >= approach_stall_steps
    ):
        return True, "approach_stall"
    # Near object but not grasping for linger_steps (shorter = earlier).
    if min_eef_obj is not None and min_eef_obj <= near_thresh and first_near_t is not None:
        if t - first_near_t >= linger_steps:
            return True, "near_linger"
    if t >= hard_t and grasped_frames < 5:
        return True, "hard_t"
    return False, None


def run_episode(
    *,
    env,
    client,
    inits,
    init_index: int,
    prompt: str,
    task_id: int,
    max_student_steps: int,
    num_steps_wait: int,
    near_thresh: float,
    linger_steps: int,
    hard_t: int,
    replan_steps: int,
    approach_stall_steps: int,
    approach_active_thresh: float,
    require_close_fail: bool,
    close_cmd_thresh: float,
    keep_student_pose: bool,
    allow_near_without_close: bool,
    allow_post_grasp_fail: bool = False,
    place_near_thresh: float = 0.08,
    video_path: Path | None,
) -> dict:
    env.reset()
    obs = env.set_init_state(inits[init_index % len(inits)])
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    obs = _maybe_sawyer_home_w30(
        env,
        obs,
        task_id=int(task_id),
        task_suite_name="libero_object",
        robot_variant="sawyer",
        enabled=True,
        ik_backend="mujoco",
    )

    action_plan: collections.deque = collections.deque()
    student_frames: list[dict] = []
    replay_images: list[np.ndarray] = []
    min_eef_obj = None
    first_near_t = None
    last_improve_t = None
    first_close_t = None
    grasped_frames = 0
    first_lift_t = None
    min_obj_basket = None
    last_basket_improve_t = None
    z0 = _obj_xy_z(env)[1]
    zmax = z0
    takeover = False
    takeover_reason = None
    takeover_t = None
    student_done = False
    student_reward = 0.0

    for t in range(max_student_steps):
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        replay_images.append(img)
        if not action_plan:
            chunk = client.infer(_policy_element(obs, prompt))["actions"]
            action_plan.extend(np.asarray(chunk[:replan_steps], dtype=np.float32))
        action = np.asarray(action_plan.popleft(), dtype=np.float32)
        # Record student for diagnostics only (not exported to train by default).
        fr = _frame_from_obs_action(obs, action, prompt)
        fr["dagger_tag"] = "student"
        student_frames.append(fr)
        obs, student_reward, student_done, _ = env.step(action.tolist())

        obj_xy, obj_z = _obj_xy_z(env)
        zmax = max(zmax, obj_z)
        eef_xy = np.asarray(obs["robot0_eef_pos"][:2], dtype=np.float64)
        d_eo = float(np.linalg.norm(eef_xy - obj_xy))
        basket_xy, _ = _basket_xy_zmax(env)
        d_ob = float(np.linalg.norm(obj_xy - basket_xy))
        if min_eef_obj is None or d_eo < (min_eef_obj - 1e-4):
            min_eef_obj = d_eo
            last_improve_t = t
        elif min_eef_obj is None:
            min_eef_obj = d_eo
            last_improve_t = t
        if d_eo <= near_thresh and first_near_t is None:
            first_near_t = t
        # Close attempt while near the object (cmd > thresh).
        if (
            first_close_t is None
            and first_near_t is not None
            and d_eo <= near_thresh
            and float(action[-1]) >= close_cmd_thresh
        ):
            first_close_t = t
        if (obj_z - z0) > 0.05 and d_eo <= 0.08:
            grasped_frames += 1
            if first_lift_t is None and grasped_frames >= 5:
                first_lift_t = t
                min_obj_basket = d_ob
                last_basket_improve_t = t
        if first_lift_t is not None:
            if min_obj_basket is None or d_ob < (min_obj_basket - 1e-4):
                min_obj_basket = d_ob
                last_basket_improve_t = t

        if student_done and student_reward > 0:
            break
        do_take, reason = should_takeover(
            t=t,
            min_eef_obj=min_eef_obj,
            first_near_t=first_near_t,
            last_improve_t=last_improve_t,
            first_close_t=first_close_t,
            grasped_frames=grasped_frames,
            near_thresh=near_thresh,
            linger_steps=linger_steps,
            hard_t=hard_t,
            approach_stall_steps=approach_stall_steps,
            approach_active_thresh=approach_active_thresh,
            require_close_fail=require_close_fail,
            allow_near_without_close=allow_near_without_close,
            allow_post_grasp_fail=allow_post_grasp_fail,
            first_lift_t=first_lift_t,
            min_obj_basket=min_obj_basket,
            last_basket_improve_t=last_basket_improve_t,
            place_near_thresh=place_near_thresh,
        )
        if do_take:
            takeover = True
            takeover_t = t
            takeover_reason = reason
            break
    else:
        takeover_t = max_student_steps - 1
        if require_close_fail:
            near_ok = (
                grasped_frames < 5
                and min_eef_obj is not None
                and min_eef_obj <= near_thresh
            )
            if near_ok and first_close_t is not None:
                takeover = True
                takeover_reason = "close_fail_max"
            elif near_ok and first_close_t is None and allow_near_without_close:
                takeover = True
                takeover_reason = "near_fail_max"
            elif (
                allow_post_grasp_fail
                and grasped_frames >= 5
                and (min_obj_basket is None or min_obj_basket > place_near_thresh)
            ):
                takeover = True
                takeover_reason = "place_fail_max"
            else:
                takeover = False
                takeover_reason = None
        else:
            takeover = grasped_frames < 5
            takeover_reason = "max_steps" if takeover else None
            if (
                not takeover
                and allow_post_grasp_fail
                and grasped_frames >= 5
                and (min_obj_basket is None or min_obj_basket > place_near_thresh)
            ):
                takeover = True
                takeover_reason = "place_fail_max"

    expert_frames: list[dict] = []
    expert_info: dict = {"skipped": True}
    if takeover and not (student_done and student_reward > 0):
        resume_carry = bool(
            takeover_reason
            and str(takeover_reason).startswith(("lift_no_transport", "place_fail"))
        )
        expert_frames, expert_info = expert_takeover_osc(
            env,
            obs,
            task_id=int(task_id),
            prompt=prompt,
            keep_student_pose=bool(keep_student_pose),
            resume_carry=resume_carry,
        )
        for fr in expert_frames:
            replay_images.append(np.ascontiguousarray(fr["image"]))

    if video_path is not None and replay_images:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=DATASET_FPS)

    return {
        "init_index": init_index,
        "task_id": int(task_id),
        "takeover": bool(takeover),
        "takeover_t": int(takeover_t) if takeover else None,
        "takeover_reason": takeover_reason,
        "first_close_t": None if first_close_t is None else int(first_close_t),
        "first_lift_t": None if first_lift_t is None else int(first_lift_t),
        "min_obj_basket": None if min_obj_basket is None else float(min_obj_basket),
        "keep_student_pose": bool(keep_student_pose),
        "min_eef_obj": None if min_eef_obj is None else float(min_eef_obj),
        "grasped_frames_student": int(grasped_frames),
        "student_success": bool(student_done and student_reward > 0),
        "n_student_frames": len(student_frames),
        "n_expert_frames": len(expert_frames),
        "expert_info": expert_info,
        "student_frames": student_frames,
        "expert_frames": expert_frames,
    }


def _upsample_expert_head(frames: list[dict], head_frames: int, head_repeat: int) -> list[dict]:
    """Repeat the first head_frames (recovery-from-deviation) to thicken those labels."""
    if not frames or head_frames <= 0 or head_repeat <= 1:
        return frames
    n = min(head_frames, len(frames))
    head = frames[:n]
    rest = frames[n:]
    out: list[dict] = []
    for _ in range(head_repeat):
        out.extend(head)
    out.extend(rest)
    return out


_GRASP_TAGS = frozenset(
    {
        "expert_hover",
        "expert_transit",
        "expert_slide",
        "expert_reopen",
        "expert_descend",
        "expert_squeeze",
        "expert_close",
        "expert_hold",
        "expert_lift",
    }
)


def _upsample_grasp_phase(frames: list[dict], grasp_repeat: int) -> list[dict]:
    """Prepend repeated grasp-contact expert frames (hover→lift) to thicken grasp labels."""
    if not frames or grasp_repeat <= 1:
        return frames
    grasp = [fr for fr in frames if fr.get("dagger_tag") in _GRASP_TAGS]
    if not grasp:
        return frames
    out: list[dict] = []
    for _ in range(grasp_repeat - 1):
        out.extend(grasp)
    out.extend(frames)
    return out


def _save_lerobot(
    repo: Path,
    episodes: list[dict],
    *,
    include_student: bool,
    overwrite: bool,
    head_frames: int = 0,
    head_repeat: int = 1,
    grasp_repeat: int = 1,
):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if repo.exists() and overwrite:
        shutil.rmtree(repo)
    features = {
        "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }
    appending = repo.exists()
    if appending:
        dst = LeRobotDataset(str(repo))
        dst.start_image_writer(num_processes=1, num_threads=4)
    else:
        dst = LeRobotDataset.create(
            repo_id=str(repo),
            robot_type="sawyer",
            fps=DATASET_FPS,
            features=features,
            image_writer_threads=4,
            image_writer_processes=1,
        )

    saved = 0
    for ep in episodes:
        frames = list(ep["expert_frames"])
        if include_student:
            frames = list(ep["student_frames"]) + frames
        # Only keep episodes that produced expert frames (takeover path).
        if not ep["expert_frames"]:
            continue
        frames = _upsample_expert_head(frames, head_frames, head_repeat)
        frames = _upsample_grasp_phase(frames, grasp_repeat)
        for fr in frames:
            dst.add_frame(
                {
                    "image": fr["image"],
                    "wrist_image": fr["wrist_image"],
                    "state": fr["state"],
                    "actions": fr["actions"],
                    "task": fr["task"],
                }
            )
        dst.save_episode()
        saved += 1
    if appending:
        dst.stop_image_writer()
    return saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8086)
    p.add_argument("--task-id", type=int, default=1, choices=[0, 1])
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--init-start", type=int, default=0)
    p.add_argument("--max-student-steps", type=int, default=140)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--near-thresh", type=float, default=0.08)
    p.add_argument("--linger-steps", type=int, default=40)
    p.add_argument("--hard-t", type=int, default=110)
    p.add_argument(
        "--approach-stall-steps",
        type=int,
        default=0,
        help="If >0, takeover when eef-object distance stops improving for this many steps.",
    )
    p.add_argument(
        "--approach-active-thresh",
        type=float,
        default=0.20,
        help="Only arm approach-stall after min eef-object distance is below this.",
    )
    p.add_argument(
        "--head-frames",
        type=int,
        default=0,
        help="Upsample first N expert frames (recovery-from-deviation) when saving.",
    )
    p.add_argument(
        "--head-repeat",
        type=int,
        default=1,
        help="How many times to emit the expert head before the rest of the episode.",
    )
    p.add_argument(
        "--grasp-repeat",
        type=int,
        default=1,
        help="How many times to emit grasp-phase expert tags (hover→lift) before the full episode.",
    )
    p.add_argument(
        "--require-close-fail",
        action="store_true",
        help="Takeover only after near + close-cmd attempt without lift.",
    )
    p.add_argument(
        "--allow-near-without-close",
        action="store_true",
        help="With --require-close-fail, also takeover when near but student never closed.",
    )
    p.add_argument(
        "--allow-post-grasp-fail",
        action="store_true",
        help="Also takeover after lift when transport/place stalls (lift_no_transport / place_fail).",
    )
    p.add_argument(
        "--place-near-thresh",
        type=float,
        default=0.08,
        help="XY obj-basket distance treated as near-place (post-grasp success).",
    )
    p.add_argument(
        "--close-cmd-thresh",
        type=float,
        default=0.2,
        help="Action gripper threshold counting as a close attempt.",
    )
    p.add_argument(
        "--keep-student-pose",
        action="store_true",
        help="Expert continues from student pose (no rise/clear retreat).",
    )
    p.add_argument(
        "--require-expert-success",
        action="store_true",
        help="Only save episodes where expert placed the object in the basket (placed_xy).",
    )
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-repo", type=Path, required=True)
    p.add_argument("--video-dir", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--include-student-frames", action="store_true")
    p.add_argument("--control-freq", type=int, default=CONTROL_FREQ)
    args = p.parse_args()

    bench = benchmark.get_benchmark_dict()["libero_object"]()
    task_id = int(args.task_id)
    task = bench.get_task(task_id)
    inits = bench.get_task_init_states(task_id)
    env, lang = _get_libero_env(task, 256, args.seed, "sawyer", control_freq=args.control_freq)
    prompt = str(lang).strip() or PROMPT_BY_TASK.get(task_id, PROMPT_BY_TASK[1])

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    # Warmup ping
    print(
        f"[dagger] connected policy={args.host}:{args.port} task_id={task_id} prompt={prompt!r}",
        flush=True,
    )

    meta = []
    episodes = []
    for i in range(args.num_episodes):
        init_index = args.init_start + i
        video_path = None
        if args.video_dir is not None:
            video_path = Path(args.video_dir) / f"dagger_init{init_index:03d}.mp4"
        print(f"[dagger] ep={i} init={init_index}", flush=True)
        ep = run_episode(
            env=env,
            client=client,
            inits=inits,
            init_index=init_index,
            prompt=prompt,
            task_id=task_id,
            max_student_steps=args.max_student_steps,
            num_steps_wait=args.num_steps_wait,
            near_thresh=args.near_thresh,
            linger_steps=args.linger_steps,
            hard_t=args.hard_t,
            replan_steps=args.replan_steps,
            approach_stall_steps=args.approach_stall_steps,
            approach_active_thresh=args.approach_active_thresh,
            require_close_fail=bool(args.require_close_fail),
            close_cmd_thresh=float(args.close_cmd_thresh),
            keep_student_pose=bool(args.keep_student_pose),
            allow_near_without_close=bool(args.allow_near_without_close),
            allow_post_grasp_fail=bool(args.allow_post_grasp_fail),
            place_near_thresh=float(args.place_near_thresh),
            video_path=video_path,
        )
        summary = {k: v for k, v in ep.items() if k not in ("student_frames", "expert_frames")}
        print(f"  -> {json.dumps(summary, default=str)}", flush=True)
        meta.append(summary)
        expert_ok = bool((ep.get("expert_info") or {}).get("placed_xy"))
        if args.require_expert_success and not expert_ok:
            print("  -> skip save (expert placed_xy=false)", flush=True)
            continue
        if not ep.get("takeover") and not ep.get("student_success"):
            print("  -> skip save (no takeover)", flush=True)
            continue
        episodes.append(ep)

    env.close()
    n_saved = _save_lerobot(
        Path(args.out_repo),
        episodes,
        include_student=bool(args.include_student_frames),
        overwrite=bool(args.overwrite),
        head_frames=int(args.head_frames),
        head_repeat=int(args.head_repeat),
        grasp_repeat=int(args.grasp_repeat),
    )
    meta_path = Path(args.out_repo) / "meta" / "dagger_takeover_summary.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    prev: list[dict] = []
    if meta_path.is_file() and not args.overwrite:
        try:
            prev = list(json.loads(meta_path.read_text()).get("episodes") or [])
        except Exception:
            prev = []
    all_meta = prev + meta
    meta_path.write_text(
        json.dumps(
            {
                "episodes": all_meta,
                "saved_episodes": n_saved,
                "saved_episodes_this_run": n_saved,
                "total_summary_episodes": len(all_meta),
            },
            indent=2,
        )
    )
    print(f"[dagger] saved_episodes={n_saved} repo={args.out_repo}", flush=True)
    print(f"[dagger] summary={meta_path} total_meta={len(all_meta)}", flush=True)


if __name__ == "__main__":
    main()
