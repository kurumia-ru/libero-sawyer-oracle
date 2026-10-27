#!/usr/bin/env python3
"""LIBERO grasp smoke: descend on target object, close gripper, lift; check object motion.

Compares Panda vs SawyerCustom on the same init_state. Optional start-pose strategy for Sawyer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OPENPI_EVAL = Path(os.environ.get("OPENPI_EVAL_ROOT", str(Path.home() / "openpi_ws" / "openpi_eval")))
GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0
GRIP_HOLD = 0.0  # neutral gripper cmd: no extra closing torque
GRIP_SQUEEZE = 0.35  # peak squeeze cmd; tapers to GRIP_HOLD over squeeze phase
LIFTED_DZ_MIN = 0.03
REACH_XY_MAX = 0.06
REACH_Z_MAX = 0.08
# ik_set rim/side: longer squeeze + hold at rim before lift.
SQUEEZE_STEPS = 80
SETTLE_STEPS = 0  # skip: long hold at rim knocked the can; squeeze -> lift directly
CLOSE_STEPS = 50  # light rim pinch; squeeze finishes the grasp
GRIP_CLOSE_RIM = 0.55  # partial close at rim (not full torque)
LIFT_STEPS = 40  # slow lift
RIM_LIFT_ABOVE_TOP = 0.18  # ~18 cm above object top so carry clears basket rim
SQUEEZE_ACTIVE_FRAC = 0.35  # first 35% of squeeze closes; rest GRIP_HOLD (zero torque)
# After lift: move to basket, lower, release.
PLACE_TRANSPORT_STEPS = 60
PLACE_DESCEND_STEPS = 60
PLACE_RELEASE_STEPS = 50
PLACE_RELEASE_HOLD_STEPS = 25  # hold closed over basket before opening
PLACE_RETREAT_STEPS = 20
BASKET_PLACE_ABOVE = 0.035  # Lower placement for reliable in-basket predicate.
BASKET_XY_SUCCESS = 0.10  # object xy within this of basket center counts as placed

MOTION_PROFILE_CHOICES = (
    "default",
    "compact",
    "compact_osc",
    "compact_transport_heavy",
    "compact_grasp_heavy",
    "compact_place_heavy",
    "compact_grasp_contact",
)


@dataclass(frozen=True)
class MotionProfile:
    approach_high: int
    hover_open: int
    descend: int
    use_bite: bool
    close: int
    squeeze: int
    squeeze_active_frac: float
    contact_hold: int
    lift: int
    carry_hold: int
    transport_xy: int
    place_descend: int
    release: int
    release_hold: int
    retreat: int
    transport_ik_guide: bool
    start_warmup_steps: int | None


def _motion_profile(name: str) -> MotionProfile:
    if name == "compact":
        # Start from task1's video5s pose; retain a short open-gripper approach.
        return MotionProfile(
            approach_high=3,
            hover_open=1,
            descend=4,
            use_bite=False,
            close=14,
            squeeze=16,
            squeeze_active_frac=0.85,
            contact_hold=0,
            lift=24,
            carry_hold=0,
            transport_xy=24,
            place_descend=20,
            release=22,
            release_hold=6,
            retreat=8,
            transport_ik_guide=False,
            start_warmup_steps=0,
        )
    if name == "compact_grasp_heavy":
        # Thicker close/squeeze/contact/lift; transport stays compact-sized.
        # contact_hold uses GRIP_HOLD (see phase build) so we don't over-torque.
        return MotionProfile(
            approach_high=3,
            hover_open=1,
            descend=6,
            use_bite=False,
            close=22,
            squeeze=28,
            squeeze_active_frac=0.85,
            contact_hold=12,
            lift=32,
            carry_hold=0,
            transport_xy=24,
            place_descend=22,
            release=22,
            release_hold=6,
            retreat=8,
            transport_ik_guide=False,
            start_warmup_steps=0,
        )
    if name == "compact_transport_heavy":
        # Same grasp as compact; thicker post-grasp carry for BC transport learning.
        return MotionProfile(
            approach_high=3,
            hover_open=1,
            descend=4,
            use_bite=False,
            close=14,
            squeeze=16,
            squeeze_active_frac=0.85,
            contact_hold=0,
            lift=28,
            carry_hold=12,
            transport_xy=48,
            place_descend=28,
            release=22,
            release_hold=6,
            retreat=8,
            transport_ik_guide=False,
            start_warmup_steps=0,
        )
    if name == "compact_place_heavy":
        # Compact grasp/carry; thicker hover-over-basket + open + retreat for place BC.
        return MotionProfile(
            approach_high=3,
            hover_open=1,
            descend=4,
            use_bite=False,
            close=14,
            squeeze=16,
            squeeze_active_frac=0.85,
            contact_hold=0,
            lift=24,
            carry_hold=4,
            transport_xy=28,
            place_descend=36,
            release=40,
            release_hold=20,
            retreat=16,
            transport_ik_guide=False,
            start_warmup_steps=0,
        )
    if name == "compact_grasp_contact":
        # v6: thick close/squeeze/contact/lift + place_heavy release.
        # Targets contact precision (majority osc_v5 failure) while keeping place demos.
        return MotionProfile(
            approach_high=3,
            hover_open=1,
            descend=8,
            use_bite=False,
            close=24,
            squeeze=32,
            squeeze_active_frac=0.85,
            contact_hold=16,
            lift=36,
            carry_hold=4,
            transport_xy=28,
            place_descend=36,
            release=40,
            release_hold=20,
            retreat=16,
            transport_ik_guide=False,
            start_warmup_steps=0,
        )
    if name == "compact_osc":
        # Same grasp style as compact, but enough OSC steps to reach without teleport.
        return MotionProfile(
            approach_high=80,
            hover_open=25,
            descend=80,
            use_bite=False,
            close=50,
            squeeze=45,
            squeeze_active_frac=0.55,
            contact_hold=0,
            lift=45,
            carry_hold=0,
            transport_xy=70,
            place_descend=50,
            release=35,
            release_hold=12,
            retreat=18,
            transport_ik_guide=False,
            start_warmup_steps=None,
        )
    return MotionProfile(
        approach_high=15,
        hover_open=50,
        descend=10,
        use_bite=True,
        close=CLOSE_STEPS,
        squeeze=SQUEEZE_STEPS,
        squeeze_active_frac=SQUEEZE_ACTIVE_FRAC,
        contact_hold=0,
        lift=LIFT_STEPS,
        carry_hold=0,
        transport_xy=PLACE_TRANSPORT_STEPS,
        place_descend=PLACE_DESCEND_STEPS,
        release=PLACE_RELEASE_STEPS,
        release_hold=PLACE_RELEASE_HOLD_STEPS,
        retreat=PLACE_RETREAT_STEPS,
        transport_ik_guide=False,
        start_warmup_steps=None,
    )


@dataclass(frozen=True)
class Phase:
    name: str
    steps: int
    z_offset: float  # relative to object z at episode start
    gripper: float


DEFAULT_PHASES = (
    Phase("approach_xy", 40, 0.18, GRIP_OPEN),
    Phase("descend", 80, 0.04, GRIP_OPEN),
    Phase("close", 50, 0.04, GRIP_CLOSE),
    Phase("lift", 50, 0.22, GRIP_CLOSE),
)

# IK-guided phases built via _phases_for_mode().
# ik_set: joint teleport (succeeds live, NOT open-loop replayable).
# ik_osc: same targets via OSC pose_track only (replayable / learnable).
GRASP_MODE_CHOICES = ("osc", "ik_set", "ik_osc")
GRASP_STYLE_CHOICES = ("center", "rim", "cap", "side")


def _ik_guided(grasp_mode: str) -> bool:
    return grasp_mode in {"ik_set", "ik_osc"}


def _ik_teleport(grasp_mode: str) -> bool:
    return grasp_mode == "ik_set"
POSE_SWEEP_STRATEGIES = ("sawyer_home", "home_w30", "ik_pos_w20", "ik_full_w20")

DEFAULT_SIDE_X = 0.030  # meters: EEF offset +x from object (lateral approach)
DEFAULT_SIDE_Z = 0.018  # meters: EEF height above object body z at grasp
DEFAULT_CAP_DROP = 0.023  # top_z - contact_z; Franka ep813 ~2.3 cm below top
VIDEO_FPS = 20  # smoother than 10 when IK targets move via OSC


def _object_geom_bounds(sim, body_id: int) -> dict:
    """World-frame AABB of object geoms (for rim / top targeting)."""
    xs: list[float] = []
    ys: list[float] = []
    zs_lo: list[float] = []
    zs_hi: list[float] = []
    for gid in range(sim.model.ngeom):
        if int(sim.model.geom_bodyid[gid]) != int(body_id):
            continue
        p = sim.data.geom_xpos[gid]
        s = sim.model.geom_size[gid]
        xs.extend([float(p[0] - s[0]), float(p[0] + s[0])])
        ys.extend([float(p[1] - s[1]), float(p[1] + s[1])])
        zs_lo.append(float(p[2] - s[2]))
        zs_hi.append(float(p[2] + s[2]))
    if not zs_hi:
        p = sim.data.body_xpos[body_id]
        return {
            "xy": np.array(p[:2], dtype=np.float64),
            "z_min": float(p[2]),
            "z_max": float(p[2]),
            "xy_radius": 0.03,
        }
    return {
        "xy": np.array([(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2], dtype=np.float64),
        "z_min": min(zs_lo),
        "z_max": max(zs_hi),
        "xy_radius": max(max(xs) - min(xs), max(ys) - min(ys)) / 2,
    }


def _phase_target_z(
    phase_name: str,
    *,
    grasp_style: str,
    obj_z_now: float,
    bounds: dict,
    z_offset: float,
    rim_inset: float,
    bite_depth: float,
    insert_depth: float,
    side_z: float,
    cap_drop: float,
    cap_skip_bite: bool = False,
) -> float:
    """World-frame EEF target z for a phase."""
    if grasp_style == "side":
        # Lateral pinch at can/box mid-height (body z), not top-down at top_z.
        # close/squeeze sit slightly below descend so fingers seat before pinch.
        if phase_name == "approach_high":
            return obj_z_now + 0.18
        if phase_name == "hover_open":
            return obj_z_now + 0.10
        if phase_name == "descend":
            return obj_z_now + side_z + 0.008
        if phase_name == "bite":
            return obj_z_now + side_z
        if phase_name in {"close", "squeeze", "settle", "contact_hold"}:
            return obj_z_now + side_z - 0.006
        if phase_name == "lift":
            return obj_z_now + 0.20
    if grasp_style == "rim":
        top = bounds["z_max"]
        if phase_name == "approach_high":
            return top + 0.15
        if phase_name == "hover_open":
            return top + 0.07
        if phase_name == "descend":
            return top + rim_inset - 0.5 * insert_depth
        if phase_name == "bite":
            # Deep insert: fingers straddle below rim opening.
            return top + rim_inset - bite_depth - insert_depth
        if phase_name in {"close", "squeeze", "settle", "contact_hold"}:
            # Retract to rim height before pinch; settle holds there before lift.
            return top + rim_inset
        if phase_name == "lift":
            return top + RIM_LIFT_ABOVE_TOP
    if grasp_style == "cap":
        # Top-down pose; finger contact on cylinder sidewall (Franka-like), not rim only.
        top = bounds["z_max"]
        contact_z = top - cap_drop
        if phase_name == "approach_high":
            return top + 0.15
        if phase_name == "hover_open":
            return top + 0.07
        if phase_name == "descend":
            if cap_skip_bite:
                return contact_z - insert_depth
            return contact_z + 0.012
        if phase_name == "bite":
            return contact_z - insert_depth
        if phase_name in {"close", "squeeze", "settle", "contact_hold"}:
            return contact_z
        if phase_name == "lift":
            return top + RIM_LIFT_ABOVE_TOP
    return obj_z_now + z_offset


def _ik_proprio_target(
    ik_ctx,
    obs: dict,
    target_pos: np.ndarray,
    gripper: float,
    *,
    ik_backend: str,
) -> tuple[np.ndarray, dict]:
    """Solve IK and return 8D proprio target for OSC (no joint teleport)."""
    from sawyer_ik import fk_proprio_at_joints, solve_sawyer_ik

    q_current = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
    target_wxyz = _eef_wxyz_from_obs(obs)
    q_target = solve_sawyer_ik(
        ik_ctx,
        target_pos,
        target_wxyz,
        q_current,
        backend=ik_backend,  # type: ignore[arg-type]
        pos_only=True,
    )
    proprio = fk_proprio_at_joints(ik_ctx, q_target, gripper)
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    err = float(np.linalg.norm(eef - np.asarray(target_pos, dtype=np.float64)))
    return proprio, {"ik_pos_err_m": err, "eef_z": float(proprio[2])}


def _eef_wxyz_from_obs(obs: dict) -> np.ndarray:
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def _hold_pose_action(obs: dict, gripper: float, _sim_proprio_state, _pose_track_action) -> np.ndarray:
    cur = _sim_proprio_state(obs)
    return _pose_track_action(obs, cur, gripper)


def _gripper_only_action(gripper: float) -> np.ndarray:
    """OSC action: hold arm, drive gripper only (reliable close/open on Sawyer)."""
    return np.concatenate(
        [
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.array([gripper], dtype=np.float32),
        ]
    )


def _force_sawyer_gripper_full_open(
    env,
    obs: dict,
    *,
    settle_steps: int = 0,
) -> tuple[dict, dict]:
    """Set finger joints to XML limits (skip OSC settle — it pulls fingers off limits)."""
    from sawyer_ik import apply_gripper_qpos, gripper_aperture_from_obs

    before = gripper_aperture_from_obs(obs)
    obs = apply_gripper_qpos(env)
    after = gripper_aperture_from_obs(obs)
    return obs, {"aperture_before": before, "aperture_after": after}


_PLACE_PHASES = frozenset({"transport_xy", "place_descend", "release", "retreat"})


def _append_place_phases(phases: tuple[Phase, ...], profile: MotionProfile) -> tuple[Phase, ...]:
    return (
        *phases,
        Phase("carry_hold", profile.carry_hold, 0.0, GRIP_CLOSE),
        Phase("transport_xy", profile.transport_xy, 0.0, GRIP_CLOSE),
        Phase("place_descend", profile.place_descend, 0.0, GRIP_CLOSE),
        Phase("release", profile.release, 0.0, GRIP_OPEN),
        Phase("retreat", profile.retreat, 0.0, GRIP_OPEN),
    )


def _phases_for_mode(
    grasp_mode: str,
    grasp_z: float,
    grasp_style: str = "cap",
    *,
    place_to_basket: bool = False,
    motion_profile: str = "default",
) -> tuple[Phase, ...]:
    profile = _motion_profile(motion_profile)
    if not _ik_guided(grasp_mode):
        phases = DEFAULT_PHASES
        return _append_place_phases(phases, profile) if place_to_basket else phases
    if grasp_style in {"rim", "side", "cap"}:
        # Side pinch needs a firm close; rim/cap keep a lighter first pinch.
        close_grip = GRIP_CLOSE if grasp_style == "side" else GRIP_CLOSE_RIM
        tail = (
            Phase("close", profile.close, 0.0, close_grip),
            Phase("squeeze", profile.squeeze, 0.0, GRIP_SQUEEZE),
        )
        if SETTLE_STEPS > 0:
            tail = (*tail, Phase("settle", SETTLE_STEPS, 0.0, GRIP_HOLD))
        pre_close = (
            Phase("approach_high", profile.approach_high, 0.25, GRIP_OPEN),
            Phase("hover_open", profile.hover_open, 0.12, GRIP_OPEN),
            Phase("descend", profile.descend, 0.05, GRIP_OPEN),
        )
        if profile.use_bite:
            pre_close = (*pre_close, Phase("bite", 10, 0.0, GRIP_OPEN))
        post_squeeze = ()
        if profile.contact_hold > 0:
            # Hold seated grasp without extra closing torque (avoids knock/slip).
            post_squeeze = (Phase("contact_hold", profile.contact_hold, 0.0, GRIP_HOLD),)
        phases = (
            *pre_close,
            *tail,
            *post_squeeze,
            Phase("lift", profile.lift, 0.22, GRIP_CLOSE),
        )
    else:
        tail = (
            Phase("close", profile.close, grasp_z, GRIP_CLOSE_RIM),
            Phase("squeeze", profile.squeeze, grasp_z, GRIP_SQUEEZE),
        )
        if SETTLE_STEPS > 0:
            tail = (*tail, Phase("settle", SETTLE_STEPS, grasp_z, GRIP_HOLD))
        post_squeeze = ()
        if profile.contact_hold > 0:
            post_squeeze = (Phase("contact_hold", profile.contact_hold, grasp_z, GRIP_HOLD),)
        phases = (
            Phase("approach_high", profile.approach_high, 0.25, GRIP_OPEN),
            Phase("hover_open", profile.hover_open, 0.12, GRIP_OPEN),
            Phase("descend", profile.descend, grasp_z, GRIP_OPEN),
            *tail,
            *post_squeeze,
            Phase("lift", profile.lift, 0.22, GRIP_CLOSE),
        )
    if place_to_basket:
        phases = _append_place_phases(phases, profile)
    # Drop zero-length phases (e.g. compact skips high approach / hover).
    return tuple(ph for ph in phases if ph.steps > 0)


# ik_set: reposition via IK; close re-IKs to rim height after deep bite (settle holds fixed pose).
_IK_VERTICAL_PHASES = frozenset({"approach_high", "hover_open", "descend", "bite", "close"})
_IK_GRIPPER_DRIVE_PHASES = frozenset({"hover_open", "close", "squeeze", "settle", "contact_hold"})


def _ik_set_eef(
    env,
    ik_ctx,
    obs: dict,
    target_pos: np.ndarray,
    *,
    ik_backend: str,
    pos_only: bool = True,
    target_wxyz: np.ndarray | None = None,
) -> tuple[dict, dict]:
    from sawyer_ik import apply_arm_qpos, solve_sawyer_ik

    q_current = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
    if target_wxyz is None:
        target_wxyz = _eef_wxyz_from_obs(obs)
    q_target = solve_sawyer_ik(
        ik_ctx,
        target_pos,
        target_wxyz,
        q_current,
        backend=ik_backend,  # type: ignore[arg-type]
        pos_only=pos_only,
    )
    obs = apply_arm_qpos(env, ik_ctx, q_target)
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    err = float(np.linalg.norm(eef - np.asarray(target_pos, dtype=np.float64)))
    return obs, {"ik_pos_err_m": err, "eef_z": float(eef[2])}


def _torch_patch() -> None:
    import torch

    _orig = torch.load

    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)

    torch.load = _load  # type: ignore


def _setup_paths() -> None:
    openpi_libero = Path(os.environ.get("OPENPI_LIBERO_ROOT", str(Path.home() / "openpi_ws" / "openpi" / "third_party" / "libero")))
    sys.path.insert(0, str(openpi_libero))
    sys.path.insert(0, str(OPENPI_EVAL))


def _target_object_name(env) -> str:
    objs = list(env.env.objects_dict.keys())
    for name in objs:
        if "basket" not in name.lower():
            return name
    if objs:
        return objs[0]
    raise RuntimeError("no objects in scene")


def _basket_object_name(env) -> str:
    for name in env.env.objects_dict:
        if "basket" in name.lower():
            return name
    raise RuntimeError("no basket in scene")


def _object_body_id(sim, object_name: str) -> int:
    for suffix in ("_main", ""):
        candidate = f"{object_name}{suffix}"
        try:
            return sim.model.body_name2id(candidate)
        except Exception:
            continue
    raise KeyError(f"body for object {object_name!r} not found")


def _render_frame(env, obs: dict, *, video_camera: str = "agentview") -> np.ndarray:
    """Capture a video frame. Use galleryview/frontview to avoid arm occluding agentview."""
    if video_camera == "agentview":
        return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1], dtype=np.uint8)
    sim = env.env.sim
    img = np.asarray(sim.render(width=256, height=256, camera_name=video_camera))
    return np.ascontiguousarray(img[::-1], dtype=np.uint8)


def _record_demo_frame(
    demo_frames: list[dict],
    obs: dict,
    action: np.ndarray,
    *,
    _render_pair,
    _sim_proprio_state,
) -> None:
    img, wrist = _render_pair(obs)
    q = np.asarray(obs.get("robot0_joint_pos", np.zeros(7)), dtype=np.float32).reshape(-1)[:7]
    demo_frames.append(
        {
            "image": img,
            "wrist_image": wrist,
            "state": _sim_proprio_state(obs),
            "actions": np.asarray(action, dtype=np.float32),
            "arm_qpos": q.copy(),
        }
    )


def run_grasp_attempt(
    *,
    robot: str,
    suite: str,
    task_id: int,
    init_index: int,
    start_pose_strategy: str | None,
    phases: tuple[Phase, ...],
    record_video: bool,
    ik_backend: str = "mujoco",
    grasp_mode: str = "osc",
    grasp_z: float = 0.05,
    grasp_style: str = "center",
    rim_inset: float = 0.004,
    bite_depth: float = 0.003,
    insert_depth: float = 0.0,
    force_gripper_open: bool = False,
    cap_drop: float = DEFAULT_CAP_DROP,
    side_x: float | None = None,
    side_z: float | None = None,
    video_camera: str = "agentview",
    place_to_basket: bool = False,
    motion_profile: str = "default",
    record_demo: bool = False,
) -> dict:
    from generate_b_demos import (
        LIBERO_DUMMY,
        NUM_STABILIZE,
        _make_env,
        _pose_track_action,
        _render_pair,
        _setup_libero_config,
        _sim_proprio_state,
        apply_start_pose_strategy,
    )
    from libero.libero import benchmark

    _setup_libero_config()
    profile = _motion_profile(motion_profile)
    bench_name = {"object": "libero_object", "spatial": "libero_spatial", "long": "libero_10"}[suite]
    task_suite = benchmark.get_benchmark_dict()[bench_name]()
    task = task_suite.get_task(task_id)
    init_states = task_suite.get_task_init_states(task_id)

    if robot == "panda":
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            robots=["Panda"],
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
        )
        robot_label = "Panda"
    else:
        env = _make_env(task, robot, seed=7)
        robot_label = f"SawyerCustom ({robot})"

    env.reset()
    obs = env.set_init_state(init_states[init_index % len(init_states)])
    for _ in range(NUM_STABILIZE):
        obs, _, _, _ = env.step(LIBERO_DUMMY)

    demo_frames: list[dict] = []

    def _maybe_record(action: np.ndarray) -> None:
        if record_demo:
            _record_demo_frame(
                demo_frames,
                obs,
                action,
                _render_pair=_render_pair,
                _sim_proprio_state=_sim_proprio_state,
            )

    start_info = None
    if start_pose_strategy and robot != "panda":
        from sawyer_ik import ik_context_from_env

        frame0 = _franka_frame0_for_task(suite, task_id)
        ik_ctx = ik_context_from_env(env)
        obs, start_info = apply_start_pose_strategy(
            env,
            ik_ctx,
            obs,
            frame0,
            start_pose_strategy,
            ik_backend=ik_backend,
            warmup_steps_override=profile.start_warmup_steps,
            on_step=(lambda o, a: _record_demo_frame(
                demo_frames, o, a, _render_pair=_render_pair, _sim_proprio_state=_sim_proprio_state
            ))
            if record_demo
            else None,
        )
        if start_pose_strategy == "video5s":
            # The captured arm pose was during grasp approach. Start with fully
            # open fingers so the object is not contacted by a closed gripper.
            from sawyer_ik import apply_gripper_qpos

            obs = apply_gripper_qpos(env)

    sim = env.env.robots[0].sim
    obj_name = _target_object_name(env)
    obj_body = _object_body_id(sim, obj_name)
    basket_body = None
    basket_name = None
    if place_to_basket:
        basket_name = _basket_object_name(env)
        basket_body = _object_body_id(sim, basket_name)

    def obj_pos() -> np.ndarray:
        return np.array(sim.data.body_xpos[obj_body], dtype=np.float64)

    def obj_z() -> float:
        return float(obj_pos()[2])

    def obj_tilt_delta() -> float:
        q = np.array(sim.data.body_xquat[obj_body], dtype=np.float64)
        return float(np.linalg.norm(q - obj_quat0))

    obj_quat0 = np.array(sim.data.body_xquat[obj_body], dtype=np.float64).copy()
    obj_z0 = obj_z()
    obj_xy0 = obj_pos()[:2].copy()
    eef0 = np.array(obs["robot0_eef_pos"], dtype=np.float64)

    ik_ctx = None
    if robot != "panda" and _ik_guided(grasp_mode):
        from sawyer_ik import ik_context_from_env

        ik_ctx = ik_context_from_env(env)

    sx = DEFAULT_SIDE_X if side_x is None else side_x
    sz = DEFAULT_SIDE_Z if side_z is None else side_z

    target = np.zeros(8, dtype=np.float32)
    frames: list[np.ndarray] = []

    if record_video:
        frames.append(_render_frame(env, obs, video_camera=video_camera))

    gripper_force_info = None
    if force_gripper_open and robot != "panda":
        obs, gripper_force_info = _force_sawyer_gripper_full_open(env, obs)
        if record_video:
            frames.append(_render_frame(env, obs, video_camera=video_camera))

    phase_log: list[dict] = []
    for ph in phases:
        bounds = _object_geom_bounds(sim, obj_body)
        obj_now = obj_pos()
        obj_xy = obj_now[:2]
        basket_bounds = None
        if ph.name in _PLACE_PHASES:
            assert basket_body is not None
            basket_bounds = _object_geom_bounds(sim, basket_body)
            target_xy = basket_bounds["xy"]
        elif grasp_style == "side":
            target_xy = obj_xy + np.array([sx, 0.0], dtype=np.float64)
        elif grasp_style in {"rim", "cap"}:
            target_xy = bounds["xy"]
        else:
            target_xy = obj_xy
        obj_z_now = obj_z()
        if force_gripper_open and robot != "panda" and ph.name == "hover_open":
            obs, hover_force = _force_sawyer_gripper_full_open(env, obs)
            if gripper_force_info is None:
                gripper_force_info = {}
            gripper_force_info["hover_open"] = hover_force
            if record_video:
                frames.append(_render_frame(env, obs, video_camera=video_camera))
        target_z = _phase_target_z(
            ph.name,
            grasp_style=grasp_style,
            obj_z_now=obj_z_now,
            bounds=bounds,
            z_offset=ph.z_offset,
            rim_inset=rim_inset,
            bite_depth=bite_depth,
            insert_depth=insert_depth,
            side_z=sz,
            cap_drop=cap_drop,
            cap_skip_bite=grasp_style == "cap" and not profile.use_bite,
        )
        if ph.name in _PLACE_PHASES and basket_bounds is not None:
            target_z = float(basket_bounds["z_max"] + BASKET_PLACE_ABOVE)
        target[:2] = target_xy.astype(np.float32)
        target[2] = target_z
        eef_before = np.array(obs["robot0_eef_pos"], dtype=np.float64)
        tilt_before = obj_tilt_delta()
        ik_info = None
        phase_osc_target = None
        ik_target_pos = None
        if _ik_guided(grasp_mode) and ph.name in _IK_VERTICAL_PHASES:
            ik_target_pos = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float64)
        phase_hold_state = None
        lift_start_state = None
        lift_z0: float | None = None
        place_carry_z: float | None = None
        place_start_xy: np.ndarray | None = None
        place_basket_xy: np.ndarray | None = None
        place_z_target: float | None = None
        place_hold_state: np.ndarray | None = None
        if _ik_guided(grasp_mode) and ph.name in {"squeeze", "settle", "contact_hold"}:
            phase_hold_state = _sim_proprio_state(obs).copy()
        if _ik_guided(grasp_mode) and ph.name == "lift":
            lift_start_state = _sim_proprio_state(obs).copy()
            lift_z0 = float(lift_start_state[2])
        if ph.name == "carry_hold":
            # Hold closed at post-lift pose so demos clearly teach "carry before move".
            place_hold_state = _sim_proprio_state(obs).copy()
            place_carry_z = float(place_hold_state[2])
        elif ph.name == "transport_xy":
            eef = np.array(obs["robot0_eef_pos"], dtype=np.float64)
            place_carry_z = float(eef[2])
            place_start_xy = eef[:2].copy()
            assert basket_bounds is not None
            place_basket_xy = basket_bounds["xy"].copy()
            place_z_target = float(basket_bounds["z_max"] + BASKET_PLACE_ABOVE)
        elif ph.name == "place_descend":
            place_hold_state = _sim_proprio_state(obs).copy()
            place_carry_z = float(place_hold_state[2])
            assert basket_bounds is not None
            place_basket_xy = basket_bounds["xy"].copy()
            place_z_target = float(basket_bounds["z_max"] + BASKET_PLACE_ABOVE)
        elif ph.name in {"release", "retreat"}:
            place_hold_state = _sim_proprio_state(obs).copy()
            place_carry_z = float(place_hold_state[2])
            assert basket_bounds is not None
            place_basket_xy = basket_bounds["xy"].copy()
        for step_i in range(ph.steps):
            if force_gripper_open and robot != "panda" and ph.gripper == GRIP_OPEN:
                from sawyer_ik import apply_gripper_qpos

                obs = apply_gripper_qpos(env)
            if ph.name == "carry_hold" and place_hold_state is not None:
                action = _hold_pose_action(obs, GRIP_CLOSE, _sim_proprio_state, _pose_track_action)
            elif (
                ph.name == "transport_xy"
                and _ik_guided(grasp_mode)
                and profile.transport_ik_guide
                and place_carry_z is not None
                and place_start_xy is not None
                and place_basket_xy is not None
            ):
                frac = (step_i + 1) / ph.steps
                xy = place_start_xy + (place_basket_xy - place_start_xy) * frac
                ik_pos = np.array([xy[0], xy[1], place_carry_z], dtype=np.float64)
                proprio_tgt, ik_step = _ik_proprio_target(
                    ik_ctx, obs, ik_pos, GRIP_CLOSE, ik_backend=ik_backend
                )
                if ik_info is None:
                    ik_info = ik_step
                action = _pose_track_action(obs, proprio_tgt, GRIP_CLOSE)
            elif ph.name == "transport_xy" and place_carry_z is not None and place_start_xy is not None:
                frac = (step_i + 1) / ph.steps
                xy = place_start_xy + (place_basket_xy - place_start_xy) * frac  # type: ignore[operator]
                tgt = _sim_proprio_state(obs).copy()
                tgt[0] = np.float32(xy[0])
                tgt[1] = np.float32(xy[1])
                tgt[2] = np.float32(place_carry_z)
                action = _pose_track_action(obs, tgt, GRIP_CLOSE)
            elif (
                ph.name == "place_descend"
                and place_hold_state is not None
                and place_carry_z is not None
                and place_z_target is not None
                and place_basket_xy is not None
            ):
                frac = (step_i + 1) / ph.steps
                z = place_carry_z + (place_z_target - place_carry_z) * frac
                tgt = place_hold_state.copy()
                tgt[0] = np.float32(place_basket_xy[0])
                tgt[1] = np.float32(place_basket_xy[1])
                tgt[2] = np.float32(z)
                action = _pose_track_action(obs, tgt, GRIP_CLOSE)
            elif ph.name == "release" and place_hold_state is not None:
                if step_i < profile.release_hold:
                    action = _hold_pose_action(obs, GRIP_CLOSE, _sim_proprio_state, _pose_track_action)
                else:
                    # Force-open fingers on Sawyer so demos clearly teach release
                    # (OSC alone often leaves a weak open signal).
                    if robot != "panda" and step_i == profile.release_hold:
                        from sawyer_ik import apply_gripper_qpos

                        obs = apply_gripper_qpos(env)
                    action = _gripper_only_action(GRIP_OPEN)
            elif ph.name == "retreat" and place_hold_state is not None:
                frac = (step_i + 1) / ph.steps
                tgt = place_hold_state.copy()
                tgt[2] = np.float32(float(place_carry_z) + 0.10 * frac)
                action = _pose_track_action(obs, tgt, GRIP_OPEN)
            elif (
                _ik_teleport(grasp_mode)
                and ik_target_pos is not None
                and ph.name in _IK_VERTICAL_PHASES
                and step_i == 0
            ):
                obs, ik_info = _ik_set_eef(env, ik_ctx, obs, ik_target_pos, ik_backend=ik_backend)
                phase_osc_target = _sim_proprio_state(obs).copy()
                if force_gripper_open and robot != "panda" and ph.gripper == GRIP_OPEN:
                    from sawyer_ik import apply_gripper_qpos

                    obs = apply_gripper_qpos(env)
                    phase_osc_target = _sim_proprio_state(obs).copy()
                action = _hold_pose_action(obs, ph.gripper, _sim_proprio_state, _pose_track_action)
            elif (
                grasp_mode == "ik_osc"
                and ik_target_pos is not None
                and ph.name in _IK_VERTICAL_PHASES
            ):
                # Continuous OSC toward Cartesian target; keep current orientation
                # so position budget is not spent fighting IK orientation mismatch.
                grip_cmd = ph.gripper
                if force_gripper_open and robot != "panda" and ph.name == "close":
                    grip_cmd = GRIP_CLOSE_RIM
                elif force_gripper_open and robot != "panda" and ph.gripper == GRIP_OPEN:
                    grip_cmd = 0.0
                eef_now = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                end_pos = np.asarray(ik_target_pos, dtype=np.float64)
                # Approach/hover/descend: linear waypoints. close/bite: lock onto final
                # target every step so we actually seat before gripping.
                if ph.name in {"close", "bite"} or (step_i + 1) / ph.steps >= 0.7:
                    wp = end_pos
                else:
                    frac = (step_i + 1) / ph.steps
                    wp = eef_before + (end_pos - eef_before) * frac
                tgt = _sim_proprio_state(obs).copy()
                tgt[0] = np.float32(wp[0])
                tgt[1] = np.float32(wp[1])
                tgt[2] = np.float32(wp[2])
                phase_osc_target = tgt.copy()
                ik_info = {
                    "ik_pos_err_m": float(np.linalg.norm(eef_now - end_pos)),
                    "eef_z": float(eef_now[2]),
                }
                # Once seated at the grasp pose, drive gripper without fighting arm OSC.
                pos_err = float(np.linalg.norm(eef_now - end_pos))
                z_err = abs(float(eef_now[2] - end_pos[2]))
                if ph.name == "close" and pos_err < 0.012 and z_err < 0.004:
                    action = _gripper_only_action(grip_cmd)
                else:
                    action = _pose_track_action(obs, tgt, grip_cmd)
            elif _ik_guided(grasp_mode) and ph.name == "lift" and lift_start_state is not None and lift_z0 is not None:
                bounds_now = _object_geom_bounds(sim, obj_body)
                z_end = bounds_now["z_max"] + RIM_LIFT_ABOVE_TOP
                frac = (step_i + 1) / ph.steps
                z_step = lift_z0 + (z_end - lift_z0) * frac
                lift_tgt = lift_start_state.copy()
                lift_tgt[2] = np.float32(z_step)
                action = _pose_track_action(obs, lift_tgt, GRIP_CLOSE)
            elif (
                _ik_teleport(grasp_mode)
                and phase_osc_target is not None
                and ph.name in _IK_VERTICAL_PHASES
                and step_i > 0
            ):
                if force_gripper_open and robot != "panda" and ph.name == "close":
                    action = _gripper_only_action(GRIP_CLOSE_RIM)
                elif force_gripper_open and robot != "panda" and ph.gripper == GRIP_OPEN:
                    action = _pose_track_action(obs, phase_osc_target, 0.0)
                else:
                    action = _pose_track_action(obs, phase_osc_target, ph.gripper)
            elif _ik_guided(grasp_mode) and ph.name in _IK_GRIPPER_DRIVE_PHASES:
                if force_gripper_open and robot != "panda" and ph.gripper == GRIP_OPEN:
                    action = _hold_pose_action(obs, 0.0, _sim_proprio_state, _pose_track_action)
                elif force_gripper_open and robot != "panda" and ph.name == "close":
                    action = _gripper_only_action(GRIP_CLOSE_RIM)
                else:
                    grip_cmd = ph.gripper
                    if ph.name == "squeeze":
                        active_steps = max(1, int(ph.steps * profile.squeeze_active_frac))
                        # Keep only a short final close so lift can start sooner.
                        tail_close = 4
                        if step_i >= ph.steps - tail_close:
                            grip_cmd = GRIP_CLOSE
                        elif step_i >= active_steps:
                            grip_cmd = GRIP_HOLD
                        else:
                            t = 1.0 - step_i / active_steps
                            grip_cmd = GRIP_HOLD + (GRIP_SQUEEZE - GRIP_HOLD) * t
                    elif ph.name == "contact_hold":
                        # Maintain seated fingers; re-assert a light close for the last few steps.
                        grip_cmd = GRIP_CLOSE if step_i >= ph.steps - 3 else GRIP_HOLD
                    if phase_hold_state is not None:
                        action = _pose_track_action(obs, phase_hold_state, grip_cmd)
                    else:
                        action = _hold_pose_action(obs, grip_cmd, _sim_proprio_state, _pose_track_action)
            elif _ik_guided(grasp_mode):
                if force_gripper_open and robot != "panda" and ph.gripper == GRIP_OPEN:
                    action = _hold_pose_action(obs, 0.0, _sim_proprio_state, _pose_track_action)
                else:
                    action = _hold_pose_action(obs, ph.gripper, _sim_proprio_state, _pose_track_action)
            else:
                action = _pose_track_action(obs, target, ph.gripper)
            obs, _, _, _ = env.step(action.tolist())
            _maybe_record(action)
            if record_video:
                frames.append(_render_frame(env, obs, video_camera=video_camera))
        eef_after = np.array(obs["robot0_eef_pos"], dtype=np.float64)
        g = obs["robot0_gripper_qpos"]
        ref_xy = target_xy if ph.name in _PLACE_PHASES else obj_now[:2]
        entry = {
            "phase": ph.name,
            "target_z": float(target_z),
            "target_xy": [float(target_xy[0]), float(target_xy[1])],
            "object_top_z": float(bounds["z_max"]),
            "eef_z": float(eef_after[2]),
            "xy_dist": float(np.linalg.norm(eef_after[:2] - ref_xy)),
            "obj_z": obj_z_now,
            "obj_z_end": float(obj_z()),
            "obj_tilt_delta": float(obj_tilt_delta()),
            "obj_tilt_step": float(obj_tilt_delta() - tilt_before),
            "aperture": float(g[0] - g[1]),
            "eef_delta_z": float(eef_after[2] - eef_before[2]),
            "grasp_mode": grasp_mode,
        }
        if ph.name in _PLACE_PHASES:
            obj_xy_end = obj_pos()[:2]
            entry["obj_xy_dist_to_basket"] = float(np.linalg.norm(obj_xy_end - target_xy))
            entry["obj_xy_dist_from_start"] = float(np.linalg.norm(obj_xy_end - obj_xy0))
        if ik_info is not None:
            entry.update(ik_info)
        if grasp_mode == "ik_set":
            entry["settle_steps"] = ph.steps
        phase_log.append(entry)

    obj_z_final = obj_z()
    obj_xy_final = obj_pos()[:2]
    close_phase = next(p for p in phase_log if p["phase"] == "close")
    squeeze_phase = next((p for p in phase_log if p["phase"] == "squeeze"), close_phase)
    lift_phase = next((p for p in phase_log if p["phase"] == "lift"), None)
    if lift_phase is not None:
        lift_dz = float(lift_phase["obj_z_end"] - obj_z0)
        lifted = lift_dz >= LIFTED_DZ_MIN
    else:
        lift_dz = float(obj_z_final - obj_z0)
        lifted = lift_dz >= LIFTED_DZ_MIN
    reached = close_phase["xy_dist"] <= REACH_XY_MAX and (close_phase["eef_z"] - obj_z0) <= REACH_Z_MAX
    gripper_closed = squeeze_phase["aperture"] < 0.03

    placed = False
    basket_xy_dist = None
    if place_to_basket and basket_body is not None:
        basket_bounds = _object_geom_bounds(sim, basket_body)
        basket_xy = basket_bounds["xy"]
        basket_xy_dist = float(np.linalg.norm(obj_xy_final - basket_xy))
        left_pickup = float(np.linalg.norm(obj_xy_final - obj_xy0)) >= 0.05
        in_basket_xy = basket_xy_dist <= BASKET_XY_SUCCESS
        above_basket_floor = obj_z_final >= basket_bounds["z_min"] + 0.01
        placed = in_basket_xy and left_pickup and above_basket_floor

    passed = lifted and reached
    if place_to_basket:
        passed = passed and placed

    result = {
        "robot": robot_label,
        "suite": suite,
        "task_id": task_id,
        "task": task.language,
        "object": obj_name,
        "basket": basket_name,
        "place_to_basket": place_to_basket,
        "motion_profile": motion_profile,
        "sawyer_base_z": float(os.environ.get("SAWYER_TABLE_BASE_Z", "0")) if robot != "panda" else 0.0,
        "grasp_mode": grasp_mode,
        "grasp_style": grasp_style,
        "grasp_z_offset": grasp_z,
        "rim_inset": rim_inset,
        "bite_depth": bite_depth,
        "insert_depth": insert_depth,
        "cap_drop": cap_drop if grasp_style == "cap" else None,
        "force_gripper_open": force_gripper_open,
        "gripper_force_info": gripper_force_info,
        "side_x": sx if grasp_style == "side" else None,
        "side_z": sz if grasp_style == "side" else None,
        "video_camera": video_camera,
        "start_pose_strategy": start_pose_strategy,
        "start_info": start_info,
        "initial_eef": eef0.tolist(),
        "initial_obj_z": obj_z0,
        "final_obj_z": obj_z_final,
        "obj_dz": obj_z_final - obj_z0,
        "lift_dz": lift_dz,
        "obj_tilt_final": obj_tilt_delta(),
        "obj_knocked": obj_tilt_delta() > 0.15,
        "lifted": lifted,
        "reached_at_close": reached,
        "gripper_closed": gripper_closed,
        "placed": placed,
        "basket_xy_dist": basket_xy_dist,
        "passed": passed,
        "phases": phase_log,
        "frames": frames,
    }
    if record_demo:
        result["demo_frames"] = demo_frames
    env.close()
    return result


def _franka_frame0_for_task(suite: str, task_id: int) -> dict:
    import pyarrow.parquet as pq
    from generate_b_demos import SUITE_TO_BENCH
    from libero.libero import benchmark

    data_root = Path(os.environ.get("LIBERO_DATA_ROOT", str(Path.home() / "openpi_data")))
    repo = data_root / "libero"
    bench_name = SUITE_TO_BENCH[suite]
    prompt = benchmark.get_benchmark_dict()[bench_name]().get_task(task_id).language.strip()
    meta_path = repo / "meta/episodes.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)

    for ep_idx, line in enumerate(meta_path.read_text().splitlines()):
        if not line.strip():
            continue
        ep_meta = json.loads(line)
        if ep_meta["tasks"][0].strip() != prompt:
            continue
        chunk = ep_idx // 1000
        pq_path = repo / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
        if not pq_path.is_file():
            hits = sorted(repo.glob(f"data/chunk-*/episode_{ep_idx:06d}.parquet"))
            if not hits:
                continue
            pq_path = hits[0]
        t = pq.read_table(pq_path)
        return {
            "state": np.asarray(t["state"][0].as_py(), dtype=np.float32),
            "actions": np.asarray(t["actions"][0].as_py(), dtype=np.float32),
        }
    raise RuntimeError(f"no source episode for task_id={task_id} suite={suite} prompt={prompt!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="object", choices=["object", "spatial", "long"])
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--init-index", type=int, default=0)
    p.add_argument("--robots", nargs="+", default=["panda", "sawyer"])
    p.add_argument("--start-pose-strategy", default=None)
    p.add_argument("--sawyer-start-pose-strategy", default="home_w30")
    p.add_argument("--grasp-mode", default="osc", choices=list(GRASP_MODE_CHOICES))
    p.add_argument(
        "--grasp-style",
        default="cap",
        choices=list(GRASP_STYLE_CHOICES),
        help="center: body z. rim: top-down at top_z. cap: top-down sidewall contact. side: lateral pinch.",
    )
    p.add_argument(
        "--grasp-z",
        type=float,
        default=0.05,
        help="Grasp height offset above object body z for center style (meters).",
    )
    p.add_argument(
        "--rim-inset",
        type=float,
        default=0.004,
        help="Rim style: hover/descend height = top_z + rim_inset (meters).",
    )
    p.add_argument(
        "--bite-depth",
        type=float,
        default=0.003,
        help="Rim style: bite/close height = top_z + rim_inset - bite_depth (meters).",
    )
    p.add_argument(
        "--cap-drop",
        type=float,
        default=DEFAULT_CAP_DROP,
        help="Cap style: contact z = top_z - cap_drop (meters). Franka ep813 ~0.023.",
    )
    p.add_argument(
        "--insert-depth",
        type=float,
        default=0.004,
        help="Rim style: extra descent below bite (meters); lowers descend/bite/close z.",
    )
    p.add_argument(
        "--force-gripper-open",
        action="store_true",
        help="Sawyer: set Rethink gripper finger joints to XML limits before grasp.",
    )
    p.add_argument(
        "--side-x",
        type=float,
        default=None,
        help="Side style: EEF +x offset from object center (meters). Default 0.030.",
    )
    p.add_argument(
        "--side-z",
        type=float,
        default=None,
        help="Side style: EEF height = object body z + side_z (meters). Default 0.018.",
    )
    p.add_argument(
        "--sweep-side",
        action="store_true",
        help="Sweep side_x for sawyer side-pinch grasp.",
    )
    p.add_argument(
        "--sweep-rim",
        action="store_true",
        help="Sweep rim_inset x insert_depth x grasp_style (rim/side/center).",
    )
    p.add_argument(
        "--rim-inset-values",
        nargs="+",
        type=float,
        default=[0.0, 0.004, 0.008],
    )
    p.add_argument(
        "--insert-depth-values",
        nargs="+",
        type=float,
        default=[0.002, 0.004],
    )
    p.add_argument(
        "--side-x-values",
        nargs="+",
        type=float,
        default=[0.018, 0.023, 0.029, 0.035, 0.042],
    )
    p.add_argument(
        "--sweep-pose",
        action="store_true",
        help="Sweep start-pose strategies x grasp-mode (osc vs ik_set) for sawyer.",
    )
    p.add_argument(
        "--pose-strategies",
        nargs="+",
        default=list(POSE_SWEEP_STRATEGIES),
    )
    p.add_argument(
        "--compare-all",
        action="store_true",
        help="Run panda, sawyer (panda init), sawyer (home_w30).",
    )
    p.add_argument(
        "--sweep-base-z",
        action="store_true",
        help="Sweep SAWYER_TABLE_BASE_Z for sawyer+home_w30 (0, -0.10, -0.15, -0.20).",
    )
    p.add_argument(
        "--base-z-values",
        nargs="+",
        type=float,
        default=[0.0, -0.10, -0.15, -0.20],
    )
    p.add_argument(
        "--video-camera",
        default="agentview",
        choices=("agentview", "frontview", "galleryview", "birdview"),
        help="Fixed camera for mp4 recording (galleryview avoids arm blocking the view).",
    )
    p.add_argument(
        "--place-to-basket",
        action="store_true",
        help="After lift: transport to basket, lower, release, retreat.",
    )
    p.add_argument(
        "--motion-profile",
        default="compact",
        choices=list(MOTION_PROFILE_CHOICES),
        help="default: conservative step counts. compact: fewer holds / skips cap bite.",
    )
    p.add_argument("--out-dir", default=str(OPENPI_EVAL / "grasp_smoke"))
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--ik-backend", default="mujoco")
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    _setup_paths()
    _torch_patch()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    grasp_modes_by_label: dict[str, str] = {}
    side_x_by_label: dict[str, float] = {}
    rim_inset_by_label: dict[str, float] = {}
    insert_depth_by_label: dict[str, float] = {}
    grasp_style_by_label: dict[str, str] = {}
    if args.sweep_pose:
        run_specs = []
        grasp_modes_by_label = {}
        for strat in args.pose_strategies:
            for mode in GRASP_MODE_CHOICES:
                label = f"sawyer_{strat}_{mode}"
                run_specs.append(("sawyer", strat, label))
                grasp_modes_by_label[label] = mode
        base_z_by_label = {}
    elif args.sweep_base_z:
        run_specs = [
            ("sawyer", "home_w30", f"sawyer_basez_{z:+.2f}".replace("+", "p").replace("-", "m"))
            for z in args.base_z_values
        ]
        base_z_by_label = {
            spec[2]: z for z, spec in zip(args.base_z_values, run_specs, strict=True)
        }
    elif args.sweep_rim:
        run_specs = []
        for style in ("rim", "side", "center"):
            if style == "rim":
                for inset in args.rim_inset_values:
                    for ins in args.insert_depth_values:
                        label = f"sawyer_{style}_ri{inset:.3f}_id{ins:.3f}".replace(".", "p").replace("-", "m")
                        run_specs.append(("sawyer", "home_w30", label))
                        rim_inset_by_label[label] = inset
                        insert_depth_by_label[label] = ins
                        grasp_style_by_label[label] = style
            else:
                label = f"sawyer_{style}"
                run_specs.append(("sawyer", "home_w30", label))
                grasp_style_by_label[label] = style
        base_z_by_label = {}
        grasp_modes_by_label = {spec[2]: "ik_set" for spec in run_specs}
    elif args.sweep_side:
        run_specs = [
            ("sawyer", "home_w30", f"sawyer_side_x_{x:.3f}".replace(".", "p"))
            for x in args.side_x_values
        ]
        base_z_by_label = {}
        grasp_modes_by_label = {spec[2]: "ik_set" for spec in run_specs}
        side_x_by_label = {spec[2]: x for x, spec in zip(args.side_x_values, run_specs, strict=True)}
    elif args.compare_all:
        run_specs = [
            ("panda", None, "panda"),
            ("sawyer", None, "sawyer_panda_init"),
            ("sawyer", "home_w30", "sawyer_home_w30"),
        ]
        base_z_by_label = {}
        grasp_modes_by_label = {}
    else:
        run_specs = []
        base_z_by_label = {}
        grasp_modes_by_label = {}
        side_x_by_label = {}
        for robot in args.robots:
            strat = args.start_pose_strategy
            if robot == "sawyer" and strat is None:
                strat = args.sawyer_start_pose_strategy
            if robot == "panda" or strat in (None, "none", ""):
                strat = None
            label = robot if strat is None else f"{robot}_{strat}"
            if robot == "sawyer":
                label = f"{label}_{args.grasp_mode}"
            run_specs.append((robot, strat, label))
            if robot == "sawyer":
                grasp_modes_by_label[label] = args.grasp_mode

    for robot, strat, label in run_specs:
        if label in base_z_by_label:
            os.environ["SAWYER_TABLE_BASE_Z"] = str(base_z_by_label[label])
        grasp_mode = grasp_modes_by_label.get(label, args.grasp_mode)
        run_side_x = side_x_by_label.get(label, args.side_x)
        run_grasp_style = grasp_style_by_label.get(
            label,
            "side" if args.sweep_side or args.grasp_style == "side" else args.grasp_style,
        )
        run_rim_inset = rim_inset_by_label.get(label, args.rim_inset)
        run_insert_depth = insert_depth_by_label.get(label, args.insert_depth)
        print(
            f"[grasp] {label} strategy={strat} mode={grasp_mode} style={run_grasp_style} "
            f"rim_inset={run_rim_inset} insert={run_insert_depth} "
            f"base_z={os.environ.get('SAWYER_TABLE_BASE_Z', '0')}",
            flush=True,
        )
        res = run_grasp_attempt(
            robot=robot,
            suite=args.suite,
            task_id=args.task_id,
            init_index=args.init_index,
            start_pose_strategy=strat,
            phases=_phases_for_mode(
                grasp_mode,
                args.grasp_z,
                run_grasp_style,
                place_to_basket=args.place_to_basket,
                motion_profile=args.motion_profile,
            ),
            record_video=not args.no_video,
            ik_backend=args.ik_backend,
            grasp_mode=grasp_mode,
            grasp_z=args.grasp_z,
            grasp_style=run_grasp_style,
            rim_inset=run_rim_inset,
            bite_depth=args.bite_depth,
            insert_depth=run_insert_depth,
            cap_drop=args.cap_drop,
            force_gripper_open=args.force_gripper_open
            or (robot == "sawyer" and run_grasp_style in ("rim", "cap")),
            side_x=run_side_x,
            side_z=args.side_z,
            video_camera=args.video_camera,
            place_to_basket=args.place_to_basket,
            motion_profile=args.motion_profile,
        )
        res["run_label"] = label
        frames = res.pop("frames")
        if frames and not args.no_video:
            import imageio

            vid = out_dir / f"grasp_{label}_task{args.task_id}.mp4"
            imageio.mimwrite(vid, frames, fps=VIDEO_FPS, codec="libx264", quality=8, macro_block_size=1)
            res["video"] = str(vid)

        runs.append(res)
        status = "LIFTED" if res["lifted"] else "no lift"
        reach = "REACH" if res["reached_at_close"] else "no reach"
        close_ph = next(p for p in res["phases"] if p["phase"] == "close")
        gap_z = close_ph["eef_z"] - res["initial_obj_z"]
        lift_dz = res.get("lift_dz", res["obj_dz"])
        place_msg = ""
        if args.place_to_basket:
            place_msg = f" placed={res.get('placed')} basket_xy={res.get('basket_xy_dist', 'n/a')}"
        print(
            f"  {label}: {status} lift_dz={lift_dz:+.4f} final_dz={res['obj_dz']:+.4f} | {reach} "
            f"close_eef_z={close_ph['eef_z']:.3f} gap={gap_z:.3f} "
            f"xy={close_ph['xy_dist']:.3f} grip_closed={res.get('gripper_closed')} mode={grasp_mode}"
            f"{place_msg}",
            flush=True,
        )

    if args.sweep_pose:
        print("\n=== POSE SWEEP ===", flush=True)
        for r in runs:
            cp = next(p for p in r["phases"] if p["phase"] == "close")
            dp = next(p for p in r["phases"] if p["phase"] == "descend")
            gap = cp["eef_z"] - r["initial_obj_z"]
            print(
                f"  {r['run_label']}: close_z={cp['eef_z']:.3f} gap={gap:.3f} "
                f"descend_ik_err={dp.get('ik_pos_err_m', 'n/a')} "
                f"lifted={r['lifted']} reach={r['reached_at_close']}",
                flush=True,
            )

    if args.sweep_rim:
        print("\n=== RIM/SIDE/CENTER SWEEP ===", flush=True)
        for r in sorted(runs, key=lambda x: -x["obj_dz"]):
            lp = next(p for p in r["phases"] if p["phase"] == "lift")
            print(
                f"  {r['run_label']}: dz={r['obj_dz']:+.4f} lifted={r['lifted']} "
                f"style={r['grasp_style']} rim={r.get('rim_inset')} ins={r.get('insert_depth')} "
                f"lift_obj_z_end={lp.get('obj_z_end', 'n/a'):.4f} grip={r['gripper_closed']}",
                flush=True,
            )

    if args.sweep_side:
        print("\n=== SIDE X SWEEP ===", flush=True)
        for r in runs:
            cp = next(p for p in r["phases"] if p["phase"] == "close")
            print(
                f"  side_x={r.get('side_x')} close_z={cp['eef_z']:.3f} xy={cp['xy_dist']:.3f} "
                f"lifted={r['lifted']} grip={r['gripper_closed']} knocked={r['obj_knocked']}",
                flush=True,
            )

    if args.sweep_base_z:
        print("\n=== BASE Z SWEEP ===", flush=True)
        for r in runs:
            cp = next(p for p in r["phases"] if p["phase"] == "close")
            gap = cp["eef_z"] - r["initial_obj_z"]
            print(
                f"  base_z={r['sawyer_base_z']:+.2f} close_z={cp['eef_z']:.3f} "
                f"gap={gap:.3f} lifted={r['lifted']} reach={r['reached_at_close']}",
                flush=True,
            )

    report = {
        "suite": args.suite,
        "task_id": args.task_id,
        "lifted_dz_min": LIFTED_DZ_MIN,
        "sweep_base_z": args.sweep_base_z,
        "sweep_pose": args.sweep_pose,
        "all_passed": all(r["passed"] for r in runs),
        "any_lifted": any(r["lifted"] for r in runs),
        "runs": runs,
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
