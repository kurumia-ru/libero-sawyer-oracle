#!/usr/bin/env python3
"""Minimal Sawyer smoke test: OSC moves (up/down/xy) + gripper, optional mp4.

Modes:
  robosuite — standard robosuite Sawyer on Lift (no LIBERO)
  libero    — LIBERO OffScreenRenderEnv + SawyerCustom (no demo replay)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POS_SCALE = np.array([0.05, 0.05, 0.05], dtype=np.float32)
ROT_SCALE = np.array([0.5, 0.5, 0.5], dtype=np.float32)
MIN_AXIS_DELTA_M = 0.02
MIN_DOWN_DELTA_M = 0.02
MIN_GRIPPER_APERTURE_DELTA = 0.025
GRIPPER_STEPS = 50
# LIBERO / robosuite OSC: +1 = close, -1 = open (RethinkGripper)
GRIP_CLOSE = 1.0
GRIP_OPEN = -1.0


@dataclass(frozen=True)
class MoveSpec:
    name: str
    action: np.ndarray
    steps: int
    expect_axis: int | None = None  # 0=x, 1=y, 2=z
    expect_sign: int = 1


def _osc_action(
    dpos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    drot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gripper: float = -1.0,
) -> np.ndarray:
    dpos_n = np.clip(np.asarray(dpos, dtype=np.float32) / POS_SCALE, -1.0, 1.0)
    drot_n = np.clip(np.asarray(drot, dtype=np.float32) / ROT_SCALE, -1.0, 1.0)
    return np.concatenate([dpos_n, drot_n, np.array([gripper], dtype=np.float32)])


def _eef_pos(obs: dict) -> np.ndarray:
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)


def _gripper_aperture(obs: dict) -> float:
    """Finger separation for RethinkGripper (symmetric qpos; mean is always ~0)."""
    g = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)
    return float(g[0] - g[1])


def _render_frame(obs: dict, camera: str = "agentview") -> np.ndarray:
    key = f"{camera}_image" if f"{camera}_image" in obs else "agentview_image"
    img = np.ascontiguousarray(obs[key][::-1, ::-1])
    return img


def _run_moves(env, obs: dict, moves: list[MoveSpec], *, record_video: bool) -> tuple[dict, list[np.ndarray], list[dict]]:
    frames: list[np.ndarray] = []
    results: list[dict] = []

    if record_video:
        frames.append(_render_frame(obs))

    for move in moves:
        start = _eef_pos(obs)
        start_aperture = _gripper_aperture(obs)
        for _ in range(move.steps):
            obs, _, _, _ = env.step(move.action.tolist())
            if record_video:
                frames.append(_render_frame(obs))
        end = _eef_pos(obs)
        end_aperture = _gripper_aperture(obs)
        delta = end - start
        aperture_delta = end_aperture - start_aperture

        ok = True
        if move.expect_axis is not None:
            axis_delta = float(delta[move.expect_axis] * move.expect_sign)
            ok = axis_delta >= MIN_AXIS_DELTA_M
            if move.name == "move_down":
                ok = (-float(delta[2])) >= MIN_DOWN_DELTA_M
        elif move.name in ("gripper_close", "gripper_open"):
            ok = abs(aperture_delta) >= MIN_GRIPPER_APERTURE_DELTA
        elif move.name == "gripper_hold":
            ok = True

        results.append(
            {
                "name": move.name,
                "steps": move.steps,
                "delta_xyz": delta.tolist(),
                "aperture_start": start_aperture,
                "aperture_end": end_aperture,
                "aperture_delta": aperture_delta,
                "ok": bool(ok),
            }
        )
    return obs, frames, results


def _default_moves() -> list[MoveSpec]:
    still = _osc_action()
    return [
        MoveSpec("move_up", _osc_action(dpos=(0, 0, 0.05)), 20, expect_axis=2, expect_sign=1),
        MoveSpec("move_down", _osc_action(dpos=(0, 0, -0.05)), 25, expect_axis=2, expect_sign=-1),
        MoveSpec("move_forward_x", _osc_action(dpos=(0.05, 0, 0)), 20, expect_axis=0, expect_sign=1),
        MoveSpec("move_left_y", _osc_action(dpos=(0, 0.05, 0)), 20, expect_axis=1, expect_sign=1),
        # Close before open: from open home, -1 alone barely moves; +1 then -1 is reliable.
        MoveSpec("gripper_close", _osc_action(gripper=GRIP_CLOSE), GRIPPER_STEPS),
        MoveSpec("gripper_open", _osc_action(gripper=GRIP_OPEN), GRIPPER_STEPS),
        MoveSpec("gripper_hold", still, 5),
    ]


def make_robosuite_env(*, env_name: str, robot: str, camera: str, resolution: int):
    import robosuite as suite

    return suite.make(
        env_name,
        robots=robot,
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera,
        camera_heights=resolution,
        camera_widths=resolution,
        reward_shaping=False,
    )


def run_robosuite_smoke(
    *,
    env_name: str,
    robot: str,
    camera: str,
    resolution: int,
    record_video: bool,
) -> dict:
    env = make_robosuite_env(env_name=env_name, robot=robot, camera=camera, resolution=resolution)
    obs = env.reset()
    obs, frames, checks = _run_moves(env, obs, _default_moves(), record_video=record_video)
    final_eef = _eef_pos(obs).tolist()
    env.close()
    passed = all(c["ok"] for c in checks)
    arm_passed = all(c["ok"] for c in checks if c["name"].startswith("move_"))
    gripper_passed = all(c["ok"] for c in checks if c["name"] in ("gripper_close", "gripper_open"))
    return {
        "mode": "robosuite",
        "env": env_name,
        "robot": robot,
        "passed": passed,
        "arm_passed": arm_passed,
        "gripper_passed": gripper_passed,
        "checks": checks,
        "final_eef": final_eef,
        "frames": frames,
    }


def run_libero_smoke(
    *,
    suite_name: str,
    task_id: int,
    robot_variant: str,
    resolution: int,
    record_video: bool,
) -> dict:
    openpi_libero = Path(os.environ.get("OPENPI_LIBERO_ROOT", str(Path.home() / "openpi_ws" / "openpi" / "third_party" / "libero")))
    openpi_eval = Path(os.environ.get("OPENPI_EVAL_ROOT", str(Path.home() / "openpi_ws" / "openpi_eval")))
    sys.path.insert(0, str(openpi_libero))
    sys.path.insert(0, str(openpi_eval))

    import torch

    _orig = torch.load

    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)

    torch.load = _load  # type: ignore

    from generate_b_demos import LIBERO_DUMMY, NUM_STABILIZE, _make_env, _setup_libero_config
    from libero.libero import benchmark

    _setup_libero_config()
    bench_name = {"object": "libero_object", "spatial": "libero_spatial", "long": "libero_10"}[suite_name]
    task_suite = benchmark.get_benchmark_dict()[bench_name]()
    task = task_suite.get_task(task_id)
    init_states = task_suite.get_task_init_states(task_id)

    env = _make_env(task, robot_variant, seed=7)
    env.reset()
    obs = env.set_init_state(init_states[0])
    for _ in range(NUM_STABILIZE):
        obs, _, _, _ = env.step(LIBERO_DUMMY)

    initial_eef = _eef_pos(obs).tolist()
    obs, frames, checks = _run_moves(env, obs, _default_moves(), record_video=record_video)
    final_eef = _eef_pos(obs).tolist()
    env.close()

    passed = all(c["ok"] for c in checks)
    arm_passed = all(c["ok"] for c in checks if c["name"].startswith("move_"))
    gripper_passed = all(c["ok"] for c in checks if c["name"] in ("gripper_close", "gripper_open"))
    return {
        "mode": "libero",
        "suite": suite_name,
        "task_id": task_id,
        "robot_variant": robot_variant,
        "task": task.language,
        "passed": passed,
        "arm_passed": arm_passed,
        "gripper_passed": gripper_passed,
        "checks": checks,
        "initial_eef": initial_eef,
        "final_eef": final_eef,
        "frames": frames,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["robosuite", "libero", "both"], default="both")
    p.add_argument("--env-name", default="Lift", help="robosuite env (Lift, Stack, ...)")
    p.add_argument("--robot", default="Sawyer", help="robosuite robot name")
    p.add_argument("--libero-suite", default="object", choices=["object", "spatial", "long"])
    p.add_argument("--libero-task-id", type=int, default=0)
    p.add_argument("--libero-robot-variant", default="sawyer")
    p.add_argument("--camera", default="agentview")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--out-dir", default=str(Path(os.environ.get("OPENPI_EVAL_ROOT", str(Path.home() / "openpi_ws" / "openpi_eval"))) / "robosuite_sawyer_smoke"))
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    record_video = not args.no_video

    if args.mode in ("robosuite", "both"):
        print(f"[robosuite] {args.env_name} robot={args.robot}", flush=True)
        res = run_robosuite_smoke(
            env_name=args.env_name,
            robot=args.robot,
            camera=args.camera,
            resolution=args.resolution,
            record_video=record_video,
        )
        frames = res.pop("frames")
        if record_video and frames:
            import imageio

            vid = out_dir / f"robosuite_{args.robot.lower()}_{args.env_name.lower()}.mp4"
            imageio.mimwrite(vid, frames, fps=10)
            res["video"] = str(vid)
        summaries.append(res)
        _print_summary(res)

    if args.mode in ("libero", "both"):
        print(f"[libero] suite={args.libero_suite} task={args.libero_task_id} variant={args.libero_robot_variant}", flush=True)
        res = run_libero_smoke(
            suite_name=args.libero_suite,
            task_id=args.libero_task_id,
            robot_variant=args.libero_robot_variant,
            resolution=args.resolution,
            record_video=record_video,
        )
        frames = res.pop("frames")
        if record_video and frames:
            import imageio

            vid = out_dir / f"libero_{args.libero_robot_variant}_task{args.libero_task_id}.mp4"
            imageio.mimwrite(vid, frames, fps=10)
            res["video"] = str(vid)
        summaries.append(res)
        _print_summary(res)

    all_passed = all(s["passed"] for s in summaries)
    all_arm_passed = all(s["arm_passed"] for s in summaries)
    report = {"all_passed": all_passed, "all_arm_passed": all_arm_passed, "runs": summaries}
    report_path = out_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"WROTE {report_path}")
    if not all_passed:
        raise SystemExit(1)


def _print_summary(res: dict) -> None:
    tag = "PASS" if res["passed"] else "FAIL"
    print(f"\n=== {tag} {res.get('mode')} (arm={'OK' if res['arm_passed'] else 'NG'} gripper={'OK' if res['gripper_passed'] else 'NG'}) ===", flush=True)
    for c in res["checks"]:
        mark = "ok" if c["ok"] else "NG"
        d = c["delta_xyz"]
        if "gripper" in c["name"]:
            extra = (
                f" aperture {c['aperture_start']:.4f}->{c['aperture_end']:.4f}"
                f" (d={c['aperture_delta']:+.4f})"
            )
        else:
            extra = ""
        print(f"  [{mark}] {c['name']:16s} delta=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}){extra}", flush=True)


if __name__ == "__main__":
    main()
