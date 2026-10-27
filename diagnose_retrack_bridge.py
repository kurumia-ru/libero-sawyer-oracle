#!/usr/bin/env python3
"""Diagnose OSC bridge reach error after ik_set teleports."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

EVAL = os.environ["OPENPI_EVAL_ROOT"]
sys.path.insert(0, EVAL)
sys.path.insert(0, os.environ["OPENPI_LIBERO_ROOT"])

_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": k.get("weights_only", False)})

from export_scripted_demos import _pose_track_action_scaled  # noqa: E402
from generate_b_demos import LIBERO_DUMMY, NUM_STABILIZE, _make_env, _sim_proprio_state  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from libero_grasp_smoke import _phases_for_mode, run_grasp_attempt  # noqa: E402
from preview_scripted_demos import _apply_export_setup  # noqa: E402


def main() -> None:
    res = run_grasp_attempt(
        robot="sawyer",
        suite="object",
        task_id=1,
        init_index=0,
        start_pose_strategy="home_w30",
        phases=_phases_for_mode("ik_set", 0.05, "side", place_to_basket=True, motion_profile="compact"),
        record_video=False,
        record_demo=True,
        ik_backend="mujoco",
        grasp_mode="ik_set",
        grasp_style="side",
        side_x=0.025,
        side_z=0.015,
        place_to_basket=True,
        motion_profile="compact",
    )
    frames = res["demo_frames"]
    for i in range(1, len(frames)):
        a = np.asarray(frames[i - 1]["state"][:3])
        b = np.asarray(frames[i]["state"][:3])
        d = float(np.linalg.norm(b - a))
        if d > 0.025:
            g = float(np.asarray(frames[i]["actions"])[-1])
            print(f"teleport frame={i} dist={d:.3f} grip={g:.2f} z={frames[i]['state'][2]:.3f}")

    bench = benchmark.get_benchmark_dict()["libero_object"]()
    task = bench.get_task(1)
    inits = bench.get_task_init_states(1)
    env = _make_env(task, "sawyer", 7)
    env.reset()
    obs = env.set_init_state(inits[0])
    for _ in range(NUM_STABILIZE):
        obs, _, _, _ = env.step(LIBERO_DUMMY)
    obs, _ = _apply_export_setup(env, obs, task_id=1, suite="object", ik_backend="mujoco")
    sim = env.env.robots[0].sim
    obj = [n for n in env.env.objects_dict if "cream" in n.lower()][0]
    bid = sim.model.body_name2id(obj + "_main")
    obj0 = sim.data.body_xpos[bid].copy()
    z0 = float(obj0[2])
    jump_m = 0.025
    pos_tol = 0.008
    for i, fr in enumerate(frames):
        target = np.asarray(fr["state"], dtype=np.float32)
        orig = np.asarray(fr["actions"], dtype=np.float32)
        grip = float(orig[-1])
        if i == 0:
            live_jump = 0.0
        else:
            prev = np.asarray(frames[i - 1]["state"], dtype=np.float32)
            live_jump = float(np.linalg.norm(target[:3] - prev[:3]))
        if live_jump <= jump_m:
            obs, _, _, _ = env.step(orig.tolist())
            continue
        transit = -1.0
        budget = max(80, int(live_jump / 0.002) + 20)
        steps = 0
        for _ in range(budget):
            cur = _sim_proprio_state(obs)
            pos_tgt = cur.copy()
            pos_tgt[:3] = target[:3]
            action = _pose_track_action_scaled(obs, pos_tgt, transit, pos_scale=0.02, rot_scale=0.5)
            obs, _, _, _ = env.step(action.tolist())
            steps += 1
            if float(np.linalg.norm(_sim_proprio_state(obs)[:3] - target[:3])) < pos_tol:
                break
        for _ in range(budget // 2):
            action = _pose_track_action_scaled(obs, target, transit, pos_scale=0.02, rot_scale=0.25)
            obs, _, _, _ = env.step(action.tolist())
            cur = _sim_proprio_state(obs)
            if float(np.linalg.norm(cur[:3] - target[:3])) < pos_tol and float(
                np.linalg.norm(cur[3:6] - target[3:6])
            ) < 0.15:
                break
        for _ in range(8):
            action = _pose_track_action_scaled(obs, target, grip, pos_scale=0.02, rot_scale=0.25)
            obs, _, _, _ = env.step(action.tolist())
            cur = _sim_proprio_state(obs)
            if float(np.linalg.norm(cur[:3] - target[:3])) < pos_tol:
                break
        cur = _sim_proprio_state(obs)
        objn = sim.data.body_xpos[bid].copy()
        print(
            f"bridge i={i} jump={live_jump:.3f} steps={steps} "
            f"pos_err={float(np.linalg.norm(cur[:3] - target[:3])):.4f} "
            f"rot_err={float(np.linalg.norm(cur[3:6] - target[3:6])):.4f} "
            f"obj_d={float(np.linalg.norm(objn - obj0)):.4f} "
            f"obj_dz={float(objn[2] - z0):+.4f}"
        )
        obj0 = objn
    print("END obj_dz", float(sim.data.body_xpos[bid][2] - z0))
    env.close()


if __name__ == "__main__":
    main()
