#!/usr/bin/env python3
"""Compare scripted parquet vs open-loop replay for one episode."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

EVAL = os.environ.get("OPENPI_EVAL_ROOT", str(Path.home() / "openpi_ws" / "openpi_eval"))
sys.path.insert(0, EVAL)
sys.path.insert(0, os.environ.get("OPENPI_LIBERO_ROOT", str(Path.home() / "openpi_ws" / "openpi" / "third_party" / "libero")))

import torch

_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": k.get("weights_only", False)})

from generate_b_demos import LIBERO_DUMMY, NUM_STABILIZE, _make_env, _setup_libero_config, _sim_proprio_state
from libero.libero import benchmark
from libero_grasp_smoke import (
    _object_body_id,
    _phases_for_mode,
    _target_object_name,
    run_grasp_attempt,
)
from preview_scripted_demos import _apply_export_setup

_setup_libero_config()

repo = str(Path(os.environ.get("LIBERO_DATA_ROOT", str(Path.home() / "openpi_data"))) / "libero_sawyer_scripted_v1")
ep = int(os.environ.get("EP", "50"))
t = pq.read_table(f"{repo}/data/chunk-000/episode_{ep:06d}.parquet")
rec0 = np.asarray(t["state"][0].as_py())
recl = np.asarray(t["state"][-1].as_py())
actions = [np.asarray(a.as_py()) for a in t["actions"]]
print(f"parquet ep{ep}: n={len(t)} eef_z0={rec0[2]:.3f} eef_zL={recl[2]:.3f}")

bench = benchmark.get_benchmark_dict()["libero_object"]()
task = bench.get_task(1)
inits = bench.get_task_init_states(1)
init_i = ep - 50

env = _make_env(task, "sawyer", 7)
env.reset()
obs = env.set_init_state(inits[init_i])
for _ in range(NUM_STABILIZE):
    obs, _, _, _ = env.step(LIBERO_DUMMY)
obs, _ = _apply_export_setup(env, obs, task_id=1, suite="object", ik_backend="mujoco")
sim = env.env.robots[0].sim
obj = _target_object_name(env)
bid = _object_body_id(sim, obj)
z0 = float(sim.data.body_xpos[bid][2])
st = _sim_proprio_state(obs)
print(f"replay setup: ||d0||={np.linalg.norm(st - rec0):.4f} eef_z={st[2]:.3f}")

for a in actions:
    obs, r, d, _ = env.step(a.tolist())
z1 = float(sim.data.body_xpos[bid][2])
stf = _sim_proprio_state(obs)
print(f"open-loop: obj_dz={z1 - z0:+.3f} done={d} reward={r:.3f} ||dL||={np.linalg.norm(stf - recl):.4f}")
env.close()

res = run_grasp_attempt(
    robot="sawyer",
    suite="object",
    task_id=1,
    init_index=init_i,
    start_pose_strategy="home_w30",
    phases=_phases_for_mode("ik_set", 0.05, "side", place_to_basket=True, motion_profile="compact"),
    record_video=False,
    record_demo=True,
    grasp_mode="ik_set",
    grasp_style="side",
    side_x=0.025,
    side_z=0.015,
    place_to_basket=True,
    motion_profile="compact",
)
live0 = res["demo_frames"][0]["state"]
print(f"live: passed={res['passed']} lift_dz={res['lift_dz']:+.3f} placed={res['placed']}")
print(f"live vs parquet0: ||d||={np.linalg.norm(live0 - rec0):.4f}")
