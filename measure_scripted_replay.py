#!/usr/bin/env python3
"""Open-loop sim replay success for scripted Sawyer datasets (all episodes)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

OPENPI_EVAL = Path(os.environ.get("OPENPI_EVAL_ROOT", str(Path.home() / "openpi_ws" / "openpi_eval")))
DATA_ROOT = Path(os.environ.get("LIBERO_DATA_ROOT", str(Path.home() / "openpi_data")))
sys.path.insert(0, str(OPENPI_EVAL))

from generate_b_demos import (  # noqa: E402
    LIBERO_DUMMY,
    NUM_STABILIZE,
    _build_task_maps,
    _make_env,
    _setup_libero_config,
    _sim_proprio_state,
)
from preview_scripted_demos import (  # noqa: E402
    _apply_export_setup,
    _episode_init_indices,
    _parquet_path,
)


def _torch_patch() -> None:
    import torch

    _orig = torch.load

    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)

    torch.load = _load  # type: ignore


def replay_one(
    *,
    repo: Path,
    out_ep: int,
    init_i: int,
    lang_to_task,
    bench_dict,
    meta_row: dict,
    suite: str,
    ik_backend: str,
    track_drift: bool,
) -> dict:
    from libero_grasp_smoke import _object_body_id, _target_object_name

    prompt = meta_row["tasks"][0].strip()
    bench_name, task_id = lang_to_task[prompt]
    t = pq.read_table(_parquet_path(repo, out_ep))
    actions = np.stack(t["actions"].to_pylist())
    rec0 = np.asarray(t["state"][0].as_py(), dtype=np.float32)
    recl = np.asarray(t["state"][-1].as_py(), dtype=np.float32)

    task = bench_dict[bench_name]().get_task(task_id)
    inits = bench_dict[bench_name]().get_task_init_states(task_id)
    env = _make_env(task, "sawyer", 7)
    env.reset()
    obs = env.set_init_state(inits[init_i % len(inits)])
    for _ in range(NUM_STABILIZE):
        obs, _, _, _ = env.step(LIBERO_DUMMY)

    setup_ok = True
    setup_err = ""
    try:
        obs, _ = _apply_export_setup(
            env, obs, task_id=task_id, suite=suite, ik_backend=ik_backend
        )
    except Exception as exc:
        setup_ok = False
        setup_err = str(exc)

    st0 = _sim_proprio_state(obs)
    d0 = float(np.linalg.norm(st0 - rec0))

    sim = env.env.robots[0].sim
    obj = _target_object_name(env)
    bid = _object_body_id(sim, obj)
    obj_z0 = float(sim.data.body_xpos[bid][2])

    done = False
    reward = 0.0
    for act in actions:
        obs, reward, done, _ = env.step(act.tolist())
        if bool(done):
            break

    obj_z1 = float(sim.data.body_xpos[bid][2])
    stf = _sim_proprio_state(obs)
    dL = float(np.linalg.norm(stf - recl))
    env.close()

    success = bool(done and reward > 0)
    rec: dict = {
        "episode": out_ep,
        "task_id": task_id,
        "init_index": init_i,
        "task": prompt,
        "success": success,
        "done": bool(done),
        "reward": float(reward),
        "n_actions": len(actions),
        "setup_ok": setup_ok,
    }
    if track_drift:
        rec.update(
            {
                "setup_err": setup_err,
                "state_d0": d0,
                "state_dL": dL,
                "obj_dz": obj_z1 - obj_z0,
                "eef_z0": float(st0[2]),
                "eef_zL": float(stf[2]),
            }
        )
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=str(DATA_ROOT / "libero_sawyer_scripted_v1"))
    p.add_argument("--suite", default="object", choices=["object", "spatial", "long"])
    p.add_argument("--ik-backend", default="mujoco")
    p.add_argument("--max-ep", type=int, default=None)
    p.add_argument("--every", type=int, default=50, help="Progress print interval.")
    p.add_argument("--out", default=None, help="JSON summary path.")
    p.add_argument("--track-drift", action="store_true")
    args = p.parse_args()

    _setup_libero_config()
    os.environ.setdefault("MUJOCO_GL", "egl")
    _torch_patch()

    from libero.libero import benchmark

    repo = Path(args.repo)
    meta_path = repo / "meta/episodes.jsonl"
    meta = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    if args.max_ep is not None:
        meta = meta[: args.max_ep]

    lang_to_task = _build_task_maps()
    init_indices = _episode_init_indices(meta_path, lang_to_task)
    bench_dict = benchmark.get_benchmark_dict()

    results: list[dict] = []
    by_task: dict[int, dict] = defaultdict(lambda: {"n": 0, "ok": 0, "name": ""})

    for out_ep, row in enumerate(meta):
        r = replay_one(
            repo=repo,
            out_ep=out_ep,
            init_i=init_indices[out_ep],
            lang_to_task=lang_to_task,
            bench_dict=bench_dict,
            meta_row=row,
            suite=args.suite,
            ik_backend=args.ik_backend,
            track_drift=args.track_drift,
        )
        results.append(r)
        tid = r["task_id"]
        by_task[tid]["n"] += 1
        by_task[tid]["ok"] += int(r["success"])
        by_task[tid]["name"] = r["task"][:70]

        if args.every and (out_ep + 1) % args.every == 0:
            s = sum(x["success"] for x in results)
            print(
                f"{out_ep + 1}/{len(meta)} success={s}/{len(results)} ({100 * s / len(results):.1f}%)",
                flush=True,
            )

    succ = sum(r["success"] for r in results)
    done_ct = sum(r["done"] for r in results)
    n = len(results)
    summary = {
        "repo": str(repo),
        "suite": args.suite,
        "n_episodes": n,
        "success": succ,
        "success_rate": succ / n if n else 0.0,
        "done_before_end": done_ct,
        "done_rate": done_ct / n if n else 0.0,
        "by_task": {
            str(tid): {
                "task_id": tid,
                "n": v["n"],
                "success": v["ok"],
                "rate": v["ok"] / v["n"] if v["n"] else 0.0,
                "name": v["name"],
            }
            for tid, v in sorted(by_task.items())
        },
        "episodes": results,
    }

    out_path = Path(args.out) if args.out else OPENPI_EVAL / "scripted_replay_all.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== SCRIPTED OPEN-LOOP REPLAY ===")
    print(f"repo: {repo}")
    print(f"total: {succ}/{n} ({100 * succ / n:.1f}%)  done={done_ct}/{n}")
    for tid in sorted(by_task):
        v = by_task[tid]
        rate = 100 * v["ok"] / v["n"] if v["n"] else 0.0
        print(f"  task{tid}: {v['ok']}/{v['n']} ({rate:.0f}%) | {v['name']}")
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
