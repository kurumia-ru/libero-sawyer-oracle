#!/usr/bin/env python3
"""Replay scripted Sawyer demos (libero_sawyer_scripted_v*) in sim; record mp4."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import imageio
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
    _render_pair,
    _setup_libero_config,
    apply_start_pose_strategy,
    sawyer_home_qpos,
)

# Must match export_scripted_demos.py
SCRIPTED_TASK_PRESETS: dict[int, dict] = {
    0: {"grasp_style": "cap", "force_gripper_open": True, "start_pose_strategy": "home_w30"},
    1: {"grasp_style": "side", "force_gripper_open": False, "start_pose_strategy": "video5s"},
    2: {"grasp_style": "cap", "force_gripper_open": True, "start_pose_strategy": "home_w30"},
    3: {"grasp_style": "cap", "force_gripper_open": True, "start_pose_strategy": "home_w30"},
    4: {"grasp_style": "cap", "force_gripper_open": True, "start_pose_strategy": "home_w30"},
    5: {"grasp_style": "cap", "force_gripper_open": True, "start_pose_strategy": "home_w30"},
    6: {"grasp_style": "side", "force_gripper_open": False, "start_pose_strategy": "home_w30"},
    7: {"grasp_style": "side", "force_gripper_open": False, "start_pose_strategy": "home_w30"},
    8: {"grasp_style": "side", "force_gripper_open": False, "start_pose_strategy": "home_w30"},
    9: {"grasp_style": "side", "force_gripper_open": False, "start_pose_strategy": "home_w30"},
}


def _apply_export_setup(
    env,
    obs,
    *,
    task_id: int,
    suite: str,
    ik_backend: str,
    start_pose_strategy: str | None = None,
) -> dict:
    """Match live export start pose before retrack / open-loop replay.

    If start_pose_strategy is provided, it overrides the per-task preview preset.
    This keeps live (CLI) and retrack/open-loop on the same initial arm pose —
    otherwise task0 can live with video5s while retrack resets to home_w30 and
    open-loop fails after a ~0.4m bridge.
    """
    preset = SCRIPTED_TASK_PRESETS.get(task_id, {"start_pose_strategy": "home_w30"})
    from sawyer_ik import apply_arm_qpos, apply_gripper_qpos, ik_context_from_env
    from libero_grasp_smoke import _franka_frame0_for_task

    info: dict = {"task_id": task_id}
    strat = start_pose_strategy or preset.get("start_pose_strategy", "home_w30")
    info["start_pose_strategy"] = strat
    if strat == "home_w30":
        ik_ctx = ik_context_from_env(env)
        obs = apply_arm_qpos(env, ik_ctx, sawyer_home_qpos())
        info["setup"] = "sawyer_home_teleport"
    elif strat:
        ik_ctx = ik_context_from_env(env)
        frame0 = _franka_frame0_for_task(suite, task_id)
        obs, align = apply_start_pose_strategy(
            env,
            ik_ctx,
            obs,
            frame0,
            strat,
            ik_backend=ik_backend,
            warmup_steps_override=0,
        )
        info["setup"] = strat
        info["align"] = align
        if strat == "video5s":
            obs = apply_gripper_qpos(env)
            info["gripper_setup"] = "full_open"
    if preset.get("force_gripper_open"):
        obs = apply_gripper_qpos(env)
        info["gripper_forced_open"] = True
    return obs, info


def _torch_patch() -> None:
    import torch

    _orig = torch.load

    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)

    torch.load = _load  # type: ignore


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s[:max_len]


def _parquet_path(repo: Path, ep: int) -> Path:
    chunk = ep // 1000
    path = repo / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
    if path.is_file():
        return path
    hits = sorted(repo.glob(f"data/chunk-*/episode_{ep:06d}.parquet"))
    if hits:
        return hits[0]
    raise FileNotFoundError(path)


def _episode_init_indices(meta_path: Path, lang_to_task) -> list[int]:
    """Per-episode init index within task (export order: init 0,1,... per task)."""
    per_task: dict[tuple[str, int], int] = defaultdict(int)
    out: list[int] = []
    for line in meta_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row["tasks"][0].strip()
        bench_name, task_id = lang_to_task[prompt]
        local_i = per_task[(bench_name, task_id)]
        per_task[(bench_name, task_id)] += 1
        out.append(local_i)
    return out


def replay_episodes(
    *,
    repo: Path,
    out_dir: Path,
    episodes: list[int],
    fps: int = 10,
    video_tag: str = "scripted",
    suite: str = "object",
    ik_backend: str = "mujoco",
) -> list[dict]:
    from libero.libero import benchmark

    meta_path = repo / "meta/episodes.jsonl"
    meta = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    lang_to_task = _build_task_maps()
    init_indices = _episode_init_indices(meta_path, lang_to_task)
    bench_dict = benchmark.get_benchmark_dict()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for out_ep in episodes:
        if out_ep >= len(meta):
            print(f"skip ep {out_ep}: only {len(meta)} episodes")
            continue
        row = meta[out_ep]
        prompt = row["tasks"][0].strip()
        bench_name, task_id = lang_to_task[prompt]
        t = pq.read_table(_parquet_path(repo, out_ep))
        actions = np.stack([np.asarray(a.as_py(), dtype=np.float32) for a in t["actions"]])

        task = bench_dict[bench_name]().get_task(task_id)
        inits = bench_dict[bench_name]().get_task_init_states(task_id)
        init_i = init_indices[out_ep]

        env = _make_env(task, "sawyer", 7)
        env.reset()
        obs = env.set_init_state(inits[init_i % len(inits)])
        frames: list[np.ndarray] = []
        for _ in range(NUM_STABILIZE):
            obs, _, _, _ = env.step(LIBERO_DUMMY)
            agent, _ = _render_pair(obs)
            frames.append(agent)

        setup_info = None
        try:
            obs, setup_info = _apply_export_setup(
                env, obs, task_id=task_id, suite=suite, ik_backend=ik_backend
            )
            agent, _ = _render_pair(obs)
            frames.append(agent)
        except Exception as exc:
            print(f"  setup warning ep {out_ep}: {exc}", flush=True)

        done = False
        reward = 0.0
        for act in actions:
            obs, reward, done, _ = env.step(act.tolist())
            agent, _ = _render_pair(obs)
            frames.append(agent)
            if bool(done):
                break
        env.close()

        success = bool(done and reward > 0)
        tag = "ok" if success else "replay"
        fname = f"{video_tag}_ep{out_ep:03d}_init{init_i:02d}_t{task_id}_{tag}.mp4"
        out_path = out_dir / fname
        imageio.mimwrite(out_path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)
        rec = {
            "episode": out_ep,
            "task_id": task_id,
            "init_index": init_i,
            "task": prompt,
            "n_frames": len(frames),
            "success": success,
            "reward": float(reward),
            "video": str(out_path),
            "setup_info": setup_info,
        }
        results.append(rec)
        print(f"ep {out_ep:03d} init={init_i} task={task_id} -> {out_path.name}", flush=True)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=str(DATA_ROOT / "libero_sawyer_scripted_v1"))
    p.add_argument("--out-dir", default=str(OPENPI_EVAL / "scripted_preview"))
    p.add_argument("--task-id", type=int, default=None, help="Filter episodes by LIBERO task id.")
    p.add_argument("--episodes", type=int, nargs="*", default=None)
    p.add_argument("--sample", type=int, default=5, help="If --episodes omitted, sample N per task-id.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--video-tag", default="scripted")
    p.add_argument("--suite", default="object", choices=["object", "spatial", "long"])
    p.add_argument("--ik-backend", default="mujoco")
    args = p.parse_args()

    _setup_libero_config()
    os.environ.setdefault("MUJOCO_GL", "egl")
    _torch_patch()

    repo = Path(args.repo)
    meta_path = repo / "meta/episodes.jsonl"
    lang_to_task = _build_task_maps()
    meta = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]

    if args.episodes:
        episodes = list(args.episodes)
    else:
        by_task: dict[int, list[int]] = defaultdict(list)
        for i, row in enumerate(meta):
            prompt = row["tasks"][0].strip()
            _, tid = lang_to_task[prompt]
            if args.task_id is None or tid == args.task_id:
                by_task[tid].append(i)
        episodes = []
        for tid in sorted(by_task):
            eps = by_task[tid]
            step = max(1, len(eps) // args.sample)
            episodes.extend(eps[::step][: args.sample])

    out_dir = Path(args.out_dir)
    if args.task_id is not None:
        out_dir = out_dir / f"task{args.task_id}"
    results = replay_episodes(
        repo=repo,
        out_dir=out_dir,
        episodes=episodes,
        fps=args.fps,
        video_tag=args.video_tag,
        suite=args.suite,
        ik_backend=args.ik_backend,
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({"episodes": results}, indent=2))
    print(f"WROTE {len(results)} videos -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
