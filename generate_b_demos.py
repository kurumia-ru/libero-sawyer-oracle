#!/usr/bin/env python3
"""Replay LeRobot LIBERO demos in sim on Sawyer (helpers also used by scripted export / DAgger)."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# --- paths (override via env) ---
OPENPI_LIBERO = Path(os.environ.get("OPENPI_LIBERO_ROOT", str(Path.home() / "openpi_ws" / "openpi" / "third_party" / "libero")))
OPENPI_EVAL = Path(os.environ.get("OPENPI_EVAL_ROOT", str(Path.home() / "openpi_ws" / "openpi_eval")))
DATA_ROOT = Path(os.environ.get("LIBERO_DATA_ROOT", str(Path.home() / "openpi_data")))
SRC_REPO = DATA_ROOT / "libero"

sys.path.insert(0, str(OPENPI_LIBERO))
sys.path.insert(0, str(OPENPI_EVAL))

from libero_robot_variants import register_robot_variants, robots_arg_for_variant  # noqa: E402

VARIANT_TO_REPO = {
    "sawyer": "sawyer",
}


SUITE_TO_BENCH = {"object": "libero_object", "long": "libero_10", "spatial": "libero_spatial"}
LIBERO_DUMMY = [0.0] * 6 + [-1.0]
RESOLUTION = 256
NUM_STABILIZE = 10
# Keep dataset sampling and env control on the same clock as LIBERO / π0.5 (20Hz).
DATASET_FPS = 20
CONTROL_FREQ = 20
DEFAULT_START_WARMUP_STEPS = 20
# OSC delta scaling (LIBERO / robosuite; normalized actions in [-1, 1]).
POS_SCALE = np.array([0.05, 0.05, 0.05], dtype=np.float32)
ROT_SCALE = np.array([0.5, 0.5, 0.5], dtype=np.float32)

ACTION_MODE_CHOICES = ("copy", "pose_track", "ik_track")

# Sawyer start-pose strategies (after LIBERO set_init_state + stabilize).
START_POSE_STRATEGIES = (
    "panda_state",       # baseline: Panda-derived joint state, no fix
    "sawyer_home",       # reset arm to Sawyer default init_qpos
    "ik_full",           # IK to Franka frame-0 EEF (pos+rot), direct joint set
    "ik_full_w20",       # ik_full + 20-step pose_track warmup
    "ik_full_w50",       # ik_full + 50-step warmup
    "ik_pos_w20",        # position-only IK + 20-step warmup
    "home_ik_w20",       # sawyer_home then ik_full + warmup 20
    "home_w30",          # sawyer_home + 30-step pose_track warmup (no IK set)
    "video5s",            # fixed pose measured at ~5s of successful task1 replay
)


def sawyer_home_qpos() -> np.ndarray:
    from libero_robot_variants import MountedSawyerCustom

    return np.asarray(MountedSawyerCustom().init_qpos, dtype=np.float32)


def sawyer_video5s_qpos() -> np.ndarray:
    """Common pose at step 50 of successful task1 init0/init2 open-loop runs."""
    return np.asarray(
        [
            -0.07438014,
            -1.47756635,
            -0.33164454,
            2.13287513,
            0.04455957,
            0.90515520,
            -2.00055364,
        ],
        dtype=np.float32,
    )


def _quat2axisangle(quat) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def _sim_proprio_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        [
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)


def _pose_track_action(obs: dict, target_state: np.ndarray, gripper: float) -> np.ndarray:
    """Delta OSC action on current robot toward Franka demo target pose."""
    cur = _sim_proprio_state(obs)
    target_state = np.asarray(target_state, dtype=np.float32)
    dpos = (target_state[:3] - cur[:3]) / POS_SCALE
    drot = (target_state[3:6] - cur[3:6]) / ROT_SCALE
    return np.concatenate(
        [
            np.clip(dpos, -1.0, 1.0),
            np.clip(drot, -1.0, 1.0),
            np.array([gripper], dtype=np.float32),
        ]
    ).astype(np.float32)


def _franka_target_pose(frame: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """EEF position, wxyz orientation, and gripper command from a Franka demo frame."""
    from sawyer_ik import axisangle_to_wxyz

    target_state = np.asarray(frame["state"], dtype=np.float32)
    gripper = float(np.asarray(frame["actions"], dtype=np.float32)[-1])
    target_pos = target_state[:3]
    target_wxyz = axisangle_to_wxyz(target_state[3:6])
    return target_pos, target_wxyz, gripper


def _ik_track_action(
    obs: dict,
    frame: dict,
    ik_ctx,
    *,
    ik_backend: str = "auto",
) -> np.ndarray:
    """IK retarget: Franka EEF + gripper -> Sawyer IK -> OSC toward FK target pose."""
    from sawyer_ik import fk_proprio_at_joints, solve_sawyer_ik

    target_pos, target_wxyz, gripper = _franka_target_pose(frame)
    q_current = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
    q_target = solve_sawyer_ik(
        ik_ctx,
        target_pos,
        target_wxyz,
        q_current,
        backend=ik_backend,  # type: ignore[arg-type]
    )
    ik_state = fk_proprio_at_joints(ik_ctx, q_target, gripper)
    return _pose_track_action(obs, ik_state, gripper)


def _eef_error(obs: dict, target_state: np.ndarray) -> tuple[float, float]:
    cur = _sim_proprio_state(obs)
    target_state = np.asarray(target_state, dtype=np.float32)
    return (
        float(np.linalg.norm(cur[:3] - target_state[:3])),
        float(np.linalg.norm(cur[3:6] - target_state[3:6])),
    )


def apply_start_pose_strategy(
    env,
    ik_ctx,
    obs: dict,
    frame: dict,
    strategy: str,
    *,
    ik_backend: str = "auto",
    warmup_steps_override: int | None = None,
    on_step=None,
) -> tuple[dict, dict]:
    """Apply a Sawyer start-pose strategy; return updated obs and metrics."""
    from sawyer_ik import apply_arm_qpos, solve_sawyer_ik

    if strategy not in START_POSE_STRATEGIES:
        raise ValueError(f"Unknown start_pose strategy {strategy!r}")

    target_state = np.asarray(frame["state"], dtype=np.float32)
    pos_before, rot_before = _eef_error(obs, target_state)
    target_pos, target_wxyz, gripper = _franka_target_pose(frame)

    pos_only = False
    warmup_steps = 0
    q_seed = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)

    if strategy == "panda_state":
        pass
    elif strategy == "sawyer_home":
        obs = apply_arm_qpos(env, ik_ctx, sawyer_home_qpos())
    elif strategy == "ik_full":
        q_target = solve_sawyer_ik(
            ik_ctx, target_pos, target_wxyz, q_seed, backend=ik_backend, pos_only=False
        )
        obs = apply_arm_qpos(env, ik_ctx, q_target)
    elif strategy == "ik_full_w20":
        q_target = solve_sawyer_ik(
            ik_ctx, target_pos, target_wxyz, q_seed, backend=ik_backend, pos_only=False
        )
        obs = apply_arm_qpos(env, ik_ctx, q_target)
        warmup_steps = 20
    elif strategy == "ik_full_w50":
        q_target = solve_sawyer_ik(
            ik_ctx, target_pos, target_wxyz, q_seed, backend=ik_backend, pos_only=False
        )
        obs = apply_arm_qpos(env, ik_ctx, q_target)
        warmup_steps = 50
    elif strategy == "ik_pos_w20":
        pos_only = True
        q_target = solve_sawyer_ik(
            ik_ctx, target_pos, target_wxyz, q_seed, backend=ik_backend, pos_only=True
        )
        obs = apply_arm_qpos(env, ik_ctx, q_target)
        warmup_steps = 20
    elif strategy == "home_ik_w20":
        q_seed = sawyer_home_qpos()
        obs = apply_arm_qpos(env, ik_ctx, q_seed)
        q_target = solve_sawyer_ik(
            ik_ctx, target_pos, target_wxyz, q_seed, backend=ik_backend, pos_only=False
        )
        obs = apply_arm_qpos(env, ik_ctx, q_target)
        warmup_steps = 20
    elif strategy == "home_w30":
        obs = apply_arm_qpos(env, ik_ctx, sawyer_home_qpos())
        warmup_steps = 30
    elif strategy == "video5s":
        obs = apply_arm_qpos(env, ik_ctx, sawyer_video5s_qpos())
        warmup_steps = 0
    else:
        raise AssertionError(strategy)

    if warmup_steps_override is not None:
        warmup_steps = warmup_steps_override

    pos_after_ik, rot_after_ik = _eef_error(obs, target_state)
    for _ in range(warmup_steps):
        action = _pose_track_action(obs, target_state, gripper)
        obs, _, _, _ = env.step(action.tolist())
        if on_step is not None:
            on_step(obs, action)

    pos_final, rot_final = _eef_error(obs, target_state)
    info = {
        "strategy": strategy,
        "pos_err_before_m": pos_before,
        "rot_err_before": rot_before,
        "pos_err_after_ik_m": pos_after_ik,
        "rot_err_after_ik": rot_after_ik,
        "pos_err_after_warmup_m": pos_final,
        "rot_err_after_warmup": rot_final,
        "warmup_steps": warmup_steps,
        "pos_only_ik": pos_only,
    }
    return obs, info


def align_sawyer_start_pose(
    env,
    ik_ctx,
    obs: dict,
    frame: dict,
    *,
    ik_backend: str = "auto",
    warmup_steps: int = 0,
) -> tuple[dict, dict]:
    """IK-align Sawyer joints to Franka demo frame-0 EEF; optional OSC warmup (not recorded)."""
    strategy = "ik_full_w20" if warmup_steps == 20 else "ik_full_w50" if warmup_steps == 50 else "ik_full"
    if warmup_steps not in (0, 20, 50):
        obs, info = apply_start_pose_strategy(
            env, ik_ctx, obs, frame, "ik_full", ik_backend=ik_backend
        )
        target_state = np.asarray(frame["state"], dtype=np.float32)
        gripper = float(np.asarray(frame["actions"], dtype=np.float32)[-1])
        for _ in range(warmup_steps):
            action = _pose_track_action(obs, target_state, gripper)
            obs, _, _, _ = env.step(action.tolist())
        pos_final, rot_final = _eef_error(obs, target_state)
        info["warmup_steps"] = warmup_steps
        info["pos_err_after_warmup_m"] = pos_final
        info["rot_err_after_warmup"] = rot_final
        return obs, info
    return apply_start_pose_strategy(env, ik_ctx, obs, frame, strategy, ik_backend=ik_backend)


def _action_for_frame(
    robot_variant: str,
    action_mode: str,
    obs: dict,
    frame: dict,
    *,
    ik_ctx=None,
    ik_backend: str = "auto",
) -> np.ndarray:
    if robot_variant == "sawyer" and action_mode == "pose_track":
        target_state = np.asarray(frame["state"], dtype=np.float32)
        gripper = float(np.asarray(frame["actions"], dtype=np.float32)[-1])
        return _pose_track_action(obs, target_state, gripper)
    if robot_variant == "sawyer" and action_mode == "ik_track":
        if ik_ctx is None:
            raise ValueError("ik_track requires ik_ctx from ik_context_from_env()")
        return _ik_track_action(obs, frame, ik_ctx, ik_backend=ik_backend)
    return np.asarray(frame["actions"], dtype=np.float32)


def _setup_libero_config() -> None:
    cfg_dir = OPENPI_EVAL / "libero_config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    libero_pkg = OPENPI_LIBERO / "libero/libero"
    cfg = {
        "benchmark_root": str(libero_pkg),
        "assets": str(libero_pkg / "assets"),
        "bddl_files": str(libero_pkg / "bddl_files"),
        "init_states": str(libero_pkg / "init_files"),
        "datasets": str(OPENPI_LIBERO / "datasets"),
    }
    import yaml

    with open(cfg_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg_dir)


def _build_task_maps():
    from libero.libero import benchmark

    bench_dict = benchmark.get_benchmark_dict()
    lang_to_task: dict[str, tuple[str, int]] = {}
    for suite, bench_name in SUITE_TO_BENCH.items():
        ts = bench_dict[bench_name]()
        for i in range(ts.n_tasks):
            lang_to_task[ts.get_task(i).language.strip()] = (bench_name, i)
    return lang_to_task


def _make_env(task, robot_variant: str, seed: int, *, horizon: int = 1000, control_freq: int = CONTROL_FREQ):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    register_robot_variants()
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        robots=robots_arg_for_variant(robot_variant),
        camera_heights=RESOLUTION,
        camera_widths=RESOLUTION,
        horizon=horizon,
        control_freq=control_freq,
    )
    env.seed(seed)
    return env


def _render_pair(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return agent, wrist


def _collect_episodes(suite: str, meta) -> list[int]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.training import libero_task_filter

    allowed = libero_task_filter.task_indices_for_suite(meta.tasks, suite)
    ds = LeRobotDataset(str(SRC_REPO))
    episodes: set[int] = set()
    task_col = ds.hf_dataset["task_index"]
    ep_col = ds.hf_dataset["episode_index"]
    for i in range(len(task_col)):
        if int(task_col[i]) in allowed:
            episodes.add(int(ep_col[i]))
    return sorted(episodes)


def _source_episode_manifest(out_repo: Path, suite: str) -> Path:
    return out_repo / "meta" / f"source_episodes_{suite}.jsonl"


def _processed_source_episodes(out_repo: Path, suite: str) -> set[int]:
    manifest = _source_episode_manifest(out_repo, suite)
    if not manifest.is_file():
        return set()
    done: set[int] = set()
    for line in manifest.read_text().splitlines():
        if line.strip():
            done.add(int(json.loads(line)["source_episode_index"]))
    return done


def _record_source_episode(out_repo: Path, suite: str, source_episode_index: int) -> None:
    manifest = _source_episode_manifest(out_repo, suite)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a") as f:
        f.write(json.dumps({"source_episode_index": source_episode_index}) + "\n")


def _episode_frames(ds, episode_index: int) -> list[dict]:
    task_col = ds.hf_dataset["task_index"]
    ep_col = ds.hf_dataset["episode_index"]
    rows = [i for i in range(len(ep_col)) if int(ep_col[i]) == episode_index]
    rows.sort(key=lambda i: int(ds.hf_dataset["frame_index"][i]))
    out = []
    for i in rows:
        row = ds.hf_dataset[i]
        out.append({k: row[k] for k in ("image", "wrist_image", "state", "actions", "task_index")})
    return out


def generate(
    robot_variant: str,
    suite: str,
    *,
    max_episodes: int | None = None,
    seed: int = 7,
    overwrite: bool = False,
    action_mode: str = "copy",
    output_suffix: str = "",
    ik_backend: str = "auto",
    align_start: bool = False,
    start_warmup_steps: int = DEFAULT_START_WARMUP_STEPS,
    start_pose_strategy: str | None = None,
    task_ids: list[int] | None = None,
) -> Path:
    _setup_libero_config()
    os.environ.setdefault("MUJOCO_GL", "egl")

    # LIBERO init states are numpy pickles; PyTorch 2.6+ defaults weights_only=True.
    import torch

    _torch_load = torch.load

    def _load_init_states(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _torch_load(*args, **kwargs)

    torch.load = _load_init_states  # type: ignore[method-assign]

    repo_tag = VARIANT_TO_REPO[robot_variant]
    if action_mode not in ACTION_MODE_CHOICES:
        raise ValueError(f"action_mode must be one of {ACTION_MODE_CHOICES}, got {action_mode!r}")
    if robot_variant != "sawyer" and action_mode != "copy":
        raise ValueError("pose_track / ik_track action modes are only supported for sawyer")
    repo_name = f"libero_{repo_tag}{output_suffix}"
    out_repo = DATA_ROOT / repo_name
    append_to_repo = out_repo.exists()
    if out_repo.exists():
        if overwrite and suite == "object":
            shutil.rmtree(out_repo)
            append_to_repo = False
        elif overwrite and suite == "long":
            append_to_repo = True
        elif overwrite:
            shutil.rmtree(out_repo)
            append_to_repo = False
        elif not overwrite:
            raise SystemExit(f"{out_repo} exists; pass --overwrite or remove it")

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from libero.libero import benchmark

    src_meta = LeRobotDatasetMetadata(str(SRC_REPO))
    src_ds = LeRobotDataset(str(SRC_REPO))
    lang_to_task = _build_task_maps()

    episode_indices = _collect_episodes(suite, src_meta)

    features = {
        "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }
    if append_to_repo:
        dst = LeRobotDataset(str(out_repo))
        existing = _processed_source_episodes(out_repo, suite)
        episode_indices = [ep for ep in episode_indices if ep not in existing]
        print(f"Appending {len(episode_indices)} episodes to {out_repo} (suite={suite})")
    else:
        dst = LeRobotDataset.create(
            repo_id=str(out_repo),
            robot_type="panda" if robot_variant != "sawyer" else "sawyer",
            fps=DATASET_FPS,
            features=features,
            image_writer_threads=4,
            image_writer_processes=2,
        )

    bench_dict = benchmark.get_benchmark_dict()
    per_task_ep: dict[tuple[str, int], int] = defaultdict(int)
    align_log_path = out_repo / "meta" / f"start_align_{suite}.jsonl"
    if (align_start or start_pose_strategy) and not append_to_repo:
        align_log_path.parent.mkdir(parents=True, exist_ok=True)

    selected_task_ids = set(task_ids) if task_ids else None
    # Pre-filter by task before loading frames (loading every ep is very slow).
    if selected_task_ids is not None:
        allowed_prompts = {
            prompt
            for prompt, (bench_name, tid) in lang_to_task.items()
            if SUITE_TO_BENCH.get(suite) == bench_name and tid in selected_task_ids
        }
        allowed_task_indices = {
            int(idx) for idx, text in src_meta.tasks.items() if text.strip() in allowed_prompts
        }
        ep_col = src_ds.hf_dataset["episode_index"]
        task_col = src_ds.hf_dataset["task_index"]
        keep = {
            int(ep_col[i])
            for i in range(len(task_col))
            if int(task_col[i]) in allowed_task_indices
        }
        before = len(episode_indices)
        episode_indices = [ep for ep in episode_indices if ep in keep]
        print(
            f"Prefiltered task_ids={sorted(selected_task_ids)}: "
            f"{before} -> {len(episode_indices)} source episodes",
            flush=True,
        )

    processed_count = 0
    for ep_idx in episode_indices:
        frames = _episode_frames(src_ds, ep_idx)
        if not frames:
            continue
        task_index = int(frames[0]["task_index"])
        prompt = src_meta.tasks[task_index].strip()
        if prompt not in lang_to_task:
            print(f"skip ep {ep_idx}: unknown prompt {prompt!r}")
            continue
        bench_name, task_id = lang_to_task[prompt]
        if SUITE_TO_BENCH.get(suite) != bench_name:
            continue
        if selected_task_ids is not None and task_id not in selected_task_ids:
            continue

        task_suite = bench_dict[bench_name]()
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        local_i = per_task_ep[(bench_name, task_id)]
        per_task_ep[(bench_name, task_id)] += 1

        env = _make_env(task, robot_variant, seed)
        env.reset()
        init_state = init_states[local_i % len(init_states)]
        obs = env.set_init_state(init_state)
        for _ in range(NUM_STABILIZE):
            obs, _, _, _ = env.step(LIBERO_DUMMY)

        ik_ctx = None
        if robot_variant == "sawyer" and action_mode in ("pose_track", "ik_track"):
            from sawyer_ik import ik_context_from_env

            ik_ctx = ik_context_from_env(env)

        if start_pose_strategy is not None:
            if robot_variant != "sawyer" or action_mode == "copy" or ik_ctx is None:
                raise ValueError("start_pose_strategy requires sawyer with pose_track or ik_track")
            obs, align_info = apply_start_pose_strategy(
                env, ik_ctx, obs, frames[0], start_pose_strategy, ik_backend=ik_backend
            )
        elif align_start:
            if robot_variant != "sawyer" or action_mode == "copy" or ik_ctx is None:
                raise ValueError("align_start requires sawyer with pose_track or ik_track")
            obs, align_info = align_sawyer_start_pose(
                env,
                ik_ctx,
                obs,
                frames[0],
                ik_backend=ik_backend,
                warmup_steps=start_warmup_steps,
            )
        else:
            align_info = None

        if align_info is not None:
            with align_log_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "source_episode_index": ep_idx,
                            "bench": bench_name,
                            "task_id": task_id,
                            **align_info,
                        }
                    )
                    + "\n"
                )
            print(
                f"  start ep {ep_idx} [{align_info.get('strategy', 'align')}]:"
                f" pos {align_info['pos_err_before_m']:.3f}"
                f" -> {align_info['pos_err_after_warmup_m']:.3f} m",
                flush=True,
            )

        for fr in frames:
            action = _action_for_frame(
                robot_variant,
                action_mode,
                obs,
                fr,
                ik_ctx=ik_ctx,
                ik_backend=ik_backend,
            )
            obs, _, _, _ = env.step(action.tolist())
            img, wrist = _render_pair(obs)
            state = _sim_proprio_state(obs)

            dst.add_frame(
                {
                    "image": img,
                    "wrist_image": wrist,
                    "state": state,
                    "actions": action,
                    "task": prompt,
                }
            )
        dst.save_episode()
        _record_source_episode(out_repo, suite, ep_idx)
        env.close()
        processed_count += 1
        if (local_i + 1) % 5 == 0:
            print(f"  {bench_name} task {task_id}: {local_i + 1} episodes done")
        if max_episodes is not None and processed_count >= max_episodes:
            break

    if robot_variant != "sawyer":
        ns = DATA_ROOT / "libero" / "norm_stats.json"
        if ns.is_file():
            shutil.copy(ns, out_repo / "norm_stats.json")

    print(f"Wrote {out_repo} ({processed_count} source episodes written)")
    return out_repo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=list(VARIANT_TO_REPO))
    p.add_argument("--suite", required=True, choices=list(SUITE_TO_BENCH))
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--action-mode",
        default="copy",
        choices=list(ACTION_MODE_CHOICES),
        help="copy: replay Franka actions. pose_track: EEF delta OSC. ik_track: EEF+IK+Jacobian OSC.",
    )
    p.add_argument(
        "--ik-backend",
        default="auto",
        choices=["auto", "mujoco", "pyroki"],
        help="IK solver for ik_track (auto: PyRoki if installed, else MuJoCo DLS).",
    )
    p.add_argument(
        "--output-suffix",
        default="",
        help="Append to dataset dir name, e.g. _v2 -> libero_b_sawyer_v2",
    )
    p.add_argument(
        "--align-start",
        action="store_true",
        help="Sawyer only: IK-align arm to Franka frame-0 EEF before recording.",
    )
    p.add_argument(
        "--start-warmup-steps",
        type=int,
        default=DEFAULT_START_WARMUP_STEPS,
        help="OSC pose_track steps after IK align (not recorded in dataset).",
    )
    p.add_argument(
        "--start-pose-strategy",
        default=None,
        choices=list(START_POSE_STRATEGIES),
        help="Sawyer start-pose strategy (overrides --align-start when set).",
    )
    p.add_argument(
        "--task-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional LIBERO local task IDs to include (e.g. --task-ids 1 5).",
    )
    args = p.parse_args()
    generate(
        args.variant,
        args.suite,
        max_episodes=args.max_episodes,
        seed=args.seed,
        overwrite=args.overwrite,
        action_mode=args.action_mode,
        output_suffix=args.output_suffix,
        ik_backend=args.ik_backend,
        align_start=args.align_start,
        start_warmup_steps=args.start_warmup_steps,
        start_pose_strategy=args.start_pose_strategy,
        task_ids=args.task_ids,
    )


if __name__ == "__main__":
    main()
