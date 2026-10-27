#!/usr/bin/env python3
"""Smoke: open-loop vs closed-loop trajectory comparison (dataset / policy).

Diagnoses the classic split:
  - Open-loop teacher-force actions match recorded  -> training captured the demo
  - Open-loop action replay tracks joints/EEF       -> demo is physically consistent
  - Closed-loop diverges                            -> disturbance / error accumulation

Produces figures in the style of joint-trajectory smoke plots:
  green = recorded / open-loop replay
  red   = model (teacher-force actions, or closed-loop rollout)

Requires a running serve_policy websocket (for teacher-force + closed-loop).
Open-loop replay alone works without a policy server if --replay-only.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

EVAL = Path(os.environ.get("OPENPI_EVAL_ROOT", str(Path.home() / "openpi_ws" / "openpi_eval")))
sys.path.insert(0, str(EVAL))
sys.path.insert(0, os.environ.get("OPENPI_LIBERO_ROOT", str(Path.home() / "openpi_ws" / "openpi" / "third_party" / "libero")))

import torch

_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": k.get("weights_only", False)})

from eval_libero_sawyer import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    _get_libero_env,
    _maybe_sawyer_home_w30,
    _quat2axisangle,
)
from generate_b_demos import _setup_libero_config, _sim_proprio_state  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from openpi_client import image_tools  # noqa: E402


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _decode_image(cell) -> np.ndarray:
    if isinstance(cell, dict):
        return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"), dtype=np.uint8)
    return np.asarray(cell, dtype=np.uint8)


def _load_episode(repo: Path, ep: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], str | None]:
    table = pq.read_table(repo / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
    state_col = "state" if "state" in table.column_names else "observation.state"
    act_col = "actions" if "actions" in table.column_names else "action"
    states = np.stack([np.asarray(s, dtype=np.float32) for s in table[state_col].to_pylist()])
    actions = np.stack([np.asarray(a, dtype=np.float32) for a in table[act_col].to_pylist()])
    images = [_decode_image(table["image"][i].as_py()) for i in range(len(states))]
    wrists = [_decode_image(table["wrist_image"][i].as_py()) for i in range(len(states))]
    prompt = None
    tasks_path = repo / "meta" / "episodes.jsonl"
    if tasks_path.exists():
        lines = [json.loads(l) for l in tasks_path.read_text().splitlines() if l.strip()]
        if 0 <= ep < len(lines) and lines[ep].get("tasks"):
            prompt = str(lines[ep]["tasks"][0]).strip()
    return states, actions, images, wrists, prompt


def _resolve_init_index(repo: Path, ep: int, task_id: int, override: int | None) -> int:
    if override is not None:
        return int(override)
    for name in (f"scripted_export_summary_{task_id}.json", "scripted_export_summary.json"):
        p = repo / "meta" / name
        if not p.exists():
            continue
        summary = json.loads(p.read_text())
        records = summary.get("records") or []
        if 0 <= ep < len(records) and "init_index" in records[ep]:
            return int(records[ep]["init_index"])
    # LIBERO Franka demos: episode index usually maps to init_state index.
    return int(ep)


def _obs_to_element(obs, prompt: str, resize: int = 224) -> dict:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize, resize))
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": prompt,
    }


def _demo_frame_element(
    image: np.ndarray,
    wrist: np.ndarray,
    state: np.ndarray,
    prompt: str,
    resize: int = 224,
) -> dict:
    return {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(image, resize, resize)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist, resize, resize)
        ),
        "observation/state": np.asarray(state, dtype=np.float32),
        "prompt": prompt,
    }


def _joint_pos(obs) -> np.ndarray:
    q = np.asarray(obs.get("robot0_joint_pos", np.zeros(7)), dtype=np.float32).reshape(-1)[:7]
    return q.copy()


def _reset_env(
    env,
    inits,
    init_index: int,
    *,
    num_steps_wait: int,
    robot_variant: str,
    task_id: int,
    suite: str,
    sawyer_home_w30: bool,
    ik_backend: str,
):
    env.reset()
    obs = env.set_init_state(inits[init_index % len(inits)])
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    obs = _maybe_sawyer_home_w30(
        env,
        obs,
        task_id=task_id,
        task_suite_name=suite,
        robot_variant=robot_variant,
        enabled=sawyer_home_w30 and robot_variant == "sawyer",
        ik_backend=ik_backend,
    )
    return obs


def _step_action(env, obs, action, *, robot_variant: str, action_space: str, ik_ctx):
    action = np.asarray(action, dtype=np.float32)
    if robot_variant == "sawyer" and action_space == "abs_eef":
        from scripted_action_spaces import step_abs_eef_ik

        obs, reward, done = step_abs_eef_ik(env, obs, action, ik_ctx)
        return obs, float(reward), bool(done)
    obs, reward, done, _ = env.step(action.tolist())
    return obs, float(reward), bool(done)


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------

def rollout_open_loop_replay(
    env,
    inits,
    demo_actions: np.ndarray,
    *,
    init_index: int,
    n_steps: int,
    num_steps_wait: int,
    robot_variant: str,
    task_id: int,
    suite: str,
    sawyer_home_w30: bool,
    ik_backend: str,
    action_space: str,
    ik_ctx,
) -> dict:
    obs = _reset_env(
        env,
        inits,
        init_index,
        num_steps_wait=num_steps_wait,
        robot_variant=robot_variant,
        task_id=task_id,
        suite=suite,
        sawyer_home_w30=sawyer_home_w30,
        ik_backend=ik_backend,
    )
    joints, states, rewards = [], [], []
    done = False
    n = min(n_steps, len(demo_actions))
    for t in range(n):
        joints.append(_joint_pos(obs))
        states.append(_sim_proprio_state(obs))
        if done:
            break
        obs, reward, done = _step_action(
            env, obs, demo_actions[t], robot_variant=robot_variant, action_space=action_space, ik_ctx=ik_ctx
        )
        rewards.append(reward)
    return {
        "joints": np.stack(joints) if joints else np.zeros((0, 7), np.float32),
        "states": np.stack(states) if states else np.zeros((0, 8), np.float32),
        "rewards": rewards,
        "done": done,
        "n": len(joints),
    }


def rollout_closed_loop(
    env,
    inits,
    client,
    prompt: str,
    *,
    init_index: int,
    n_steps: int,
    num_steps_wait: int,
    robot_variant: str,
    task_id: int,
    suite: str,
    sawyer_home_w30: bool,
    ik_backend: str,
    action_space: str,
    ik_ctx,
    replan_steps: int,
) -> dict:
    obs = _reset_env(
        env,
        inits,
        init_index,
        num_steps_wait=num_steps_wait,
        robot_variant=robot_variant,
        task_id=task_id,
        suite=suite,
        sawyer_home_w30=sawyer_home_w30,
        ik_backend=ik_backend,
    )
    joints, states, actions, rewards = [], [], [], []
    plan: collections.deque = collections.deque()
    done = False
    for t in range(n_steps):
        joints.append(_joint_pos(obs))
        states.append(_sim_proprio_state(obs))
        if done:
            break
        if not plan:
            chunk = client.infer(_obs_to_element(obs, prompt))["actions"]
            plan.extend(np.asarray(a, dtype=np.float32) for a in chunk[:replan_steps])
        act = plan.popleft()
        actions.append(act)
        obs, reward, done = _step_action(
            env, obs, act, robot_variant=robot_variant, action_space=action_space, ik_ctx=ik_ctx
        )
        rewards.append(reward)
    return {
        "joints": np.stack(joints) if joints else np.zeros((0, 7), np.float32),
        "states": np.stack(states) if states else np.zeros((0, 8), np.float32),
        "actions": np.stack(actions) if actions else np.zeros((0, 7), np.float32),
        "rewards": rewards,
        "done": done,
        "n": len(joints),
    }


def rollout_open_loop_policy(
    env,
    inits,
    client,
    prompt: str,
    demo_images: list[np.ndarray],
    demo_wrists: list[np.ndarray],
    demo_states: np.ndarray,
    *,
    init_index: int,
    n_steps: int,
    num_steps_wait: int,
    robot_variant: str,
    task_id: int,
    suite: str,
    sawyer_home_w30: bool,
    ik_backend: str,
    action_space: str,
    ik_ctx,
) -> dict:
    """Open-loop model rollout: policy always sees GT demo obs (no live feedback)."""
    obs = _reset_env(
        env,
        inits,
        init_index,
        num_steps_wait=num_steps_wait,
        robot_variant=robot_variant,
        task_id=task_id,
        suite=suite,
        sawyer_home_w30=sawyer_home_w30,
        ik_backend=ik_backend,
    )
    joints, states, actions, rewards = [], [], [], []
    done = False
    n = min(n_steps, len(demo_images), len(demo_states))
    for t in range(n):
        joints.append(_joint_pos(obs))
        states.append(_sim_proprio_state(obs))
        if done:
            break
        el = _demo_frame_element(demo_images[t], demo_wrists[t], demo_states[t], prompt)
        act = np.asarray(client.infer(el)["actions"][0], dtype=np.float32)
        actions.append(act)
        obs, reward, done = _step_action(
            env, obs, act, robot_variant=robot_variant, action_space=action_space, ik_ctx=ik_ctx
        )
        rewards.append(reward)
    return {
        "joints": np.stack(joints) if joints else np.zeros((0, 7), np.float32),
        "states": np.stack(states) if states else np.zeros((0, 8), np.float32),
        "actions": np.stack(actions) if actions else np.zeros((0, 7), np.float32),
        "rewards": rewards,
        "done": done,
        "n": len(joints),
    }


def teacher_force_actions(
    client,
    demo_images: list[np.ndarray],
    demo_wrists: list[np.ndarray],
    demo_states: np.ndarray,
    demo_actions: np.ndarray,
    prompt: str,
    *,
    n_steps: int,
    stride: int = 1,
) -> dict:
    """Open-loop on dataset observations: predict action given GT obs (no env)."""
    pred, gt = [], []
    idxs = list(range(0, min(n_steps, len(demo_actions)), stride))
    for t in idxs:
        el = _demo_frame_element(demo_images[t], demo_wrists[t], demo_states[t], prompt)
        # Take first action of chunk (same as one-step teacher force).
        a = np.asarray(client.infer(el)["actions"][0], dtype=np.float32)
        pred.append(a)
        gt.append(demo_actions[t])
    pred_a = np.stack(pred) if pred else np.zeros((0, 7), np.float32)
    gt_a = np.stack(gt) if gt else np.zeros((0, 7), np.float32)
    err = np.linalg.norm(pred_a - gt_a, axis=-1) if len(pred_a) else np.zeros(0)
    return {"t": np.asarray(idxs, dtype=np.int32), "pred": pred_a, "gt": gt_a, "l2": err}


# ---------------------------------------------------------------------------
# Metrics / plots
# ---------------------------------------------------------------------------

def _first_exceed(err: np.ndarray, thr: float) -> int | None:
    hits = np.where(err > thr)[0]
    return int(hits[0]) if len(hits) else None


def _normalize_traj(arr: np.ndarray) -> np.ndarray:
    """Per-dim mean/std normalize for visualization (shared example style)."""
    out = arr.astype(np.float64).copy()
    for d in range(out.shape[1]):
        col = out[:, d]
        std = float(np.std(col))
        if std < 1e-8:
            out[:, d] = col - float(np.mean(col))
        else:
            out[:, d] = (col - float(np.mean(col))) / std
    return out


def _plot_series_grid(
    series: list[tuple[str, np.ndarray, str]],
    *,
    title: str,
    ylabel_prefix: str,
    out_path: Path,
    normalize: bool,
    labels: list[str] | None = None,
    shade: float = 0.18,
) -> None:
    """Mentor-style stacked trajectories: recorded (green) vs model (red) + soft band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Align length
    n = min(s.shape[0] for _, s, _ in series)
    dims = series[0][1].shape[1]
    if labels is None:
        labels = [f"{ylabel_prefix}{i+1}" for i in range(dims)]

    fig, axes = plt.subplots(dims, 1, figsize=(8.5, 1.15 * dims), sharex=True)
    if dims == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=11)

    plotted = []
    for name, arr, color in series:
        a = arr[:n].astype(np.float64)
        if normalize:
            a = _normalize_traj(a)
        plotted.append((name, a, color))

    # Shared normalize across recorded+model so bands are comparable.
    if normalize and len(plotted) >= 2:
        stacked = np.concatenate([a for _, a, _ in plotted], axis=0)
        shared = []
        for name, a, color in plotted:
            b = a.copy()
            for d in range(dims):
                # already per-series normalized above; keep as-is for readability
                pass
            shared.append((name, b, color))
        plotted = shared

    frames = np.arange(n)
    for d, ax in enumerate(axes):
        for name, a, color in plotted:
            y = a[:, d]
            if shade > 0:
                ax.fill_between(
                    frames,
                    y - shade,
                    y + shade,
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )
            ax.plot(frames, y, color=color, label=name, linewidth=1.4, alpha=0.95)
        ax.set_ylabel(labels[d], fontsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=7)
        if d == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("frame")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _normalize_pair(recorded: np.ndarray, model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize recorded+model with shared per-dim mean/std (fair overlay)."""
    n = min(len(recorded), len(model))
    r = recorded[:n].astype(np.float64)
    m = model[:n].astype(np.float64)
    out_r = np.empty_like(r)
    out_m = np.empty_like(m)
    for d in range(r.shape[1]):
        both = np.concatenate([r[:, d], m[:, d]])
        mu = float(np.mean(both))
        std = float(np.std(both))
        if std < 1e-8:
            out_r[:, d] = r[:, d] - mu
            out_m[:, d] = m[:, d] - mu
        else:
            out_r[:, d] = (r[:, d] - mu) / std
            out_m[:, d] = (m[:, d] - mu) / std
    return out_r, out_m


def _plot_recorded_vs_model(
    recorded: np.ndarray,
    model: np.ndarray,
    *,
    title: str,
    labels: list[str],
    out_path: Path,
    normalize: bool = True,
    shade: float = 0.18,
) -> None:
    """Two-panel mentor style: green recorded, red model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(recorded), len(model))
    r = recorded[:n].astype(np.float64)
    m = model[:n].astype(np.float64)
    if normalize:
        r, m = _normalize_pair(r, m)

    dims = r.shape[1]
    fig, axes = plt.subplots(dims, 1, figsize=(8.5, 1.15 * dims), sharex=True)
    if dims == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=11)
    frames = np.arange(n)
    for d, ax in enumerate(axes):
        for y, color, name in (
            (r[:, d], "#2ca02c", "recorded"),
            (m[:, d], "#d62728", "model"),
        ):
            ax.fill_between(frames, y - shade, y + shade, color=color, alpha=0.18, linewidth=0)
            ax.plot(frames, y, color=color, label=name, linewidth=1.4)
        ax.set_ylabel(labels[d], fontsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=7)
        if d == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("frame")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _summary_text(
    *,
    tf: dict | None,
    ol: dict,
    cl: dict | None,
    demo_states: np.ndarray,
) -> dict:
    out: dict = {}
    if tf is not None and len(tf["l2"]):
        out["teacher_force_action_l2_mean"] = float(np.mean(tf["l2"]))
        out["teacher_force_action_l2_p50"] = float(np.median(tf["l2"]))
        out["teacher_force_action_l2_p90"] = float(np.percentile(tf["l2"], 90))
        out["teacher_force_first_l2_gt_0.2"] = _first_exceed(tf["l2"], 0.2)
        out["teacher_force_first_l2_gt_0.5"] = _first_exceed(tf["l2"], 0.5)

    # Open-loop replay vs demo stored state (EEF)
    n = min(ol["n"], len(demo_states))
    if n > 0:
        eef_err = np.linalg.norm(ol["states"][:n, :3] - demo_states[:n, :3], axis=-1)
        out["open_loop_replay_eef_err_mean"] = float(np.mean(eef_err))
        out["open_loop_replay_eef_err_final"] = float(eef_err[-1])
        out["open_loop_first_eef_gt_2cm"] = _first_exceed(eef_err, 0.02)
        out["open_loop_first_eef_gt_5cm"] = _first_exceed(eef_err, 0.05)

    if cl is not None and cl["n"] > 0 and ol["n"] > 0:
        m = min(cl["n"], ol["n"])
        joint_err = np.linalg.norm(cl["joints"][:m] - ol["joints"][:m], axis=-1)
        eef_err_cl = np.linalg.norm(cl["states"][:m, :3] - ol["states"][:m, :3], axis=-1)
        out["closed_vs_open_joint_l2_mean"] = float(np.mean(joint_err))
        out["closed_vs_open_eef_err_mean"] = float(np.mean(eef_err_cl))
        out["closed_vs_open_eef_err_final"] = float(eef_err_cl[-1])
        out["closed_first_eef_gt_2cm"] = _first_exceed(eef_err_cl, 0.02)
        out["closed_first_eef_gt_5cm"] = _first_exceed(eef_err_cl, 0.05)
        out["closed_first_eef_gt_10cm"] = _first_exceed(eef_err_cl, 0.10)
        # Heuristic verdict
        tf_ok = out.get("teacher_force_action_l2_mean", 0.0) < 0.35
        ol_ok = out.get("open_loop_replay_eef_err_mean", 0.0) < 0.03
        cl_bad = out.get("closed_vs_open_eef_err_mean", 0.0) > 0.05
        if tf_ok and ol_ok and cl_bad:
            out["verdict"] = (
                "train_ok_openloop_ok_closedloop_weak "
                "(likely disturbance sensitivity / error accumulation)"
            )
        elif not tf_ok:
            out["verdict"] = "teacher_force_mismatch (training or obs mismatch)"
        elif not ol_ok:
            out["verdict"] = "open_loop_replay_drift (demo/action-space/control_freq issue)"
        else:
            out["verdict"] = "open_and_closed_look_ok_on_this_episode"
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--ep", type=int, default=0)
    p.add_argument("--task-id", type=int, default=1)
    p.add_argument("--init-index", type=int, default=None)
    p.add_argument("--suite", default="libero_object", choices=["libero_object", "libero_spatial", "libero_10"])
    p.add_argument("--robot-variant", default="sawyer", choices=["default", "sawyer"])
    p.add_argument("--action-space", default="delta_osc", choices=["delta_osc", "abs_eef"])
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--n-steps", type=int, default=150)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--teacher-stride", type=int, default=2, help="Subsample demo frames for teacher-force.")
    p.add_argument("--sawyer-home-w30", action="store_true", default=True)
    p.add_argument("--no-sawyer-home-w30", action="store_false", dest="sawyer_home_w30")
    p.add_argument("--ik-backend", default="mujoco")
    p.add_argument("--replay-only", action="store_true", help="Skip policy queries (no websocket).")
    p.add_argument("--normalize-plots", action="store_true", default=True)
    p.add_argument("--no-normalize-plots", action="store_false", dest="normalize_plots")
    p.add_argument("--control-freq", type=int, default=20)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    _setup_libero_config()

    repo = Path(args.repo)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_states, demo_actions, demo_images, demo_wrists, meta_prompt = _load_episode(repo, args.ep)
    init_index = _resolve_init_index(repo, args.ep, args.task_id, args.init_index)

    bench = benchmark.get_benchmark_dict()[args.suite]()
    task = bench.get_task(args.task_id)
    inits = bench.get_task_init_states(args.task_id)
    env, env_prompt = _get_libero_env(task, 256, 7, args.robot_variant, control_freq=args.control_freq)
    prompt = meta_prompt or str(env_prompt)

    ik_ctx = None
    if args.robot_variant == "sawyer" and args.action_space == "abs_eef":
        from sawyer_ik import ik_context_from_env

        env.reset()
        ik_ctx = ik_context_from_env(env)

    n_steps = min(args.n_steps, len(demo_actions))
    joint_prefix = "panda_joint" if args.robot_variant != "sawyer" else "sawyer_joint"
    joint_labels = [f"{joint_prefix}{i}" for i in range(1, 8)]

    print(
        f"[smoke] repo={repo.name} ep={args.ep} init={init_index} "
        f"task={args.task_id} robot={args.robot_variant} action={args.action_space} "
        f"n_steps={n_steps} frames_demo={len(demo_actions)}",
        flush=True,
    )

    # 1) Open-loop replay of recorded actions
    ol = rollout_open_loop_replay(
        env,
        inits,
        demo_actions,
        init_index=init_index,
        n_steps=n_steps,
        num_steps_wait=args.num_steps_wait,
        robot_variant=args.robot_variant,
        task_id=args.task_id,
        suite=args.suite,
        sawyer_home_w30=args.sawyer_home_w30,
        ik_backend=args.ik_backend,
        action_space=args.action_space,
        ik_ctx=ik_ctx,
    )
    print(f"[smoke] open-loop replay frames={ol['n']} done={ol['done']}", flush=True)

    tf = None
    cl = None
    ol_policy = None
    if not args.replay_only:
        from openpi_client import websocket_client_policy as _websocket_client_policy

        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

        # 2) Teacher-force on demo observations (action-space diagnostic)
        tf = teacher_force_actions(
            client,
            demo_images,
            demo_wrists,
            demo_states,
            demo_actions,
            prompt,
            n_steps=n_steps,
            stride=max(1, args.teacher_stride),
        )
        print(
            f"[smoke] teacher-force points={len(tf['l2'])} "
            f"action_l2_mean={float(np.mean(tf['l2'])) if len(tf['l2']) else math.nan:.4f}",
            flush=True,
        )

        # 3) Open-loop policy (GT demo obs each step) -> joint traj
        if args.robot_variant == "sawyer" and args.action_space == "abs_eef":
            from sawyer_ik import ik_context_from_env

            env.reset()
            ik_ctx = ik_context_from_env(env)
        ol_policy = rollout_open_loop_policy(
            env,
            inits,
            client,
            prompt,
            demo_images,
            demo_wrists,
            demo_states,
            init_index=init_index,
            n_steps=n_steps,
            num_steps_wait=args.num_steps_wait,
            robot_variant=args.robot_variant,
            task_id=args.task_id,
            suite=args.suite,
            sawyer_home_w30=args.sawyer_home_w30,
            ik_backend=args.ik_backend,
            action_space=args.action_space,
            ik_ctx=ik_ctx,
        )
        print(f"[smoke] open-loop policy frames={ol_policy['n']} done={ol_policy['done']}", flush=True)

        # 4) Closed-loop policy rollout
        if args.robot_variant == "sawyer" and args.action_space == "abs_eef":
            from sawyer_ik import ik_context_from_env

            env.reset()
            ik_ctx = ik_context_from_env(env)
        cl = rollout_closed_loop(
            env,
            inits,
            client,
            prompt,
            init_index=init_index,
            n_steps=n_steps,
            num_steps_wait=args.num_steps_wait,
            robot_variant=args.robot_variant,
            task_id=args.task_id,
            suite=args.suite,
            sawyer_home_w30=args.sawyer_home_w30,
            ik_backend=args.ik_backend,
            action_space=args.action_space,
            ik_ctx=ik_ctx,
            replan_steps=args.replan_steps,
        )
        print(f"[smoke] closed-loop frames={cl['n']} done={cl['done']}", flush=True)

    # Plots (mentor-style pair)
    tag = f"ep{args.ep}_init{init_index}"
    norm_tag = "normalized" if args.normalize_plots else "raw"
    task_title = f"task{args.task_id}"

    if ol_policy is not None and ol["n"] > 0 and ol_policy["n"] > 0:
        _plot_recorded_vs_model(
            ol["joints"],
            ol_policy["joints"],
            title=f"{task_title} joint trajectories — {norm_tag} (open_loop)",
            labels=joint_labels,
            out_path=out_dir / f"{tag}_joints_open_loop.png",
            normalize=args.normalize_plots,
        )
    if cl is not None and ol["n"] > 0 and cl["n"] > 0:
        _plot_recorded_vs_model(
            ol["joints"],
            cl["joints"],
            title=f"{task_title} joint trajectories — {norm_tag} (closed_loop)",
            labels=joint_labels,
            out_path=out_dir / f"{tag}_joints_closed_loop.png",
            normalize=args.normalize_plots,
        )

    # Extra diagnostics (kept)
    if cl is not None and ol["n"] > 0 and cl["n"] > 0:
        _plot_series_grid(
            [
                ("recorded", ol["joints"], "#2ca02c"),
                ("model", cl["joints"], "#d62728"),
            ],
            title=f"{task_title} joint trajectories — {norm_tag} (closed vs open replay)",
            ylabel_prefix="joint",
            out_path=out_dir / f"{tag}_joints_closed_vs_open.png",
            normalize=args.normalize_plots,
            labels=joint_labels,
        )

    eef_series = [("recorded (demo state)", demo_states[:n_steps, :3], "#1f77b4")]
    if ol["n"] > 0:
        eef_series.append(("open_loop replay", ol["states"][:, :3], "#2ca02c"))
    if cl is not None and cl["n"] > 0:
        eef_series.append(("closed_loop model", cl["states"][:, :3], "#d62728"))
    _plot_series_grid(
        eef_series,
        title=f"{task_title} EEF xyz — {norm_tag}",
        ylabel_prefix="eef_",
        out_path=out_dir / f"{tag}_eef_xyz.png",
        normalize=args.normalize_plots,
        labels=["eef_x", "eef_y", "eef_z"],
    )

    if tf is not None and len(tf["gt"]) > 0:
        _plot_series_grid(
            [
                ("recorded", tf["gt"], "#2ca02c"),
                ("model", tf["pred"], "#d62728"),
            ],
            title=f"{task_title} actions — {norm_tag} (open_loop teacher-force)",
            ylabel_prefix="a",
            out_path=out_dir / f"{tag}_actions_teacher_force.png",
            normalize=args.normalize_plots,
            labels=[f"a{i}" for i in range(tf["gt"].shape[1])],
        )

    metrics = _summary_text(tf=tf, ol=ol, cl=cl, demo_states=demo_states)
    if ol_policy is not None and ol["n"] > 0 and ol_policy["n"] > 0:
        m = min(ol["n"], ol_policy["n"])
        metrics["open_loop_policy_vs_replay_joint_l2_mean"] = float(
            np.mean(np.linalg.norm(ol_policy["joints"][:m] - ol["joints"][:m], axis=-1))
        )
    report = {
        "repo": str(repo),
        "ep": args.ep,
        "init_index": init_index,
        "task_id": args.task_id,
        "suite": args.suite,
        "robot_variant": args.robot_variant,
        "action_space": args.action_space,
        "prompt": prompt,
        "n_steps_requested": n_steps,
        "open_loop_n": ol["n"],
        "open_loop_policy_n": None if ol_policy is None else ol_policy["n"],
        "closed_loop_n": None if cl is None else cl["n"],
        "teacher_force_n": None if tf is None else int(len(tf["l2"])),
        "metrics": metrics,
    }
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({"metrics": metrics}, indent=2), flush=True)
    print(f"WROTE {out_dir}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
