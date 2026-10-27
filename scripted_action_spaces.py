#!/usr/bin/env python3
"""Action-space encode/decode and replay for scripted Sawyer datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from generate_b_demos import (
    LIBERO_DUMMY,
    NUM_STABILIZE,
    POS_SCALE,
    ROT_SCALE,
    _make_env,
    _pose_track_action,
    _setup_libero_config,
    _sim_proprio_state,
)

ActionSpace = Literal["delta_osc", "abs_eef", "joint"]


@dataclass
class EefTarget:
    pos_aa: np.ndarray  # (6,) eef pos + axis-angle
    gripper: float


def encode_abs_eef(target: EefTarget) -> np.ndarray:
    t = np.asarray(target.pos_aa, dtype=np.float32).reshape(6)
    return np.concatenate([t[:3], t[3:6], np.array([target.gripper], dtype=np.float32)])


def decode_abs_eef(action: np.ndarray) -> EefTarget:
    a = np.asarray(action, dtype=np.float32).reshape(7)
    return EefTarget(pos_aa=a[:6].copy(), gripper=float(a[6]))


def encode_joint(q_target: np.ndarray, ik_ctx) -> np.ndarray:
    q = np.asarray(q_target, dtype=np.float32).reshape(7)
    lo = ik_ctx.joint_limits[:, 0]
    hi = ik_ctx.joint_limits[:, 1]
    mid = (lo + hi) * 0.5
    half = np.maximum((hi - lo) * 0.5, 1e-6)
    return np.clip((q - mid) / half, -1.0, 1.0).astype(np.float32)


def decode_joint(action: np.ndarray, ik_ctx) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).reshape(7)
    lo = ik_ctx.joint_limits[:, 0]
    hi = ik_ctx.joint_limits[:, 1]
    mid = (lo + hi) * 0.5
    half = (hi - lo) * 0.5
    return (mid + a * half).astype(np.float64)


def step_abs_eef_ik(
    env,
    obs: dict,
    action: np.ndarray,
    ik_ctx,
    *,
    # Default: no IK teleport. Densified demos should already be smooth; teleport
    # on open-gripper approach is the main source of jerky open-loop / eval motion.
    jump_m: Optional[float] = None,
) -> tuple[dict, float, bool]:
    """Execute abs EEF target for Sawyer via OSC position chase.

    Optional IK teleport (jump_m) is only for recovery on huge open-gripper gaps.
    While carrying (gripper close), never teleport — joint jumps drop the object.
    """
    from generate_b_demos import _pose_track_action, _sim_proprio_state
    from libero_grasp_smoke import _eef_wxyz_from_obs
    from sawyer_ik import apply_arm_qpos, axisangle_to_wxyz, solve_sawyer_ik

    tgt = decode_abs_eef(action)
    cur = _sim_proprio_state(obs)
    err = float(np.linalg.norm(tgt.pos_aa[:3] - cur[:3]))
    # Gripper convention: +1 close, -1 open.
    carrying = float(tgt.gripper) > 0.0
    use_teleport = (
        jump_m is not None and err >= float(jump_m) and not carrying
    )
    if use_teleport:
        seed = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
        try:
            target_wxyz = axisangle_to_wxyz(tgt.pos_aa[3:6])
            if not np.isfinite(target_wxyz).all() or float(np.linalg.norm(target_wxyz)) < 0.5:
                target_wxyz = _eef_wxyz_from_obs(obs)
        except Exception:
            target_wxyz = _eef_wxyz_from_obs(obs)
        q = solve_sawyer_ik(
            ik_ctx,
            tgt.pos_aa[:3],
            target_wxyz,
            seed,
            backend="mujoco",
            pos_only=True,
        )
        obs = apply_arm_qpos(env, ik_ctx, q)
        hold = _sim_proprio_state(obs)
        step_action = _pose_track_action(obs, hold, tgt.gripper)
    else:
        # Position-only chase; keep current orientation to avoid OSC stall.
        pos_tgt = cur.copy()
        pos_tgt[:3] = tgt.pos_aa[:3]
        step_action = _pose_track_action(obs, pos_tgt, tgt.gripper)
    obs, reward, done, _ = env.step(step_action.tolist())
    return obs, float(reward), bool(done)


def step_arm_qpos_gripper(
    env,
    obs: dict,
    q_arm: np.ndarray,
    gripper: float,
    ik_ctx,
) -> tuple[dict, float, bool]:
    """Exact joint teleport + gripper step (open-loop of recorded ik_set demos)."""
    from libero_grasp_smoke import _gripper_only_action
    from sawyer_ik import apply_arm_qpos

    obs = apply_arm_qpos(env, ik_ctx, np.asarray(q_arm, dtype=np.float64).reshape(7))
    obs, reward, done, _ = env.step(_gripper_only_action(float(gripper)).tolist())
    return obs, float(reward), bool(done)


def reencode_abs_eef_from_qpos(frames: list[dict]) -> list[dict]:
    """BC pairs: obs_t -> abs EEF of state_{t+1}, with arm_qpos_{t+1} for exact replay.

    Open-loop verification should prefer arm_qpos (exact), while the policy learns
    abs_eef (7D, openpi-compatible) executed via IK at eval.
    """
    if len(frames) < 2:
        return []
    out: list[dict] = []
    for i in range(len(frames) - 1):
        cur = frames[i]
        nxt = frames[i + 1]
        nxt_state = np.asarray(nxt["state"], dtype=np.float32)
        nxt_grip = float(np.asarray(nxt["actions"], dtype=np.float32).reshape(-1)[-1])
        fr = {
            "image": cur["image"],
            "wrist_image": cur["wrist_image"],
            "state": np.asarray(cur["state"], dtype=np.float32),
            "actions": encode_abs_eef(EefTarget(pos_aa=nxt_state[:6].copy(), gripper=nxt_grip)),
        }
        if "arm_qpos" in nxt:
            fr["arm_qpos"] = np.asarray(nxt["arm_qpos"], dtype=np.float32).reshape(7)
            fr["gripper_cmd"] = np.float32(nxt_grip)
        out.append(fr)
    return out


def step_env(
    env,
    obs: dict,
    action: np.ndarray,
    *,
    action_space: ActionSpace,
    ik_ctx,
    gripper_cmd: float | None = None,
) -> tuple[dict, float, bool]:
    from libero_grasp_smoke import _gripper_only_action

    if action_space == "delta_osc":
        step_action = np.asarray(action, dtype=np.float32)
        obs, reward, done, _ = env.step(step_action.tolist())
        return obs, float(reward), bool(done)
    if action_space == "abs_eef":
        return step_abs_eef_ik(env, obs, action, ik_ctx)
    if action_space == "joint":
        from sawyer_ik import apply_arm_qpos

        q = decode_joint(action, ik_ctx)
        obs = apply_arm_qpos(env, ik_ctx, q)
        g = float(gripper_cmd if gripper_cmd is not None else float(np.asarray(action).reshape(-1)[-1]))
        step_action = _gripper_only_action(g)
        obs, reward, done, _ = env.step(step_action.tolist())
        return obs, float(reward), bool(done)
    raise ValueError(f"Unknown action_space {action_space!r}")


def reencode_abs_eef_next_state(frames: list[dict]) -> list[dict]:
    """Map post-step demo frames to BC pairs: obs_t -> abs pose of state_{t+1}.

    ik_set teleports show up as large state jumps; labeling the next EEF pose as
    the action makes those jumps learnable under IK execution at train/eval time.
    """
    if len(frames) < 2:
        return []
    out: list[dict] = []
    for i in range(len(frames) - 1):
        cur = frames[i]
        nxt = frames[i + 1]
        nxt_state = np.asarray(nxt["state"], dtype=np.float32)
        nxt_grip = float(np.asarray(nxt["actions"], dtype=np.float32).reshape(-1)[-1])
        out.append(
            {
                "image": cur["image"],
                "wrist_image": cur["wrist_image"],
                "state": np.asarray(cur["state"], dtype=np.float32),
                "actions": encode_abs_eef(EefTarget(pos_aa=nxt_state[:6].copy(), gripper=nxt_grip)),
            }
        )
    return out


class RecordingPatch:
    """Capture EEF / joint targets during scripted rollout."""

    def __init__(self) -> None:
        self.eef_targets: list[EefTarget] = []
        self.q_targets: list[np.ndarray | None] = []

    def install(self) -> None:
        import generate_b_demos as gd
        import libero_grasp_smoke as lgs
        import sawyer_ik as sik

        self._gd = gd
        self._lgs = lgs
        self._sik = sik
        self._orig_pt = gd._pose_track_action
        self._orig_hold = lgs._hold_pose_action
        self._orig_grip = lgs._gripper_only_action
        self._orig_ik = sik.solve_sawyer_ik
        self._pending_q: np.ndarray | None = None

        rec = self

        def pt(obs, target, gripper):
            t = np.asarray(target, dtype=np.float32)
            rec.eef_targets.append(EefTarget(pos_aa=t[:6].copy(), gripper=float(gripper)))
            rec.q_targets.append(rec._pending_q)
            rec._pending_q = None
            return rec._orig_pt(obs, target, gripper)

        def hold(obs, gripper, sim_fn, pt_fn):
            cur = sim_fn(obs)
            rec.eef_targets.append(EefTarget(pos_aa=cur[:6].copy(), gripper=float(gripper)))
            rec.q_targets.append(rec._pending_q)
            rec._pending_q = None
            # Use unpatched pt_fn — orig_hold calls pt internally; avoid double-counting targets.
            return rec._orig_hold(obs, gripper, sim_fn, rec._orig_pt)

        def grip(gripper):
            rec.eef_targets.append(EefTarget(pos_aa=np.full(6, np.nan), gripper=float(gripper)))
            rec.q_targets.append(rec._pending_q)
            rec._pending_q = None
            return rec._orig_grip(gripper)

        def ik(ctx, target_pos, target_wxyz, seed_q, **kw):
            q = rec._orig_ik(ctx, target_pos, target_wxyz, seed_q, **kw)
            rec._pending_q = np.asarray(q, dtype=np.float32).copy()
            return q

        gd._pose_track_action = pt
        lgs._hold_pose_action = hold
        lgs._gripper_only_action = grip
        sik.solve_sawyer_ik = ik

    def remove(self) -> None:
        self._gd._pose_track_action = self._orig_pt
        self._lgs._hold_pose_action = self._orig_hold
        self._lgs._gripper_only_action = self._orig_grip
        self._sik.solve_sawyer_ik = self._orig_ik

    def reencode_frames(
        self,
        frames: list[dict],
        *,
        action_space: ActionSpace,
        ik_ctx,
    ) -> list[dict]:
        if len(frames) != len(self.eef_targets):
            raise RuntimeError(
                f"frame/target mismatch: {len(frames)} frames vs {len(self.eef_targets)} targets"
            )
        out: list[dict] = []
        last_q: np.ndarray | None = None
        for fr, tgt, q in zip(frames, self.eef_targets, self.q_targets):
            fr = dict(fr)
            if q is not None:
                last_q = np.asarray(q, dtype=np.float32)
            if action_space == "abs_eef":
                pos_aa = tgt.pos_aa
                if np.any(np.isnan(pos_aa)):
                    pos_aa = np.asarray(fr["state"], dtype=np.float32)[:6]
                fr["actions"] = encode_abs_eef(EefTarget(pos_aa, tgt.gripper))
            elif action_space == "joint":
                jq = last_q
                if jq is None:
                    from sawyer_ik import axisangle_to_wxyz, solve_sawyer_ik

                    pos_aa = tgt.pos_aa
                    if np.any(np.isnan(pos_aa)):
                        pos_aa = np.asarray(fr["state"], dtype=np.float32)[:6]
                    jq = solve_sawyer_ik(
                        ik_ctx,
                        pos_aa[:3],
                        axisangle_to_wxyz(pos_aa[3:6]),
                        np.asarray(ik_ctx.sim.data.qpos[ik_ctx.qpos_addrs], dtype=np.float32),
                        pos_only=True,
                    )
                fr["actions"] = encode_joint(jq, ik_ctx)
                fr["gripper_cmd"] = np.array([tgt.gripper], dtype=np.float32)
            out.append(fr)
        return out


def replay_episode(
    *,
    repo,
    ep: int,
    init_i: int,
    task_id: int,
    suite: str,
    action_space: ActionSpace,
    ik_backend: str = "mujoco",
) -> dict:
    import pyarrow.parquet as pq
    from libero.libero import benchmark
    from preview_scripted_demos import _apply_export_setup, _parquet_path

    t = pq.read_table(_parquet_path(repo, ep))
    actions = np.stack(t["actions"].to_pylist())
    gripper_cmds = None
    if "gripper_cmd" in t.column_names:
        gripper_cmds = np.stack(t["gripper_cmd"].to_pylist()).reshape(-1)

    bench = benchmark.get_benchmark_dict()[f"libero_{suite}"]()
    task = bench.get_task(task_id)
    inits = bench.get_task_init_states(task_id)
    env = _make_env(task, "sawyer", 7)
    env.reset()
    obs = env.set_init_state(inits[init_i % len(inits)])
    for _ in range(NUM_STABILIZE):
        obs, _, _, _ = env.step(LIBERO_DUMMY)
    obs, _ = _apply_export_setup(env, obs, task_id=task_id, suite=suite, ik_backend=ik_backend)

    from sawyer_ik import ik_context_from_env

    ik_ctx = ik_context_from_env(env)
    done = False
    reward = 0.0
    for i, act in enumerate(actions):
        gcmd = float(gripper_cmds[i]) if gripper_cmds is not None else None
        obs, reward, done = step_env(
            env, obs, act, action_space=action_space, ik_ctx=ik_ctx, gripper_cmd=gcmd
        )
        if done:
            break
    env.close()
    return {
        "episode": ep,
        "task_id": task_id,
        "init_index": init_i,
        "action_space": action_space,
        "success": bool(done and reward > 0),
        "done": bool(done),
        "reward": float(reward),
        "n_actions": len(actions),
    }
