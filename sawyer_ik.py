"""Sawyer IK for cross-embodiment demo replay (Franka EEF -> Sawyer joints).

Primary backend: MuJoCo damped least-squares on the live LIBERO sim model (same
kinematics as rollout). Optional backend: PyRoki (install via install_pyroki_for_ik.sh).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

IkBackend = Literal["mujoco", "pyroki", "auto"]


def axisangle_to_wxyz(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle (3,) to MuJoCo / PyRoki wxyz quaternion."""
    aa = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = aa / angle
    half = 0.5 * angle
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def wxyz_to_axisangle(wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to axis-angle (3,)."""
    q = np.asarray(wxyz, dtype=np.float64).reshape(4)
    q = q / max(np.linalg.norm(q), 1e-12)
    w = float(np.clip(q[0], -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = q[1:] / s
    return (axis * angle).astype(np.float32)


def _orientation_error_wxyz(cur_wxyz: np.ndarray, tgt_wxyz: np.ndarray) -> np.ndarray:
    """3D orientation error (axis-angle) from current to target, wxyz convention."""
    q_cur = np.asarray(cur_wxyz, dtype=np.float64).reshape(4)
    q_tgt = np.asarray(tgt_wxyz, dtype=np.float64).reshape(4)
    q_cur = q_cur / max(np.linalg.norm(q_cur), 1e-12)
    q_tgt = q_tgt / max(np.linalg.norm(q_tgt), 1e-12)
    # q_err = q_tgt * q_cur^{-1}; for unit quats, inv = [w, -x, -y, -z]
    w1, x1, y1, z1 = q_tgt
    w2, x2, y2, z2 = q_cur
    q_err = np.array(
        [
            w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2,
            w1 * x2 - x1 * w2 - y1 * z2 + z1 * y2,
            w1 * y2 + x1 * z2 - y1 * w2 - z1 * x2,
            w1 * z2 - x1 * y2 + y1 * x2 - z1 * w2,
        ],
        dtype=np.float64,
    )
    return wxyz_to_axisangle(q_err).astype(np.float64)


@dataclass
class SawyerIkContext:
    """Cached indices for IK on a LIBERO Sawyer env."""

    sim: object
    eef_site_id: int
    eef_site_name: str
    qpos_addrs: np.ndarray  # (7,) qpos indices for arm joints
    qvel_addrs: np.ndarray  # (7,) qvel indices for arm joints
    joint_limits: np.ndarray  # (7, 2) lower/upper


def ik_context_from_env(env) -> SawyerIkContext:
    """Build IK context from an OffScreenRenderEnv after reset."""
    robot = env.env.robots[0]
    sim = robot.sim
    model = sim.model
    qpos_addrs = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int32)
    qvel_addrs = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int32)
    joint_ids = robot._ref_joint_indexes
    limits = np.array(
        [[model.jnt_range[jid, 0], model.jnt_range[jid, 1]] for jid in joint_ids],
        dtype=np.float64,
    )
    site_id = robot.eef_site_id
    site_name = model.site_id2name(site_id)
    return SawyerIkContext(
        sim=sim,
        eef_site_id=site_id,
        eef_site_name=site_name,
        qpos_addrs=qpos_addrs,
        qvel_addrs=qvel_addrs,
        joint_limits=limits,
    )


def _clip_arm_q(q: np.ndarray, limits: np.ndarray) -> np.ndarray:
    return np.clip(q, limits[:, 0], limits[:, 1])


def _mat_to_wxyz(mat9: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    q_xyzw = R.from_matrix(np.asarray(mat9, dtype=np.float64).reshape(3, 3)).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def solve_ik_mujoco(
    ctx: SawyerIkContext,
    target_pos: np.ndarray,
    target_wxyz: np.ndarray,
    seed_q: np.ndarray,
    *,
    max_iter: int = 80,
    pos_tol: float = 1e-4,
    ori_tol: float = 1e-3,
    damping: float = 0.05,
    step_scale: float = 0.85,
    pos_only: bool = False,
) -> np.ndarray:
    """Damped least-squares IK on the live MuJoCo model (7 arm joints)."""
    sim = ctx.sim
    data = sim.data
    site_id = ctx.eef_site_id
    site_name = ctx.eef_site_name
    qpos_addrs = ctx.qpos_addrs
    qvel_addrs = ctx.qvel_addrs

    saved_qpos = np.array(data.qpos, copy=True)
    q_arm = _clip_arm_q(np.asarray(seed_q, dtype=np.float64).reshape(7), ctx.joint_limits)
    data.qpos[qpos_addrs] = q_arm
    sim.forward()

    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_wxyz = np.asarray(target_wxyz, dtype=np.float64).reshape(4)

    identity6 = np.eye(6, dtype=np.float64)
    identity3 = np.eye(3, dtype=np.float64)
    lam2 = damping * damping

    try:
        for _ in range(max_iter):
            cur_pos = np.array(data.site_xpos[site_id], dtype=np.float64)
            cur_wxyz = _mat_to_wxyz(data.site_xmat[site_id])

            pos_err = target_pos - cur_pos
            if pos_only:
                if np.linalg.norm(pos_err) < pos_tol:
                    break
                jacp = data.get_site_jacp(site_name)
                j_arm = jacp[:, qvel_addrs]
                jj_t = j_arm @ j_arm.T
                dq = j_arm.T @ np.linalg.solve(jj_t + lam2 * identity3, pos_err)
            else:
                ori_err = _orientation_error_wxyz(cur_wxyz, target_wxyz)
                err = np.concatenate([pos_err, ori_err])
                if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(ori_err) < ori_tol:
                    break

                jacp = data.get_site_jacp(site_name)
                jacr = data.get_site_jacr(site_name)
                j_arm = np.vstack([jacp[:, qvel_addrs], jacr[:, qvel_addrs]])
                jj_t = j_arm @ j_arm.T
                dq = j_arm.T @ np.linalg.solve(jj_t + lam2 * identity6, err)
            q_arm = _clip_arm_q(q_arm + step_scale * dq, ctx.joint_limits)
            data.qpos[qpos_addrs] = q_arm
            sim.forward()
    finally:
        data.qpos[:] = saved_qpos
        sim.forward()

    return q_arm.astype(np.float32)


class _PyrokiBackend:
    def __init__(self) -> None:
        import yourdfpy  # noqa: F401
        import pyroki as pk

        from sawyer_ik_pyroki_snippets import solve_ik_vel_cost

        urdf_path = _resolve_sawyer_urdf()
        urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False)
        self._robot = pk.Robot.from_urdf(urdf)
        self._target_link = "right_hand"
        self._solve = solve_ik_vel_cost
        if self._target_link not in self._robot.links.names:
            raise RuntimeError(
                f"PyRoki URDF missing link {self._target_link!r}; "
                f"available: {self._robot.links.names}"
            )

    def solve(
        self,
        target_pos: np.ndarray,
        target_wxyz: np.ndarray,
        seed_q: np.ndarray,
    ) -> np.ndarray:
        q = self._solve(
            robot=self._robot,
            target_link_name=self._target_link,
            target_position=np.asarray(target_pos, dtype=np.float64),
            target_wxyz=np.asarray(target_wxyz, dtype=np.float64),
            prev_cfg=np.asarray(seed_q, dtype=np.float64),
        )
        return np.asarray(q, dtype=np.float32).reshape(7)


_pyroki_backend: _PyrokiBackend | None = None


def _resolve_sawyer_urdf() -> "Path":
    from pathlib import Path

    env_path = __import__("os").environ.get("SAWYER_URDF_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
    local = Path(__file__).resolve().parent / "assets" / "sawyer" / "sawyer_arm.urdf"
    if local.is_file():
        return local
    try:
        import robosuite

        pkg = Path(robosuite.__file__).resolve().parent
        bundled = (
            pkg
            / "models/assets/bullet_data/sawyer_description/urdf/sawyer_arm.urdf"
        )
        if bundled.is_file():
            return bundled
    except Exception:
        pass
    raise FileNotFoundError("Sawyer URDF not found; set SAWYER_URDF_PATH")


def solve_sawyer_ik(
    ctx: SawyerIkContext,
    target_pos: np.ndarray,
    target_wxyz: np.ndarray,
    seed_q: np.ndarray,
    *,
    backend: IkBackend = "auto",
    pos_only: bool = False,
) -> np.ndarray:
    """Solve 7-DOF arm IK toward target EEF pose."""
    if pos_only or backend == "mujoco":
        return solve_ik_mujoco(
            ctx, target_pos, target_wxyz, seed_q, pos_only=pos_only
        )
    if backend in ("pyroki", "auto"):
        global _pyroki_backend
        try:
            if _pyroki_backend is None:
                _pyroki_backend = _PyrokiBackend()
            return _pyroki_backend.solve(target_pos, target_wxyz, seed_q)
        except Exception:
            if backend == "pyroki":
                raise
    return solve_ik_mujoco(ctx, target_pos, target_wxyz, seed_q, pos_only=pos_only)


def apply_arm_qpos(env, ctx: SawyerIkContext, q_arm: np.ndarray) -> dict:
    """Set Sawyer arm joints in the live sim and return a fresh observation dict."""
    data = ctx.sim.data
    data.qpos[ctx.qpos_addrs] = np.asarray(q_arm, dtype=np.float64).reshape(7)
    ctx.sim.forward()
    env.env._update_observables(force=True)
    return env.env._get_observations()


# RethinkGripper finger joint limits (l upper, r lower) — maximum aperture.
SAWYER_GRIPPER_OPEN_QPOS = np.array([0.020833, -0.020833], dtype=np.float64)


def gripper_aperture_from_obs(obs: dict) -> float:
    g = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)
    return float(g[0] - g[1])


def apply_gripper_qpos(env, qpos: np.ndarray | None = None) -> dict:
    """Force Rethink gripper finger joints to the given qpos (default: full open)."""
    robot = env.env.robots[0]
    sim = robot.sim
    target = SAWYER_GRIPPER_OPEN_QPOS if qpos is None else np.asarray(qpos, dtype=np.float64).reshape(-1)
    addrs = robot._ref_gripper_joint_pos_indexes
    for i, addr in enumerate(addrs):
        sim.data.qpos[addr] = target[i]
    sim.forward()
    env.env._update_observables(force=True)
    return env.env._get_observations()


def fk_proprio_at_joints(ctx: SawyerIkContext, q_arm: np.ndarray, gripper: float) -> np.ndarray:
    """Forward kinematics: arm joints -> 8D proprio state (eef + gripper)."""
    sim = ctx.sim
    data = sim.data
    saved_qpos = np.array(data.qpos, copy=True)
    try:
        data.qpos[ctx.qpos_addrs] = np.asarray(q_arm, dtype=np.float64).reshape(7)
        sim.forward()
        pos = np.array(data.site_xpos[ctx.eef_site_id], dtype=np.float32)
        wxyz = _mat_to_wxyz(data.site_xmat[ctx.eef_site_id])
        aa = wxyz_to_axisangle(wxyz)
        gq = np.array([gripper, gripper], dtype=np.float32)
        return np.concatenate([pos, aa, gq]).astype(np.float32)
    finally:
        data.qpos[:] = saved_qpos
        sim.forward()


def joint_delta_to_osc_action(
    ctx: SawyerIkContext,
    q_current: np.ndarray,
    q_target: np.ndarray,
    gripper: float,
    *,
    pos_scale: np.ndarray,
    rot_scale: np.ndarray,
) -> np.ndarray:
    """Map joint delta (IK output) to LIBERO OSC action via Jacobian."""
    sim = ctx.sim
    data = sim.data
    qpos_addrs = ctx.qpos_addrs
    qvel_addrs = ctx.qvel_addrs
    site_name = ctx.eef_site_name

    q_current = np.asarray(q_current, dtype=np.float64).reshape(7)
    q_target = np.asarray(q_target, dtype=np.float64).reshape(7)
    saved_qpos = np.array(data.qpos, copy=True)
    try:
        data.qpos[qpos_addrs] = q_current
        sim.forward()

        jacp = data.get_site_jacp(site_name)
        jacr = data.get_site_jacr(site_name)
        j_arm = np.vstack([jacp[:, qvel_addrs], jacr[:, qvel_addrs]])

        delta_q = q_target - q_current
        delta_x = j_arm @ delta_q
        dpos = np.clip(delta_x[:3] / pos_scale, -1.0, 1.0)
        drot = np.clip(delta_x[3:6] / rot_scale, -1.0, 1.0)
        return np.concatenate([dpos, drot, np.array([gripper], dtype=np.float32)]).astype(np.float32)
    finally:
        data.qpos[:] = saved_qpos
        sim.forward()
