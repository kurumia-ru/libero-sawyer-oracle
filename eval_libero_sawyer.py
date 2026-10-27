#!/usr/bin/env python3
"""LIBERO policy eval with Sawyer (or default Panda) robot variant."""
from __future__ import annotations

import collections
import dataclasses
import logging
import math
import os
import pathlib
import sys

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

# libero_robot_variants lives next to this script on the eval server
_EVAL_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_ROOT))
from libero_robot_variants import register_robot_variants, robots_arg_for_variant  # noqa: E402

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
# Match LIBERO / π0.5 control clock (20Hz); one policy/env step == one dataset frame.
CONTROL_FREQ = 20
DATASET_FPS = 20

# Post-grasp stage metrics (task1 pick-and-place funnel).
# near_grasp / grasp_ok already exist; these refine what happens after grasp.
LIFT_OK_HEIGHT_M = 0.08  # object rose >= 8cm from start
TRANSPORT_OK_DELTA_M = 0.08  # object got >= 8cm closer to basket than start
PLACE_NEAR_M = 0.05  # object within 5cm of basket XY


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "data/libero/videos"
    seed: int = 7
    robot_variant: str = "default"  # default | sawyer
    # If set, only evaluate these task indices (e.g. 0 1).
    task_ids: list[int] | None = None
    # Match scripted Sawyer demo export: home joint teleport after stabilize.
    sawyer_home_w30: bool = False
    ik_backend: str = "mujoco"
    # Sawyer action execution: delta_osc (default LIBERO) or abs_eef (IK teleport + gripper).
    sawyer_action_space: str = "delta_osc"
    # Ablation: when EEF is near basket XY, force gripper open (diagnose release vs grasp).
    force_open_near_basket: bool = False
    force_open_xy_thresh: float = 0.12
    control_freq: int = CONTROL_FREQ
    # Closed-loop stabilization: optional EMA smoothing on policy actions.
    # 1.0 disables smoothing; lower values add more inertia.
    action_ema_alpha: float = 1.0
    # Override episode step budget. None = suite default (Franka-scale).
    # Sawyer scripted osc_v8 demos are ~1000–1100 steps; suite default 280 cuts mid-grasp.
    max_steps: int | None = None
    # If set, evaluate these LIBERO init indices (in order) instead of 0..N-1.
    # Length overrides num_trials_per_task when provided.
    init_indices: list[int] | None = None


def _get_libero_env(
    task,
    resolution,
    seed,
    robot_variant: str,
    control_freq: int = CONTROL_FREQ,
    horizon: int | None = None,
):
    register_robot_variants()
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_kwargs = dict(
        bddl_file_name=task_bddl_file,
        robots=robots_arg_for_variant(robot_variant),
        camera_heights=resolution,
        camera_widths=resolution,
        control_freq=control_freq,
    )
    if horizon is not None:
        env_kwargs["horizon"] = int(horizon)
    env = OffScreenRenderEnv(**env_kwargs)
    env.seed(seed)
    return env, task.language


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _suite_key(task_suite_name: str) -> str:
    return {
        "libero_spatial": "spatial",
        "libero_object": "object",
        "libero_goal": "goal",
        "libero_10": "long",
    }.get(task_suite_name, "object")


def _basket_xy(env) -> np.ndarray | None:
    try:
        from libero_grasp_smoke import _basket_object_name, _object_body_id, _object_geom_bounds

        sim = env.env.robots[0].sim
        bid = _object_body_id(sim, _basket_object_name(env))
        return np.asarray(_object_geom_bounds(sim, bid)["xy"], dtype=np.float64)
    except Exception:
        return None


def _target_obj_z(env) -> float | None:
    try:
        from libero_grasp_smoke import _object_body_id, _target_object_name

        sim = env.env.robots[0].sim
        tid = _object_body_id(sim, _target_object_name(env))
        return float(sim.data.body_xpos[tid][2])
    except Exception:
        return None


def _target_obj_xy(env) -> np.ndarray | None:
    try:
        from libero_grasp_smoke import _object_body_id, _target_object_name

        sim = env.env.robots[0].sim
        tid = _object_body_id(sim, _target_object_name(env))
        return np.asarray(sim.data.body_xpos[tid][:2], dtype=np.float64)
    except Exception:
        return None


def _maybe_force_open_near_basket(action: np.ndarray, obs: dict, basket_xy: np.ndarray | None, thresh: float) -> np.ndarray:
    """Overwrite gripper to open if EEF XY is within thresh of basket center."""
    if basket_xy is None:
        return action
    eef_xy = np.asarray(obs["robot0_eef_pos"][:2], dtype=np.float64)
    if float(np.linalg.norm(eef_xy - basket_xy)) <= thresh:
        out = np.asarray(action, dtype=np.float32).copy()
        out[-1] = np.float32(-1.0)
        return out
    return action


def _maybe_sawyer_home_w30(env, obs, *, task_id: int, task_suite_name: str, robot_variant: str, enabled: bool, ik_backend: str):
    if not enabled or robot_variant != "sawyer":
        return obs
    from generate_b_demos import (
        apply_start_pose_strategy,
        sawyer_home_qpos,
        sawyer_video5s_qpos,
    )
    from libero_grasp_smoke import _franka_frame0_for_task
    from sawyer_ik import apply_arm_qpos, apply_gripper_qpos, ik_context_from_env

    # Must match export_scripted_demos.py / preview_scripted_demos.py
    presets: dict[int, dict] = {
        0: {"force_gripper_open": True, "start_pose_strategy": "home_w30"},
        1: {"force_gripper_open": False, "start_pose_strategy": "video5s"},
        2: {"force_gripper_open": True, "start_pose_strategy": "home_w30"},
        3: {"force_gripper_open": True, "start_pose_strategy": "home_w30"},
        4: {"force_gripper_open": True, "start_pose_strategy": "home_w30"},
        5: {"force_gripper_open": True, "start_pose_strategy": "home_w30"},
        6: {"force_gripper_open": False, "start_pose_strategy": "home_w30"},
        7: {"force_gripper_open": False, "start_pose_strategy": "home_w30"},
        8: {"force_gripper_open": False, "start_pose_strategy": "home_w30"},
        9: {"force_gripper_open": False, "start_pose_strategy": "home_w30"},
    }
    preset = presets.get(task_id, {"start_pose_strategy": "home_w30", "force_gripper_open": False})
    suite = _suite_key(task_suite_name)
    strat = preset.get("start_pose_strategy", "home_w30")
    if strat == "home_w30":
        ik_ctx = ik_context_from_env(env)
        obs = apply_arm_qpos(env, ik_ctx, sawyer_home_qpos())
    elif strat == "video5s":
        # Fixed task1 start does not need a Franka parquet frame (and therefore
        # keeps the lightweight Python 3.8 eval environment pyarrow-free).
        ik_ctx = ik_context_from_env(env)
        obs = apply_arm_qpos(env, ik_ctx, sawyer_video5s_qpos())
        obs = apply_gripper_qpos(env)
    elif strat:
        ik_ctx = ik_context_from_env(env)
        frame0 = _franka_frame0_for_task(suite, task_id)
        obs, _align = apply_start_pose_strategy(
            env,
            ik_ctx,
            obs,
            frame0,
            strat,
            ik_backend=ik_backend,
            warmup_steps_override=0,
        )
    if preset.get("force_gripper_open"):
        obs = apply_gripper_qpos(env)
    return obs


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)
    os.environ.setdefault("MUJOCO_GL", "egl")

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_ids = args.task_ids if args.task_ids is not None else list(range(num_tasks_in_suite))
    for tid in task_ids:
        if tid < 0 or tid >= num_tasks_in_suite:
            raise ValueError(f"task_id {tid} out of range [0, {num_tasks_in_suite})")
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    max_steps = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[args.task_suite_name]
    if args.max_steps is not None:
        max_steps = int(args.max_steps)
    elif args.robot_variant == "sawyer":
        # Retrack osc demos (~path_step 8mm) need ~1k steps; 280 ends before lift (~425).
        max_steps = max(max_steps, 1200)
    env_horizon = max(max_steps + args.num_steps_wait + 50, 1000)
    logging.info(
        "episode budget max_steps=%d env_horizon=%d variant=%s",
        max_steps,
        env_horizon,
        args.robot_variant,
    )

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    total_episodes, total_successes = 0, 0
    stage_totals = {
        "near_grasp": 0,
        "grasp_ok": 0,
        "lift_ok": 0,
        "transport_ok": 0,
        "place_near": 0,
        "success": 0,
    }
    sawyer_ik_ctx = None
    if args.robot_variant == "sawyer" and args.sawyer_action_space == "abs_eef":
        # Built per-env below after env creation.
        pass

    for task_id in tqdm.tqdm(task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(
            task,
            LIBERO_ENV_RESOLUTION,
            args.seed,
            args.robot_variant,
            control_freq=args.control_freq,
            horizon=env_horizon,
        )
        task_episodes, task_successes = 0, 0
        task_stages = {k: 0 for k in stage_totals}
        if args.robot_variant == "sawyer" and args.sawyer_action_space == "abs_eef":
            from sawyer_ik import ik_context_from_env

            env.reset()
            sawyer_ik_ctx = ik_context_from_env(env)

        if args.init_indices is not None:
            trial_init_indices = [int(i) for i in args.init_indices]
            logging.info(
                "using explicit init_indices n=%d first=%s last=%s",
                len(trial_init_indices),
                trial_init_indices[:5],
                trial_init_indices[-5:],
            )
        else:
            trial_init_indices = list(range(args.num_trials_per_task))
        n_inits_avail = len(initial_states)
        # Match export_scripted_demos.py: init_index % len(inits)
        resolved_inits = [int(i) % n_inits_avail for i in trial_init_indices]
        if any(int(i) >= n_inits_avail for i in trial_init_indices):
            logging.info(
                "wrapped init_indices with %% %d (export-compatible); raw last=%s -> %s",
                n_inits_avail,
                trial_init_indices[-6:],
                resolved_inits[-6:],
            )

        for episode_idx, init_index in enumerate(tqdm.tqdm(resolved_inits)):
            env.reset()
            if args.robot_variant == "sawyer" and args.sawyer_action_space == "abs_eef":
                from sawyer_ik import ik_context_from_env

                sawyer_ik_ctx = ik_context_from_env(env)
            action_plan = collections.deque()
            prev_action: np.ndarray | None = None
            obs = env.set_init_state(initial_states[init_index])
            t = 0
            done = False
            replay_images = []
            basket_xy = _basket_xy(env)
            z0 = _target_obj_z(env)
            zmax = z0 if z0 is not None else None
            min_basket_dist = None
            forced_open_steps = 0
            min_eef_obj = None
            min_eef_obj_when_close = None
            close_near_obj_steps = 0
            grip_close_steps = 0
            max_obj_basket = None
            min_obj_basket = None
            grasped_frames = 0  # lift>5cm and eef within 8cm of object
            # Baseline obj–basket distance; refreshed after settle/home.
            start_obj_basket = None
            if basket_xy is not None:
                obj_xy0 = _target_obj_xy(env)
                if obj_xy0 is not None:
                    start_obj_basket = float(np.linalg.norm(obj_xy0 - basket_xy))

            applied_home_w30 = False
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                if not applied_home_w30:
                    obs = _maybe_sawyer_home_w30(
                        env,
                        obs,
                        task_id=task_id,
                        task_suite_name=args.task_suite_name,
                        robot_variant=args.robot_variant,
                        enabled=args.sawyer_home_w30,
                        ik_backend=args.ik_backend,
                    )
                    applied_home_w30 = True
                    bx = _basket_xy(env)
                    if bx is not None:
                        basket_xy = bx
                    # Refresh after home — object Z can settle during the teleport.
                    z0 = _target_obj_z(env)
                    zmax = z0 if z0 is not None else zmax
                    obj_xy0 = _target_obj_xy(env)
                    if basket_xy is not None and obj_xy0 is not None:
                        start_obj_basket = float(np.linalg.norm(obj_xy0 - basket_xy))

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(task_description),
                    }
                    action_chunk = client.infer(element)["actions"]
                    assert len(action_chunk) >= args.replan_steps
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = np.asarray(action_plan.popleft(), dtype=np.float32)
                if 0.0 < args.action_ema_alpha < 1.0 and prev_action is not None:
                    a = float(args.action_ema_alpha)
                    action = (a * action + (1.0 - a) * prev_action).astype(np.float32)
                prev_action = action.copy()
                if args.force_open_near_basket:
                    before = float(action[-1])
                    action = _maybe_force_open_near_basket(action, obs, basket_xy, args.force_open_xy_thresh)
                    if float(action[-1]) < -0.5 and before > 0.0:
                        forced_open_steps += 1
                if args.robot_variant == "sawyer" and args.sawyer_action_space == "abs_eef":
                    from scripted_action_spaces import step_abs_eef_ik

                    obs, reward, done = step_abs_eef_ik(env, obs, action, sawyer_ik_ctx)
                else:
                    obs, reward, done, info = env.step(action.tolist())
                z_now = _target_obj_z(env)
                if z_now is not None and zmax is not None:
                    zmax = max(zmax, z_now)
                eef_xy = np.asarray(obs["robot0_eef_pos"][:2], dtype=np.float64)
                obj_xy = _target_obj_xy(env)
                grip_cmd = float(action[-1])
                if grip_cmd > 0.0:
                    grip_close_steps += 1
                if obj_xy is not None:
                    d_eo = float(np.linalg.norm(eef_xy - obj_xy))
                    min_eef_obj = d_eo if min_eef_obj is None else min(min_eef_obj, d_eo)
                    if grip_cmd > 0.0:
                        min_eef_obj_when_close = (
                            d_eo if min_eef_obj_when_close is None else min(min_eef_obj_when_close, d_eo)
                        )
                        if d_eo <= 0.06:
                            close_near_obj_steps += 1
                    if z_now is not None and z0 is not None and (z_now - z0) > 0.05 and d_eo <= 0.08:
                        grasped_frames += 1
                if basket_xy is not None:
                    d = float(np.linalg.norm(eef_xy - basket_xy))
                    min_basket_dist = d if min_basket_dist is None else min(min_basket_dist, d)
                    if obj_xy is not None:
                        dob = float(np.linalg.norm(obj_xy - basket_xy))
                        min_obj_basket = dob if min_obj_basket is None else min(min_obj_basket, dob)
                        max_obj_basket = dob if max_obj_basket is None else max(max_obj_basket, dob)
                replay_images.append(img)
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            task_episodes += 1
            total_episodes += 1
            lift = (zmax - z0) if (zmax is not None and z0 is not None) else None
            grasp_ok = grasped_frames >= 5
            near_grasp_attempt = (min_eef_obj_when_close is not None and min_eef_obj_when_close <= 0.06) or (
                close_near_obj_steps >= 3
            )
            lift_ok = bool(grasp_ok and lift is not None and lift >= LIFT_OK_HEIGHT_M)
            transport_delta = None
            if start_obj_basket is not None and min_obj_basket is not None:
                transport_delta = float(start_obj_basket - min_obj_basket)
            # Funnel: each stage requires the previous ones.
            transport_ok = bool(
                lift_ok and transport_delta is not None and transport_delta >= TRANSPORT_OK_DELTA_M
            )
            place_near = bool(
                transport_ok and min_obj_basket is not None and min_obj_basket <= PLACE_NEAR_M
            )

            ep_stages = {
                "near_grasp": bool(near_grasp_attempt),
                "grasp_ok": bool(grasp_ok),
                "lift_ok": bool(lift_ok),
                "transport_ok": bool(transport_ok),
                "place_near": bool(place_near),
                "success": bool(done),
            }
            for k, v in ep_stages.items():
                if v:
                    task_stages[k] += 1
                    stage_totals[k] += 1

            logging.info(
                "ep%d init=%d success=%s near_grasp=%s grasp_ok=%s lift_ok=%s transport_ok=%s place_near=%s "
                "lift=%s start_obj_basket=%s transport_delta=%s "
                "min_eef_obj=%s min_eef_obj_close=%s close_near_obj=%d gripped=%d "
                "min_basket=%s min_obj_basket=%s grasped_frames=%d forced_open=%d",
                episode_idx,
                init_index,
                bool(done),
                ep_stages["near_grasp"],
                ep_stages["grasp_ok"],
                ep_stages["lift_ok"],
                ep_stages["transport_ok"],
                ep_stages["place_near"],
                f"{lift:.3f}" if lift is not None else "n/a",
                f"{start_obj_basket:.3f}" if start_obj_basket is not None else "n/a",
                f"{transport_delta:.3f}" if transport_delta is not None else "n/a",
                f"{min_eef_obj:.3f}" if min_eef_obj is not None else "n/a",
                f"{min_eef_obj_when_close:.3f}" if min_eef_obj_when_close is not None else "n/a",
                close_near_obj_steps,
                grip_close_steps,
                f"{min_basket_dist:.3f}" if min_basket_dist is not None else "n/a",
                f"{min_obj_basket:.3f}" if min_obj_basket is not None else "n/a",
                grasped_frames,
                forced_open_steps,
            )
            suffix = f"{args.robot_variant}_task{task_id}_ep{episode_idx}_init{init_index}"
            if replay_images:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=DATASET_FPS,
                )

        logging.info(f"Task {task_id} success: {task_successes}/{task_episodes}")
        logging.info(
            "Task %d STAGE_RATES near_grasp=%d/%d grasp_ok=%d/%d lift_ok=%d/%d "
            "transport_ok=%d/%d place_near=%d/%d success=%d/%d",
            task_id,
            task_stages["near_grasp"],
            task_episodes,
            task_stages["grasp_ok"],
            task_episodes,
            task_stages["lift_ok"],
            task_episodes,
            task_stages["transport_ok"],
            task_episodes,
            task_stages["place_near"],
            task_episodes,
            task_stages["success"],
            task_episodes,
        )
        env.close()

    rate = total_successes / max(total_episodes, 1)
    logging.info(
        f"Total success rate ({args.robot_variant}, tasks={task_ids}): "
        f"{rate:.3f} ({total_successes}/{total_episodes})"
    )
    logging.info(
        "STAGE_RATES near_grasp=%d/%d grasp_ok=%d/%d lift_ok=%d/%d "
        "transport_ok=%d/%d place_near=%d/%d success=%d/%d "
        "(thresholds: lift>=%.2fm, transport_delta>=%.2fm, place_near<=%.2fm)",
        stage_totals["near_grasp"],
        total_episodes,
        stage_totals["grasp_ok"],
        total_episodes,
        stage_totals["lift_ok"],
        total_episodes,
        stage_totals["transport_ok"],
        total_episodes,
        stage_totals["place_near"],
        total_episodes,
        stage_totals["success"],
        total_episodes,
        LIFT_OK_HEIGHT_M,
        TRANSPORT_OK_DELTA_M,
        PLACE_NEAR_M,
    )
    print(
        f"SUCCESS_RATE {rate:.4f} variant={args.robot_variant} "
        f"tasks={task_ids} episodes={total_episodes}"
    )
    print(
        "STAGE_RATES "
        f"near_grasp={stage_totals['near_grasp']}/{total_episodes} "
        f"grasp_ok={stage_totals['grasp_ok']}/{total_episodes} "
        f"lift_ok={stage_totals['lift_ok']}/{total_episodes} "
        f"transport_ok={stage_totals['transport_ok']}/{total_episodes} "
        f"place_near={stage_totals['place_near']}/{total_episodes} "
        f"success={stage_totals['success']}/{total_episodes}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
