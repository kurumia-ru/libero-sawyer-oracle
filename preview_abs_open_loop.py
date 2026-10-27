#!/usr/bin/env python3
"""Regenerate and render one scripted abs_eef open-loop replay."""
from __future__ import annotations

import argparse
import pathlib
import sys

import imageio
import numpy as np
import torch

# LIBERO init-state files are trusted local benchmark assets. PyTorch 2.6
# otherwise defaults to weights_only=True and rejects their NumPy arrays.
_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(  # type: ignore[assignment]
    *a, **{**k, "weights_only": k.get("weights_only", False)}
)

EVAL_ROOT = pathlib.Path.home() / "openpi_ws" / "openpi_eval"
LIBERO_ROOT = pathlib.Path.home() / "openpi_ws" / "openpi" / "third_party" / "libero"
sys.path[:0] = [str(EVAL_ROOT), str(LIBERO_ROOT)]

from export_scripted_demos import (  # noqa: E402
    SCRIPTED_TASK_PRESETS,
    _retrack_osc_frames,
)
from generate_b_demos import (  # noqa: E402
    LIBERO_DUMMY,
    NUM_STABILIZE,
    _make_env,
    _render_pair,
)
from libero.libero import benchmark  # noqa: E402
from libero_grasp_smoke import (  # noqa: E402
    _basket_object_name,
    _object_body_id,
    _object_geom_bounds,
    _phases_for_mode,
    _target_object_name,
    run_grasp_attempt,
)
from preview_scripted_demos import _apply_export_setup  # noqa: E402
from sawyer_ik import ik_context_from_env  # noqa: E402
from scripted_action_spaces import reencode_abs_eef_from_qpos, step_abs_eef_ik  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-index", type=int, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--start-pose-strategy", default="video5s")
    parser.add_argument(
        "--dump-step",
        type=int,
        default=None,
        help="Print robot qpos/EEF at this replay step.",
    )
    args = parser.parse_args()

    preset = SCRIPTED_TASK_PRESETS[1]
    style = preset["grasp_style"]
    live = run_grasp_attempt(
        robot="sawyer",
        suite="object",
        task_id=1,
        init_index=args.init_index,
        start_pose_strategy=args.start_pose_strategy,
        phases=_phases_for_mode(
            "ik_set", 0.05, style, place_to_basket=True, motion_profile="compact"
        ),
        record_video=False,
        record_demo=True,
        ik_backend="mujoco",
        grasp_mode="ik_set",
        grasp_style=style,
        grasp_z=0.05,
        rim_inset=0.004,
        bite_depth=0.003,
        insert_depth=preset.get("insert_depth", 0.004),
        cap_drop=preset.get("cap_drop", 0.023),
        force_gripper_open=preset.get("force_gripper_open", False),
        side_x=preset.get("side_x"),
        side_z=preset.get("side_z"),
        place_to_basket=True,
        motion_profile="compact",
    )
    print(
        "LIVE_RESULT "
        f"init={args.init_index} passed={live.get('passed')} "
        f"lift={live.get('lift_dz')} placed={live.get('placed')} "
        f"frames={len(live.get('demo_frames') or [])}"
    )
    dense = _retrack_osc_frames(
        live["demo_frames"],
        suite="object",
        task_id=1,
        init_index=args.init_index,
        ik_backend="mujoco",
        pad_open=True,
    )
    actions = reencode_abs_eef_from_qpos(dense)

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(1)
    init_states = suite.get_task_init_states(1)
    env = _make_env(task, "sawyer", 7, horizon=4000)
    env.reset()
    obs = env.set_init_state(init_states[args.init_index])
    for _ in range(NUM_STABILIZE):
        obs, _, _, _ = env.step(LIBERO_DUMMY)
    obs, _ = _apply_export_setup(
        env, obs, task_id=1, suite="object", ik_backend="mujoco"
    )
    ik_ctx = ik_context_from_env(env)
    sim = env.env.robots[0].sim
    target_bid = _object_body_id(sim, _target_object_name(env))
    basket_bid = _object_body_id(sim, _basket_object_name(env))
    basket_xy = np.asarray(_object_geom_bounds(sim, basket_bid)["xy"])
    z0 = float(sim.data.body_xpos[target_bid][2])
    zmax = z0
    min_obj_basket = float("inf")
    frames: list[np.ndarray] = []
    reward = 0.0
    done = False
    for step_i, frame in enumerate(actions):
        if args.dump_step is not None and step_i == args.dump_step:
            print(
                "DUMP_STEP "
                f"init={args.init_index} step={step_i} "
                f"eef={np.asarray(obs['robot0_eef_pos']).tolist()} "
                f"qpos={np.asarray(obs['robot0_joint_pos']).tolist()}"
            )
        image, _ = _render_pair(obs)
        frames.append(np.asarray(image))
        try:
            obs, reward, done = step_abs_eef_ik(
                env, obs, np.asarray(frame["actions"]), ik_ctx
            )
        except Exception as exc:
            print(f"step error: {exc!r}")
            break
        obj_pos = np.asarray(sim.data.body_xpos[target_bid])
        zmax = max(zmax, float(obj_pos[2]))
        min_obj_basket = min(
            min_obj_basket, float(np.linalg.norm(obj_pos[:2] - basket_xy))
        )
        if reward > 0:
            break
        if done:
            break
    image, _ = _render_pair(obs)
    frames.extend([np.asarray(image)] * 20)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=10, codec="libx264", quality=8)
    print(
        f"RESULT init={args.init_index} reward={reward:.3f} done={done} "
        f"lift={zmax-z0:.3f} min_obj_basket={min_obj_basket:.3f} "
        f"steps={len(frames)-20} video={args.out}"
    )
    env.close()


if __name__ == "__main__":
    main()
