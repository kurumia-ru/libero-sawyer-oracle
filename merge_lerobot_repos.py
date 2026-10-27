#!/usr/bin/env python3
"""Merge a base LeRobot repo with extra episodes (e.g. DAgger expert takeovers).

Copies the base repo, then appends episodes from --extra (optionally filtered /
repeated). Recomputes nothing here — caller should run compute_norm_stats after.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def _to_hwc(img) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"expected image ndim=3, got shape={arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _expert_ok(summary_ep: dict | None) -> bool:
    if summary_ep is None:
        return True
    info = summary_ep.get("expert_info") or {}
    if info.get("skipped"):
        return False
    if info.get("placed_xy") is True:
        return True
    if float(info.get("reward") or 0) > 0:
        return True
    if info.get("done") is True and info.get("phase") == "done":
        return True
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True, help="Base LeRobot repo (copied).")
    p.add_argument("--extra", type=Path, required=True, help="Extra LeRobot repo to append.")
    p.add_argument("--out", type=Path, required=True, help="Output merged repo.")
    p.add_argument("--extra-repeat", type=int, default=1, help="Repeat each kept extra episode.")
    p.add_argument(
        "--require-expert-success",
        action="store_true",
        help="If extra/meta/dagger_takeover_summary.json exists, keep only successful takeovers.",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not args.base.is_dir():
        raise SystemExit(f"missing base: {args.base}")
    if not args.extra.is_dir():
        raise SystemExit(f"missing extra: {args.extra}")
    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out} exists; pass --overwrite")
        shutil.rmtree(args.out)

    print(f"[merge] copy {args.base} -> {args.out}", flush=True)
    shutil.copytree(args.base, args.out, symlinks=False)
    # Stale norms must be recomputed on the merged mix.
    for stale in ("norm_stats.json",):
        pth = args.out / stale
        if pth.is_file():
            pth.unlink()
            print(f"[merge] removed stale {pth.name}", flush=True)

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    summary_path = args.extra / "meta" / "dagger_takeover_summary.json"
    summary_eps: list[dict] | None = None
    if summary_path.is_file():
        summary_eps = json.loads(summary_path.read_text()).get("episodes") or []

    extra = LeRobotDataset(str(args.extra))
    n_extra = int(extra.meta.total_episodes)
    keep: list[int] = []
    for ep_idx in range(n_extra):
        meta_ep = summary_eps[ep_idx] if summary_eps and ep_idx < len(summary_eps) else None
        if args.require_expert_success and not _expert_ok(meta_ep):
            print(f"[merge] skip extra_ep={ep_idx} (expert not successful)", flush=True)
            continue
        keep.append(ep_idx)

    dst = LeRobotDataset(str(args.out))
    dst.start_image_writer(num_processes=1, num_threads=4)

    appended = 0
    records = []
    for ep_idx in keep:
        fr_from = int(extra.episode_data_index["from"][ep_idx])
        fr_to = int(extra.episode_data_index["to"][ep_idx])
        frames = []
        for i in range(fr_from, fr_to):
            fr = extra[i]
            task = str(fr.get("task") or "task").strip() or "task"
            frames.append(
                {
                    "image": _to_hwc(fr["image"]),
                    "wrist_image": _to_hwc(fr["wrist_image"]),
                    "state": np.asarray(fr["state"], dtype=np.float32),
                    "actions": np.asarray(fr["actions"], dtype=np.float32),
                    "task": task,
                }
            )
        for r in range(max(1, args.extra_repeat)):
            for fr in frames:
                dst.add_frame(dict(fr))
            dst.save_episode()
            appended += 1
            records.append({"extra_ep": ep_idx, "repeat": r, "n_frames": len(frames)})
        print(
            f"[merge] appended extra_ep={ep_idx} frames={len(frames)} x{args.extra_repeat}",
            flush=True,
        )

    dst.stop_image_writer()

    out_meta = {
        "base": str(args.base),
        "extra": str(args.extra),
        "out": str(args.out),
        "extra_repeat": args.extra_repeat,
        "require_expert_success": bool(args.require_expert_success),
        "n_extra_src": n_extra,
        "n_extra_kept": len(keep),
        "n_appended_episodes": appended,
        "records": records,
    }
    meta_dir = args.out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "merge_summary.json").write_text(json.dumps(out_meta, indent=2))
    print(
        f"[merge] DONE out={args.out} appended_eps={appended} "
        f"(kept {len(keep)}/{n_extra} extra)",
        flush=True,
    )


if __name__ == "__main__":
    main()
