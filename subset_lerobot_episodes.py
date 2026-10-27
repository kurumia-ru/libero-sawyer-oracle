#!/usr/bin/env python3
"""Make a LeRobot v2.1 subset by hard-linking the first N (or selected) episodes."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=7, help="If --shuffle, sample this many with seed")
    p.add_argument("--shuffle", action="store_true", help="Random sample instead of first N")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if not (src / "meta" / "info.json").exists():
        raise SystemExit(f"missing src meta: {src}")
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"dst exists (pass --overwrite): {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    episodes = [json.loads(l) for l in (src / "meta" / "episodes.jsonl").open()]
    stats_lines = list((src / "meta" / "episodes_stats.jsonl").open()) if (src / "meta" / "episodes_stats.jsonl").exists() else []
    n = min(args.num_episodes, len(episodes))
    if args.shuffle:
        import random

        rng = random.Random(args.seed)
        idxs = sorted(rng.sample(range(len(episodes)), n))
    else:
        idxs = list(range(n))

    # data parquet
    for new_i, old_i in enumerate(idxs):
        src_pq = src / "data" / "chunk-000" / f"episode_{old_i:06d}.parquet"
        if not src_pq.exists():
            # try any chunk
            matches = list((src / "data").rglob(f"episode_{old_i:06d}.parquet"))
            if not matches:
                raise FileNotFoundError(src_pq)
            src_pq = matches[0]
        _link_or_copy(src_pq, dst / "data" / "chunk-000" / f"episode_{new_i:06d}.parquet")

    # videos if present
    vid_root = src / "videos"
    if vid_root.exists():
        for new_i, old_i in enumerate(idxs):
            for cam_dir in sorted(vid_root.glob("chunk-*/observation.images.*")):
                # episode_XXXXXX.mp4 under camera folder
                matches = list(cam_dir.glob(f"episode_{old_i:06d}.*"))
                for m in matches:
                    rel = m.relative_to(src)
                    # rewrite episode index in filename
                    new_name = m.name.replace(f"episode_{old_i:06d}", f"episode_{new_i:06d}")
                    # keep chunk-000 layout
                    parts = list(rel.parts)
                    parts[-1] = new_name
                    if parts[1].startswith("chunk-"):
                        parts[1] = "chunk-000"
                    _link_or_copy(m, dst.joinpath(*parts))

    # meta
    meta = dst / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "meta" / "tasks.jsonl", meta / "tasks.jsonl")
    if (src / "meta" / "scripted_export_summary_0.json").exists():
        shutil.copy2(src / "meta" / "scripted_export_summary_0.json", meta / "scripted_export_summary_0.json")

    new_eps = []
    total_frames = 0
    with (meta / "episodes.jsonl").open("w") as f_ep, (meta / "episodes_stats.jsonl").open("w") as f_st:
        for new_i, old_i in enumerate(idxs):
            ep = dict(episodes[old_i])
            ep["episode_index"] = new_i
            f_ep.write(json.dumps(ep) + "\n")
            new_eps.append(ep)
            total_frames += int(ep.get("length", 0))
            if old_i < len(stats_lines):
                st = json.loads(stats_lines[old_i])
                if isinstance(st, dict) and "episode_index" in st:
                    st["episode_index"] = new_i
                f_st.write(json.dumps(st) + "\n")

    info = json.loads((src / "meta" / "info.json").read_text())
    info["total_episodes"] = len(new_eps)
    info["total_frames"] = total_frames
    info["total_chunks"] = 1
    info["chunks_size"] = max(info.get("chunks_size", 1000), len(new_eps))
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")

    # Prefer parent norm stats for small subsets (more stable).
    if (src / "norm_stats.json").exists():
        shutil.copy2(src / "norm_stats.json", dst / "norm_stats.json")

    mapping = {"src": str(src), "dst": str(dst), "num": len(idxs), "src_episode_indices": idxs}
    (meta / "subset_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    print(json.dumps({"dst": str(dst), "episodes": len(idxs), "frames": total_frames, "indices": idxs[:10]}, indent=2))


if __name__ == "__main__":
    main()
