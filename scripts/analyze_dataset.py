#!/usr/bin/env python3
"""Analyze motion CSV dataset lengths and relocate long motions."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class MotionStat:
    path: Path
    frames: int

    def seconds(self, fps: float) -> float:
        return self.frames / fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze motion CSV files under a directory, compute average length, "
            "and move motions longer than a threshold to a target directory while preserving structure."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root directory containing motion CSV files (subdirectories included).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where long motions (exceeding threshold) will be moved.",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.csv",
        help="Glob pattern to match motion files (default: *.csv).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate of the motions (default: 30).",
    )
    parser.add_argument(
        "--threshold_secs",
        type=float,
        default=10.0,
        help="Threshold in seconds above which motions are moved (default: 10).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report actions without moving files.",
    )
    return parser.parse_args()


def iter_motion_files(root: Path, pattern: str) -> Iterable[Path]:
    yield from root.rglob(pattern)


def count_frames(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def gather_stats(input_dir: Path, pattern: str) -> list[MotionStat]:
    stats: list[MotionStat] = []
    for csv_path in iter_motion_files(input_dir, pattern):
        if not csv_path.is_file():
            continue
        frames = count_frames(csv_path)
        stats.append(MotionStat(path=csv_path, frames=frames))
    return stats


def move_long_motions(
    stats: Iterable[MotionStat],
    input_dir: Path,
    output_dir: Path,
    threshold_frames: int,
    fps: float,
    dry_run: bool,
) -> int:
    moved = 0
    for motion in stats:
        if motion.frames <= threshold_frames:
            continue
        relative_path = motion.path.relative_to(input_dir)
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            print(
                f"[DRY]: Would move {motion.path} -> {destination} "
                f"(frames={motion.frames}, seconds={motion.frames / fps:.2f})"
            )
        else:
            print(
                f"[MOVE]: {motion.path} -> {destination} "
                f"(frames={motion.frames}, seconds={motion.frames / fps:.2f})"
            )
            shutil.move(str(motion.path), str(destination))
        moved += 1
    return moved


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = gather_stats(input_dir, args.file_pattern)
    if not stats:
        print(f"[WARN]: No files matched under {input_dir} with pattern '{args.file_pattern}'")
        return

    total_frames = sum(m.frames for m in stats)
    avg_frames = total_frames / len(stats)
    avg_seconds = avg_frames / args.fps

    print(f"[INFO]: Analyzed {len(stats)} motions.")
    print(f"[INFO]: Average length: {avg_frames:.2f} frames ({avg_seconds:.2f} sec @ {args.fps} fps)")

    threshold_frames = int(args.threshold_secs * args.fps)
    print(f"[INFO]: Moving motions longer than {args.threshold_secs} sec (> {threshold_frames} frames).")

    moved = move_long_motions(
        stats,
        input_dir=input_dir,
        output_dir=output_dir,
        threshold_frames=threshold_frames,
        fps=args.fps,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"[INFO]: Dry-run complete. Motions exceeding threshold: {moved}")
    else:
        print(f"[INFO]: Moved {moved} motions exceeding threshold to {output_dir}")


if __name__ == "__main__":
    main()

