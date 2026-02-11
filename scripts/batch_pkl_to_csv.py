#!/usr/bin/env python3
"""Batch convert PKL motion files into CSV format compatible with csv_to_npz.py."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np


REQUIRED_KEYS = ("root_pos", "root_rot", "dof_pos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert PKL motions in a folder to CSV format."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing PKL motion files.",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.pkl",
        help="Glob pattern to match input files (default: *.pkl).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output folder (default: same as input_dir).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files if present.",
    )
    return parser.parse_args()


def find_motion_sequences(data: object, prefix: str = "") -> List[Tuple[str | None, dict]]:
    """Recursively collect dicts that contain the required motion keys."""
    sequences: List[Tuple[str | None, dict]] = []
    if isinstance(data, dict):
        if all(key in data for key in REQUIRED_KEYS):
            sequences.append((prefix or None, data))
        else:
            for key, value in data.items():
                sub_prefix = f"{prefix}_{key}" if prefix else str(key)
                sequences.extend(find_motion_sequences(value, sub_prefix))
    elif isinstance(data, (list, tuple)):
        for idx, value in enumerate(data):
            sub_prefix = f"{prefix}_{idx}" if prefix else str(idx)
            sequences.extend(find_motion_sequences(value, sub_prefix))
    return sequences


def motion_to_array(motion: dict) -> np.ndarray:
    """Stack root position, root rotation, and joint DOF positions into rows."""
    root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float32)

    if not (root_pos.shape[0] == root_rot.shape[0] == dof_pos.shape[0]):
        raise ValueError(
            f"Frame count mismatch: root_pos {root_pos.shape}, root_rot {root_rot.shape}, dof_pos {dof_pos.shape}"
        )

    if root_pos.shape[1] != 3 or root_rot.shape[1] != 4:
        raise ValueError(
            f"Unexpected root shapes: root_pos {root_pos.shape}, root_rot {root_rot.shape}"
        )

    return np.concatenate([root_pos, root_rot, dof_pos], axis=1)


def process_file(input_path: Path, input_root: Path, output_dir: Path, overwrite: bool) -> None:
    with input_path.open("rb") as f:
        data = pickle.load(f)

    sequences = find_motion_sequences(data)
    if not sequences:
        raise ValueError(f"No motion sequences found in {input_path}")

    if len(sequences) > 1:
        raise ValueError(
            f"Multiple motion sequences found in {input_path}; cannot map to single CSV file."
        )

    _, motion = sequences[0]

    relative_path = input_path.relative_to(input_root)
    output_path = (output_dir / relative_path).with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        print(f"[WARN]: Skipping existing file without overwrite: {output_path}")
        return

    array = motion_to_array(motion)
    np.savetxt(output_path, array, delimiter=",", fmt="%.18e")

    fps = motion.get("fps", None)
    info = f"[INFO]: Saved {output_path} (frames={array.shape[0]}"
    if fps is not None:
        info += f", fps={fps})"
    else:
        info += ")"
    print(info)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir
    )

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = args.file_pattern
    pkl_files = sorted(input_dir.rglob(pattern))
    if not pkl_files:
        print(f"[WARN]: No files matched under {input_dir} with pattern '{pattern}'")
        return

    for file_path in pkl_files:
        process_file(file_path, input_dir, output_dir, args.overwrite)


if __name__ == "__main__":
    main()

