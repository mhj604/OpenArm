#!/usr/bin/env python3
"""Validate the expected two-camera, 16-state, 16-action LeRobot schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


EXPECTED_CAMERAS = {"observation.images.chest", "observation.images.wrist_left"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    features = dataset.features
    camera_keys = {key for key in features if key.startswith("observation.images.")}
    if camera_keys != EXPECTED_CAMERAS:
        raise ValueError(f"camera schema mismatch: expected {sorted(EXPECTED_CAMERAS)}, got {sorted(camera_keys)}")
    if any("right" in key and "image" in key for key in features):
        raise ValueError("right-wrist camera data is forbidden")

    state_shape = tuple(features["observation.state"]["shape"])
    action_shape = tuple(features["action"]["shape"])
    if state_shape != (16,) or action_shape != (16,):
        raise ValueError(f"expected state/action shape (16,), got {state_shape}/{action_shape}")

    report = {
        "repo_id": args.repo_id,
        "root": str(args.root.resolve()),
        "num_episodes": dataset.num_episodes,
        "num_frames": dataset.num_frames,
        "fps": dataset.fps,
        "camera_keys": sorted(camera_keys),
        "state_shape": state_shape,
        "action_shape": action_shape,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
