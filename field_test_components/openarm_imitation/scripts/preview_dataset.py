#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a camera/point-cloud preview without opening a GUI")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("/home/mh/openarm_imitation/outputs/dataset_preview.png"))
    args = parser.parse_args()
    root = zarr.open_group(str(args.dataset), mode="r")
    data = root["data"]
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if args.episode < 0 or args.episode >= len(ends):
        raise SystemExit(f"episode must be in [0, {len(ends) - 1}]")
    start = 0 if args.episode == 0 else int(ends[args.episode - 1])
    end = int(ends[args.episode])
    idx = (start + end) // 2
    pcd = np.asarray(data["point_cloud"][idx])
    fig = plt.figure(figsize=(14, 8))
    columns = 3 if "img" in data else 2
    ax = fig.add_subplot(2, columns, 1, projection="3d")
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=np.clip(pcd[:, 3:6], 0, 1), s=2)
    ax.set_title(f"point cloud ep={args.episode} frame={idx}")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax2 = fig.add_subplot(2, columns, columns + 1, projection="3d")
    ax2.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=np.clip(pcd[:, 3:6], 0, 1), s=2)
    ax2.view_init(elev=70, azim=-90)
    ax2.set_title("top view")
    if "img" in data:
        ax3 = fig.add_subplot(2, columns, 2)
        ax3.imshow(data["img"][idx]); ax3.set_title("D435iCB chest"); ax3.axis("off")
    if "img_hand" in data:
        ax4 = fig.add_subplot(2, columns, 3)
        ax4.imshow(data["img_hand"][idx]); ax4.set_title("D405 hand"); ax4.axis("off")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    print(args.output)


if __name__ == "__main__":
    main()

