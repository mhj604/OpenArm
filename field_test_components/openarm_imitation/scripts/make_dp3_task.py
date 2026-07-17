#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
import zarr


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Hydra task config for converted bimanual data")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/home/mh/openarm_imitation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config/task"),
    )
    args = parser.parse_args()
    root = zarr.open_group(str(args.dataset), mode="r")
    state_dim = int(root["data"]["state"].shape[1])
    action_dim = int(root["data"]["action"].shape[1])
    point_count = int(root["data"]["point_cloud"].shape[1])
    cfg = {
        "name": args.name,
        "task_name": args.name,
        "shape_meta": {
            "obs": {
                "point_cloud": {"shape": [point_count, 3], "type": "point_cloud"},
                "agent_pos": {"shape": [state_dim], "type": "low_dim"},
            },
            "action": {"shape": [action_dim]},
        },
        "env_runner": None,
        "dataset": {
            "_target_": "openarm_imitation.dp3_dataset.OpenArmBimanualDataset",
            "zarr_path": str(args.dataset.resolve()),
            "horizon": "${horizon}",
            "pad_before": "${eval:'${n_obs_steps}-1'}",
            "pad_after": "${eval:'${n_action_steps}-1'}",
            "seed": 42,
            "val_ratio": 0.1,
            "max_train_episodes": None,
            "task_name": args.name,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.name}.yaml"
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing config: {out}")
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    print(out)


if __name__ == "__main__":
    main()

