from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def validate(path: Path) -> dict[str, object]:
    root = zarr.open_group(str(path), mode="r")
    required = {"state", "action", "point_cloud"}
    if "data" not in root or "meta" not in root or set(root["data"].keys()) != required:
        raise ValueError("DP3 zarr must contain only data/{state,action,point_cloud} and meta")
    state = root["data/state"]
    action = root["data/action"]
    points = root["data/point_cloud"]
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 16:
        raise ValueError("state and action must both be shaped (T, 16)")
    if points.shape != (state.shape[0], 1024, 6):
        raise ValueError("point_cloud must be shaped (T, 1024, 6)")
    if len(ends) == 0 or np.any(np.diff(ends) <= 0) or ends[-1] != state.shape[0]:
        raise ValueError("episode_ends is invalid")
    for name, array in (("state", state), ("action", action), ("point_cloud", points)):
        for start in range(0, len(array), 256):
            if not np.isfinite(array[start : start + 256]).all():
                raise ValueError(f"{name} contains NaN or Inf")
    return {
        "status": "PASS",
        "path": str(path),
        "episodes": int(len(ends)),
        "frames": int(state.shape[0]),
        "state_shape": list(state.shape),
        "action_shape": list(action.shape),
        "point_cloud_shape": list(points.shape),
        "state_range": [float(state[:].min()), float(state[:].max())],
        "action_range": [float(action[:].min()), float(action[:].max())],
        "point_cloud_range_m": [float(points[:].min()), float(points[:].max())],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("/home/mh/openarm_dp3/data/processed/openarm_cube_to_point.zarr"),
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.path), indent=2))


if __name__ == "__main__":
    main()
