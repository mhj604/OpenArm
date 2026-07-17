from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

VECTOR_SIZE = 16
ARM_INDICES = np.asarray([*range(7), *range(8, 15)], dtype=np.int64)
GRIPPER_INDICES = np.asarray([7, 15], dtype=np.int64)


class ZeroStartAugmentationError(RuntimeError):
    """Raised when a source dataset cannot be augmented without mutation."""


def build_zero_start_prefix(
    first_state: np.ndarray,
    first_action: np.ndarray,
    first_point_cloud: np.ndarray,
    *,
    prefix_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate arm state/action from numeric zero to an episode's first frame."""
    state = np.asarray(first_state, dtype=np.float32)
    action = np.asarray(first_action, dtype=np.float32)
    points = np.asarray(first_point_cloud, dtype=np.float32)
    if state.shape != (VECTOR_SIZE,) or action.shape != (VECTOR_SIZE,):
        raise ZeroStartAugmentationError("first state/action must each contain 16 values")
    if points.ndim != 2 or points.shape[1] != 6:
        raise ZeroStartAugmentationError("first point cloud must be shaped (N, 6)")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ZeroStartAugmentationError("first state/action must be finite")
    if not np.isfinite(points).all():
        raise ZeroStartAugmentationError("first point cloud must be finite")
    if prefix_frames < 2:
        raise ZeroStartAugmentationError("zero-start prefix must contain at least two frames")

    state_prefix = np.repeat(state[None, :], prefix_frames, axis=0)
    action_prefix = np.repeat(action[None, :], prefix_frames, axis=0)
    state_alpha = np.arange(prefix_frames, dtype=np.float32) / float(prefix_frames)
    action_alpha = np.arange(1, prefix_frames + 1, dtype=np.float32) / float(prefix_frames)
    state_prefix[:, ARM_INDICES] = state_alpha[:, None] * state[ARM_INDICES]
    action_prefix[:, ARM_INDICES] = action_alpha[:, None] * action[ARM_INDICES]
    state_prefix[:, GRIPPER_INDICES] = state[GRIPPER_INDICES]
    action_prefix[:, GRIPPER_INDICES] = action[GRIPPER_INDICES]
    point_prefix = np.repeat(points[None, :, :], prefix_frames, axis=0)
    return (
        np.ascontiguousarray(state_prefix, dtype=np.float32),
        np.ascontiguousarray(action_prefix, dtype=np.float32),
        np.ascontiguousarray(point_prefix, dtype=np.float32),
    )


def _validate_source(root: zarr.Group) -> tuple[object, object, object, np.ndarray]:
    try:
        state = root["data/state"]
        action = root["data/action"]
        points = root["data/point_cloud"]
        episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    except KeyError as exc:
        raise ZeroStartAugmentationError(f"source DP3 dataset is incomplete: {exc}") from exc
    if state.shape != action.shape or len(state.shape) != 2 or state.shape[1] != VECTOR_SIZE:
        raise ZeroStartAugmentationError("source state/action must be shaped (T, 16)")
    if points.shape != (state.shape[0], 1024, 6):
        raise ZeroStartAugmentationError("source point cloud must be shaped (T, 1024, 6)")
    if (
        episode_ends.ndim != 1
        or len(episode_ends) == 0
        or int(episode_ends[-1]) != state.shape[0]
        or np.any(np.diff(episode_ends) <= 0)
    ):
        raise ZeroStartAugmentationError("source episode_ends is invalid")
    return state, action, points, episode_ends


def augment_dataset(
    source_path: Path,
    output_path: Path,
    *,
    prefix_frames: int,
) -> dict[str, object]:
    if not source_path.is_dir():
        raise ZeroStartAugmentationError(f"source dataset does not exist: {source_path}")
    if output_path.exists():
        raise ZeroStartAugmentationError(f"refusing to overwrite output dataset: {output_path}")
    if source_path.resolve() == output_path.resolve():
        raise ZeroStartAugmentationError("source and output datasets must be different paths")

    source = zarr.open_group(str(source_path), mode="r")
    source_state, source_action, source_points, source_ends = _validate_source(source)
    episode_count = len(source_ends)
    total_frames = int(source_state.shape[0]) + episode_count * prefix_frames
    temporary = output_path.with_name(f".{output_path.name}.incomplete_{os.getpid()}")
    if temporary.exists():
        raise ZeroStartAugmentationError(f"temporary output already exists: {temporary}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(temporary), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    state = data.create_dataset(
        "state",
        shape=(total_frames, VECTOR_SIZE),
        chunks=(256, VECTOR_SIZE),
        dtype="f4",
        compressor=compressor,
    )
    action = data.create_dataset(
        "action",
        shape=(total_frames, VECTOR_SIZE),
        chunks=(256, VECTOR_SIZE),
        dtype="f4",
        compressor=compressor,
    )
    points = data.create_dataset(
        "point_cloud",
        shape=(total_frames, 1024, 6),
        chunks=(64, 1024, 6),
        dtype="f4",
        compressor=compressor,
    )

    source_starts = np.concatenate((np.asarray([0], dtype=np.int64), source_ends[:-1]))
    output_ends: list[int] = []
    manifest_episodes: list[dict[str, int]] = []
    cursor = 0
    for ordinal, (source_start, source_end) in enumerate(
        zip(source_starts, source_ends, strict=True)
    ):
        start = int(source_start)
        end = int(source_end)
        prefix_state, prefix_action, prefix_points = build_zero_start_prefix(
            np.asarray(source_state[start], dtype=np.float32),
            np.asarray(source_action[start], dtype=np.float32),
            np.asarray(source_points[start], dtype=np.float32),
            prefix_frames=prefix_frames,
        )
        prefix_slice = slice(cursor, cursor + prefix_frames)
        state[prefix_slice] = prefix_state
        action[prefix_slice] = prefix_action
        points[prefix_slice] = prefix_points
        cursor += prefix_frames

        source_length = end - start
        episode_slice = slice(cursor, cursor + source_length)
        state[episode_slice] = source_state[start:end]
        action[episode_slice] = source_action[start:end]
        points[episode_slice] = source_points[start:end]
        cursor += source_length
        output_ends.append(cursor)
        manifest_episodes.append(
            {
                "episode_ordinal": ordinal,
                "source_start": start,
                "source_end": end,
                "source_frames": source_length,
                "zero_start_prefix_frames": prefix_frames,
                "output_end": cursor,
            }
        )

    meta.create_dataset(
        "episode_ends",
        data=np.asarray(output_ends, dtype=np.int64),
        chunks=(max(1, min(128, episode_count)),),
        dtype="i8",
        compressor=compressor,
    )
    root.attrs.update(dict(source.attrs))
    root.attrs.update(
        {
            "source_dataset": str(source_path.resolve()),
            "source_dataset_mutated": False,
            "zero_start_augmentation": "linear_arm_state_action_prefix",
            "zero_start_prefix_frames": prefix_frames,
            "zero_start_prefix_seconds": prefix_frames
            / float(source.attrs.get("sample_fps", 15)),
            "frame_count": total_frames,
        }
    )
    if cursor != total_frames:
        raise ZeroStartAugmentationError(
            f"augmented frame count mismatch: wrote {cursor}, expected {total_frames}"
        )
    os.replace(temporary, output_path)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "source_mutated": False,
        "episode_count": episode_count,
        "source_frame_count": int(source_state.shape[0]),
        "zero_start_prefix_frames_per_episode": prefix_frames,
        "output_frame_count": total_frames,
        "episodes": manifest_episodes,
    }
    manifest_path = output_path.with_name(f"{output_path.name}.manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a separate DP3 dataset with zero-to-demonstration interpolation."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/mh/openarm_dp3/data/processed/openarm_cube_to_point.zarr"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/mh/openarm_dp3/data/processed/openarm_cube_to_point_zero_start.zarr"
        ),
    )
    parser.add_argument("--prefix-frames", type=int, default=30)
    args = parser.parse_args()
    print(
        json.dumps(
            augment_dataset(args.source, args.output, prefix_frames=args.prefix_frames),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
