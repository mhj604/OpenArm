from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

from .config import CollectionSettings
from .pointcloud import voxel_sample_fixed


class ConversionError(RuntimeError):
    """Raised when accepted raw demonstrations cannot form a DP3 dataset."""


@dataclass(frozen=True)
class EpisodeSelection:
    path: Path
    episode_index: int
    raw_frames: int
    start_frame: int
    end_frame: int

    @property
    def selected_frames(self) -> int:
        return self.end_frame - self.start_frame


def find_accepted_episodes(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in sorted(root.glob("episode_*.zarr")):
        group = zarr.open_group(str(path), mode="r")
        if group.attrs.get("complete") is True and group.attrs.get("accepted") is True:
            result.append(path)
    return result


def trim_bounds(action: np.ndarray, settings: CollectionSettings) -> tuple[int, int]:
    values = np.asarray(action, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 16 or len(values) < 2:
        raise ConversionError(f"action array must be shaped (T, 16), got {values.shape}")
    if not np.isfinite(values).all():
        raise ConversionError("action contains NaN or Inf")
    if not settings.trim.enabled:
        return 0, len(values)

    distance_from_start = np.max(np.abs(values - values[0]), axis=1)
    moving_from_start = np.flatnonzero(
        distance_from_start >= settings.trim.motion_threshold_radian
    )
    if len(moving_from_start) == 0:
        raise ConversionError("episode contains no action motion above the configured threshold")
    start = max(0, int(moving_from_start[0]) - settings.trim.pre_roll_frames)

    per_step = np.max(np.abs(np.diff(values, axis=0)), axis=1)
    moving_steps = np.flatnonzero(per_step >= settings.trim.step_threshold_radian)
    if len(moving_steps) == 0:
        raise ConversionError("episode contains no per-step action motion")
    last_changed_frame = int(moving_steps[-1]) + 1
    end = min(len(values), last_changed_frame + 1 + settings.trim.post_roll_frames)
    if end - start < settings.sample_fps * 2:
        raise ConversionError("trimmed episode is shorter than two seconds")
    return start, end


def select_episodes(
    settings: CollectionSettings,
) -> tuple[list[EpisodeSelection], list[dict[str, str]]]:
    selected: list[EpisodeSelection] = []
    rejected: list[dict[str, str]] = []
    for path in find_accepted_episodes(settings.dataset_root):
        group = zarr.open_group(str(path), mode="r")
        action = group["data/action"][:]
        try:
            start, end = trim_bounds(action, settings)
        except ConversionError as exc:
            rejected.append({"path": str(path), "reason": str(exc)})
            continue
        selected.append(
            EpisodeSelection(
                path=path,
                episode_index=int(group.attrs["episode_index"]),
                raw_frames=len(action),
                start_frame=start,
                end_frame=end,
            )
        )
    return selected, rejected


def prepare_output_path(path: Path, *, overwrite: bool) -> tuple[Path, Path | None]:
    backup: Path | None = None
    if path.exists():
        if not overwrite:
            raise ConversionError(
                f"output exists; pass --overwrite to preserve and replace it: {path}"
            )
        backup = path.with_name(f"{path.name}.backup_{time.strftime('%Y%m%d_%H%M%S')}")
        os.replace(path, backup)
    temporary = path.with_name(f".{path.name}.incomplete_{os.getpid()}")
    if temporary.exists():
        raise ConversionError(f"temporary output already exists: {temporary}")
    return temporary, backup


def convert(settings: CollectionSettings, output: Path, *, overwrite: bool) -> dict[str, object]:
    episodes, rejected = select_episodes(settings)
    if not episodes:
        raise ConversionError("no accepted, moving demonstrations are available")
    total_frames = sum(episode.selected_frames for episode in episodes)
    point_channels = 6 if settings.use_point_color else 3
    temporary, backup = prepare_output_path(output, overwrite=overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(temporary), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    point_cloud = data.create_dataset(
        "point_cloud",
        shape=(total_frames, settings.training_points, point_channels),
        chunks=(64, settings.training_points, point_channels),
        dtype="f4",
        compressor=compressor,
    )
    state = data.create_dataset(
        "state",
        shape=(total_frames, 16),
        chunks=(256, 16),
        dtype="f4",
        compressor=compressor,
    )
    action = data.create_dataset(
        "action",
        shape=(total_frames, 16),
        chunks=(256, 16),
        dtype="f4",
        compressor=compressor,
    )
    episode_ends: list[int] = []
    manifest_episodes: list[dict[str, object]] = []
    cursor = 0
    for episode in episodes:
        source = zarr.open_group(str(episode.path), mode="r")
        source_data = source["data"]
        length = episode.selected_frames
        destination = slice(cursor, cursor + length)
        source_slice = slice(episode.start_frame, episode.end_frame)
        state[destination] = source_data["state"][source_slice]
        action[destination] = source_data["action"][source_slice]
        for local_index, raw_index in enumerate(range(episode.start_frame, episode.end_frame)):
            point_cloud[cursor + local_index] = voxel_sample_fixed(
                source_data["chest_point_cloud_raw"][raw_index],
                point_count=settings.training_points,
                voxel_size_m=settings.voxel_size_m,
                seed=episode.episode_index * 1_000_000 + raw_index,
            )
        cursor += length
        episode_ends.append(cursor)
        manifest_episodes.append(
            {
                "episode_index": episode.episode_index,
                "source": str(episode.path),
                "raw_frames": episode.raw_frames,
                "start_frame": episode.start_frame,
                "end_frame": episode.end_frame,
                "selected_frames": length,
            }
        )
        print(
            f"converted episode {episode.episode_index}: "
            f"{episode.raw_frames} -> {length} frames",
            flush=True,
        )
    meta.create_dataset(
        "episode_ends",
        data=np.asarray(episode_ends, dtype=np.int64),
        chunks=(max(1, min(128, len(episode_ends))),),
        dtype="i8",
        compressor=compressor,
    )
    root.attrs.update(
        {
            "schema_version": 1,
            "task_id": settings.task_id,
            "point_cloud_source": "chest",
            "point_cloud_coordinate_frame": "d435_depth_ir",
            "point_cloud_channels": (
                "xyz_meters_rgb_unit" if settings.use_point_color else "xyz_meters"
            ),
            "state_dim": 16,
            "action_dim": 16,
            "state_action_unit": "radian",
            "episode_count": len(episodes),
            "frame_count": total_frames,
            "sample_fps": settings.sample_fps,
        }
    )
    if not np.isfinite(state[:]).all() or not np.isfinite(action[:]).all():
        raise ConversionError("converted state/action contains NaN or Inf")
    if not np.isfinite(point_cloud[:]).all():
        raise ConversionError("converted point cloud contains NaN or Inf")
    os.replace(temporary, output)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "output": str(output),
        "backup": str(backup) if backup else None,
        "episode_count": len(episodes),
        "frame_count": total_frames,
        "episodes": manifest_episodes,
        "rejected_during_conversion": rejected,
    }
    manifest_path = output.with_name(f"{output.name}.manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        type=Path,
        default=Path("/home/mh/openarm_dp3/configs/collection.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/mh/openarm_dp3/data/processed/openarm_cube_to_point.zarr"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = CollectionSettings.load(args.collection)
    manifest = convert(settings, args.output, overwrite=args.overwrite)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
