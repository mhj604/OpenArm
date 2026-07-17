# Data Schema

## Raw episode

Each `data/raw/episodes/episode_NNNNNN.zarr` contains:

- `state`: `(T, 16)` radians, left 8 then right 8.
- `action`: `(T, 16)` outgoing CAN target radians, left 8 then right 8.
- `chest_point_cloud_raw`: `(T, 8192, 3)` meters in the wide D435 depth/IR frame.
- `wrist_left_point_cloud_raw`: `(T, 4096, 3)` meters in D405 frame.
- Aspect-preserving 320x240 RGB and depth previews from both cameras.
- Host, camera, telemetry sequence, age, and skew timestamps.
- Camera serials, intrinsics, depth scales, crop calibration, and task metadata.

Episodes are committed atomically. Rejected episodes have `accepted=false` and
remain on disk.

## Processed DP3 replay buffer

`data/processed/openarm_cube_to_point.zarr` contains:

- `data/point_cloud`: `(T, 1024, 3)` float32 XYZ meters in D435 depth/IR frame.
- `data/state`: `(T, 16)` float32 radians.
- `data/action`: `(T, 16)` float32 target radians.
- `meta/episode_ends`: cumulative frame indices.

The converter removes measured idle prefixes/suffixes without changing raw
episodes and records all trim decisions in a JSON manifest.
