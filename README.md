# OpenArm-v1 Bimanual Simple DP3

Isolated DP3 pipeline for the following physical task:

- The cube starts at a randomized position inside the center red rectangle.
- OpenArm-v1 picks it and places it centered on the red point on the left.
- ExoArm-7 and the existing bridge provide demonstrations.
- Chest camera: RealSense D435ICB (`261722072049`).
- Left-wrist camera: RealSense D405 (`260322271498`).
- There is no right-wrist camera.

The existing `/home/mh/openarm_act`, ROS2 workspaces, ExoArm bridge, CAN
configuration, and calibration files are not modified. The collector imports
their receive-only telemetry adapter and never opens a transmit CAN socket.

## First baseline

Simple DP3 is trained from:

- `point_cloud`: 1,024 XYZRGB points in the fixed, wide D435 depth/IR frame.
- `agent_pos`: 16 OpenArm joint/gripper positions in radians.
- `action`: 16 outgoing CAN target positions in radians.

Both cameras' RGB-D and raw sampled point clouds are saved. D405 data is kept
for later multi-view or calibrated wrist-camera work, but it is not fused into
the first model because the wrist camera moves and no hand-eye transform has
been measured yet.

The chest D435 uses RGB 848x480 at 30 fps. RGB is aligned into the 848x480
depth coordinate system so DP3 retains the wider stereo depth/IR field of view
instead of cropping geometry to the narrower 640x480 RGB field of view.

## Setup

```bash
/home/mh/openarm_dp3/scripts/setup_python_overlay.sh
```

Dependencies are installed under `/home/mh/openarm_dp3/.python-packages`.
Nothing is installed into the ACT virtual environment or system Python.

## Data collection sequence

With the center red rectangle and left red target point visible, and the chest
camera fixed:

```bash
/home/mh/openarm_dp3/scripts/calibrate_workspace.sh
```

Then start the existing HMI, ExoArm ROS path, and known-good OpenArm bridge as
usual. The passive collector itself does not enable or move a motor:

```bash
/home/mh/openarm_dp3/scripts/record_demonstrations.sh 40
```

Press Enter to start only after the cube and robot are ready. During recording,
press Enter immediately after releasing the cube. Mark failed or unsafe
attempts as rejected; rejected episodes remain preserved but are excluded.

## Convert and train

```bash
/home/mh/openarm_dp3/scripts/convert_dataset.sh
/home/mh/openarm_dp3/scripts/validate_dataset.sh
/home/mh/openarm_dp3/scripts/train_simple_dp3.sh
```

Training is offline and never sends robot commands. Autonomous hardware rollout
is intentionally not part of this stage.

See `docs/RUNBOOK.md`, `docs/TASK_PROTOCOL.md`, and `docs/SAFETY.md`.
