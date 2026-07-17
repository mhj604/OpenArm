# Safety Boundary

## Processes that cannot move the robot

- `calibrate_workspace.sh` reads the chest RealSense only.
- `record_demonstrations.sh` reads two RealSense cameras and localhost UDP.
- The reused CAN telemetry helper opens receive-only SocketCAN sockets.
- `convert_dataset.sh`, `validate_dataset.sh`, and `train_simple_dp3.sh` are
  offline processes.

The collector does not import `openarm_can`, call `enable_all`, or transmit a
CAN frame. Motor movement during demonstrations comes only from the user's
already-established ExoArm/OpenArm bridge.

## Not included yet

No DP3 checkpoint is connected to the motors. Active rollout will be a separate
stage after dataset inspection and training. It must retain joint limits,
per-step limits, stale-command shutdown, current-pose hold, explicit enable,
Ctrl+C stop, and disable-on-exit.
