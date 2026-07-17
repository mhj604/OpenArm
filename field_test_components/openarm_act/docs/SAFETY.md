# Safety Boundary

The new imitation-learning setup is passive with respect to OpenArm:

- `openarm_can_telemetry_udp.py` creates only receive-only RAW CAN sockets;
- the LeRobot passive robot does not import a motor-control library;
- its `send_action()` validates and returns the captured value without transmitting it;
- no calibration, zero write, motor enable/disable, bitrate change, or firmware action exists in the recording path;
- the active rollout script refuses to start unless
  `OPENARM_ACT_EXECUTE=YES` is explicitly supplied.

The existing bridge still moves the physical robot. Its established emergency-stop, start pose, support, deadman/switch, and operating procedure remain in force whenever demonstrations are performed. Starting a recorder is not permission to start or move that controller.

If telemetry or recording fails:

1. Stop the physical motion using the existing workcell procedure.
2. Stop the existing bridge if necessary.
3. Press `Ctrl-C` in the recorder and passive telemetry terminals.
4. Leave CAN, zero, and calibration settings unchanged.

Any future ACT policy rollout is a new active-control step and still requires explicit approval, reduced-range limits, stale handling, current-pose checks, and a physical emergency stop.
