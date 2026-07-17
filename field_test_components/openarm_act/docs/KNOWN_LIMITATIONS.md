# Known Limitations

- Both required RGB cameras pass simultaneous 640x480@30 capture. D435I USB 2
  streaming remains unreliable, so the chest camera must stay on its working
  SuperSpeed cable/port for collection.
- Identical PEAK adapters may exchange `can0`/`can1` names after reconnect.
- The passive decoder is prepared for the existing bridge's Classic-CAN MIT frames: command IDs 1–8, response IDs 17–24, and ±12.5 radian position encoding.
- With the active bridge intentionally stopped, no command/state frames were
  present during the receive-only smoke test. Live state/action freshness and
  exact synchronization therefore remain to be measured during the approved
  one-episode collection.
- The ExoArm WebSocket/ROS publisher starts correctly, but no ExoArm client was
  connected during the bounded smoke test. Live message length, rate, and
  timestamp behavior still need a passive measurement before physical
  collection.
- CAN telemetry reflects commands and responses visible on the bus; it does not include ExoArm buttons, recovery labels, task success, or force/torque sensors.
- LeRobot camera throughput passed, but on-disk video encoding and ACT memory
  use were not tested. Batch size 1 is only a conservative RTX 2050 starting
  point.
- Initial ACT is RGB-only; depth, intrinsics capture, and camera extrinsics are future work.
- Autonomous policy rollout is wired through a separate localhost policy/motor
  pair. The motor process reuses the known-good bridge's OpenArm CAN setup and
  refuses stale or non-finite actions, but physical success has not yet been
  established for this 10-demonstration model.
