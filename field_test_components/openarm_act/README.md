# OpenArm-v1 Bimanual + ExoArm-7 ACT Workspace

This isolated workspace adds imitation-learning data capture and ACT training on top of the user's already-working control chain:

`ExoArm-7 → existing ROS2/WebSocket path → existing exo_to_openarm bridge → OpenArm-v1 bimanual`

The new code does not replace or reimplement that controller. It passively observes the exact outgoing OpenArm CAN position targets, the corresponding motor position responses, and two RGB cameras:

- `chest`: Intel RealSense D435ICB
- `wrist_left`: Intel RealSense D405
- no right-wrist camera

## Layout

- `vendor/lerobot`: official LeRobot v0.6.0 source pinned at `30da8e687a6dfc617fcd94afc367ac7071c376ce`
- `plugins/lerobot_robot_openarm_can_passive`: dataset-only LeRobot robot/action-tap plugin
- `ros2_adapter/openarm_can_telemetry_udp.py`: receive-only SocketCAN mirror
- `configs`: camera, CAN, existing mapping reference, and environment templates
- `scripts`: telemetry, record, dataset inspection, and ACT train/resume entry points
- `docs`: audit, mapping, safety, runbook, changes, rollback, and limitations

## Key behavior

The passive CAN mirror opens RAW SocketCAN receive sockets and never transmits a CAN frame. The LeRobot robot's `send_action()` is a no-op: the existing bridge has already sent the captured command. This keeps the known-good communication and follow-control path unchanged.

State and action are both recorded as 16 radians:

- left joints 1–7, left gripper
- right joints 1–7, right gripper

## Environment

```bash
source /home/mh/openarm_act/scripts/env.sh
```

LeRobot uses project-local Python 3.12. The existing ROS2 Humble/Python 3.10 setup remains unchanged.

## Prepared sequence

```bash
# After a reboot/reconnect, restore the already-known CAN link configuration.
# This sends no application CAN frame and requires sudo.
sudo /home/mh/openarm_act/scripts/bring_up_can_receive_only.sh

# Optional camera-only check (no CAN/ROS/robot import).
source /home/mh/openarm_act/scripts/env.sh
python /home/mh/openarm_act/scripts/test_cameras.py --duration-s 10

# Terminal 1: receive-only CAN telemetry
/home/mh/openarm_act/scripts/start_can_telemetry.sh

# Start the already-working ExoArm/OpenArm control using the user's existing command.

# Terminal 2: passive LeRobot recording
/home/mh/openarm_act/scripts/record_one_episode.sh
```

Camera-only operation, receive-only CAN startup, configuration parsing, and
software safety tests have passed. Starting the existing bridge and recording
a physical episode still require explicit motion approval. Autonomous ACT
Active ACT rollout is provided as an explicit, separately launched path that
reuses the existing bridge's Classic-CAN motor configuration.
