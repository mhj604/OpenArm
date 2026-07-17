# System Audit and Setup Basis

Audit date: 2026-07-16 (Asia/Seoul)

The facts below combine host inspection, static source inspection, camera-only
tests, and a receive-only CAN smoke test. No bridge or motor-control process was
started during the final camera/CAN validation.

## A. Discovered environment

- OS: Ubuntu 22.04.5 LTS (Jammy)
- kernel: `6.8.0-124-generic`
- GPU: NVIDIA GeForce RTX 2050, 4 GiB VRAM
- NVIDIA driver: `590.48.01`; driver-reported CUDA compatibility: 13.1
- CUDA toolkit compiler: `nvcc` absent; CUDA 12 runtime libraries are present
- system Python: 3.10.12; pip 22.0.2
- conda/Miniforge/mamba: not installed
- ROS2: Humble
- LeRobot before this setup: not installed and no prior LeRobot repository found
- LeRobot selected for this isolated setup: v0.6.0, Python >=3.12, commit `30da8e687a6dfc617fcd94afc367ac7071c376ce`
- OpenArm packages: `openarm-can`/`openarm-can-utils`/`python3-openarm-can` 1.2.9
- RealSense ROS source overlay: `realsense-ros` 4.57.0; `ros-humble-librealsense2` 2.57.7
- isolated RealSense runtime: librealsense/pyrealsense2 v2.56.5 RSUSB bindings
  under `/home/mh/openarm_act/python`; the system ROS installation is unchanged
- ffmpeg executable: absent; LeRobot v0.6.0 dataset extras use Python video dependencies in the isolated environment
- CAN utilities: installed
- connected CAN adapters: two PEAK PCAN-USB FD adapters
- current interfaces: `can0` and `can1`, both UP as Classic CAN at 1,000,000 bit/s, error-active
- RealSense devices: chest D435I serial `261722072049` and left-wrist D405
  serial `260322271498`, both enumerated at USB 3.20 / 5000 Mbit/s
- ExoArm serial/HMI device at audit time: not connected (`/dev/ttyACM*` and `/dev/ttyUSB*` absent)
- live ROS graph at audit time: only `/parameter_events` and `/rosout`; no ExoArm publisher was running

Related repositories/workspaces:

- `/home/mh/openarm_can`: official repository, main at `4f5d0c107155c85b0775492d62c74bda80c94dd7`, dirty (`examples/demo.cpp`)
- `/home/mh/openarm_teleop`: official repository, main at `eb2d49338bf70ace95282ea724903849397b7811`, contains user modifications and untracked Exo bridge files
- `/home/mh/openarm_ws/src/openarm_description`: official repository, clean at `1daa95b71565a6696242013a42c4d3af689cbdb3`
- `/home/mh/exo_openarm_ws`: existing ROS2/C++ experiments, not selected as canonical
- `/home/mh/foxtech_exoarm/qnbot-exo-std-demo`: existing ExoArm ROS2 package
- `/home/mh/ros2_ws/src/realsense-ros`: Intel RealSense ROS overlay
- `/home/mh/openarm_imitation`: earlier isolated DP3-oriented work remains untouched and is not used by this ACT setup

Existing backups found:

- `/home/mh/openarm_backup_20260702.zip`
- `/home/mh/openarm_backup_20260716.zip`
- `/home/mh/Downloads/OpenArm.zip`

## B. Existing operational path

Static evidence identifies this path:

1. ExoArm HMI/WebSocket client connects to `ws://127.0.0.1:19091`.
2. `qnbot_teleoperator.websocket_teleoperator` publishes `/exo/joint_command` as `sensor_msgs/msg/JointState`.
3. Message positions are ordered as left joints 0–6, right joints 7–13, left trigger 14, right trigger 15; the package documentation identifies arm units as radians and triggers as normalized values.
4. `/home/mh/openarm_teleop/exo_openarm_bridge/exo_to_openarm.py` maps the message to both arms.
5. Current bridge values are right arm `can0`, left arm `can1`, Classic CAN, 1 Mbit/s.
6. The bridge uses motor send IDs 1–8 and receive IDs 17–24, MIT position control, and enables both arms when constructed.

The mapping reused in the isolated config is taken from the current bridge, not the older PDF example:

- common joint scale: 1.8
- left source order: `[0, 1, 2, 3, 4, 6, 5]`
- left signs: `[1, 1, -1, 1, 1, -1, 1]`
- right source order: `[7, 8, 9, 10, 11, 13, 12]`
- right signs: `[1, 1, -1, 1, 1, -1, -1]`
- first message captures the ExoArm home
- grippers: source indices 14/15, input clamped to `[0,1]`, multiplied by
  `-0.70` radians by default; `OPENARM_GRIPPER_SCALE_RAD` can override the
  magnitude up to the documented `0.7854` radian limit
- current bridge joint gains: `kp=[20,20,12,12,5,5,4]`, `kd=[1,1,.8,.8,.3,.3,.3]`; gripper `kp=2`, `kd=.1`

## C. Unresolved items

- physical can0/can1-to-left/right assignment after adapter reconnect
- live ExoArm topic rate, exact data length, trigger behavior, timestamp behavior, and switch/deadman behavior
- physical joint limits and a safe maximum per-step target for this specific robot
- existing motor zero/calibration state as seen through LeRobot
- safe initial pose, mechanical support, and emergency-stop method at the workcell
- camera intrinsics, depth scale, ROS frame names, and extrinsics
- ACT batch size that fits the RTX 2050 for the eventual dataset

These are deliberately required as explicit runtime values or documented as pending; no serial, limit, or physical assignment was invented.

## D. Implemented approach

- isolated `/home/mh/openarm_act`; no global pip/conda changes
- managed Python 3.12 for LeRobot v0.6.0
- the user's existing ROS2/ExoArm/OpenArm bridge remains the sole active controller
- receive-only SocketCAN observer decodes the exact outgoing MIT target and returned motor position for all 16 motors
- localhost UDP packets include timestamps, sequence, units, target/state vectors, and per-motor sample timestamps
- external `openarm_can_passive` LeRobot robot opens only UDP and cameras; its `send_action()` is a no-op
- external `openarm_can_action_tap` teleoperator supplies the already-sent CAN target to the dataset
- state and action are both recorded in radians, matching the existing bridge/OpenArm CAN convention
- exactly two RGB observation keys: `chest` and `wrist_left`
- LeRobot record, local dataset inspection, ACT train/resume scripts; active rollout deliberately disabled

## E. Work allowed without motion approval

- edit configs and documentation
- start the receive-only CAN telemetry mirror
- inspect camera identity, profiles, and camera-only streams
- inspect CAN link state and passively monitor frames
- train ACT from an already-recorded local dataset

## F. Work requiring explicit approval

- starting or operating the existing bridge that moves OpenArm
- any OpenArm motion, calibration, zero write, motor configuration, CAN mode/bitrate change, or firmware change
- passive recording while the active bridge is running, including one-episode
  and five-episode physical smoke collection
- replay or autonomous policy rollout
- connecting a trained policy to the existing active control path

No bridge start, physical collection, or autonomous rollout in section F was
executed during this validation. Calibration was performed earlier only after
the user's explicit request; both arms were disabled afterward. The 2026-07-16
CAN link bring-up was transient, matched the recorded existing 1 Mbit/s Classic
CAN setting, and sent zero application frames.
