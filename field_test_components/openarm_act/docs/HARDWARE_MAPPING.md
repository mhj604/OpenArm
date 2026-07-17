# Hardware and Dataset Mapping

## Existing controller

The existing file `/home/mh/openarm_teleop/exo_openarm_bridge/exo_to_openarm.py` remains authoritative for ExoArm home capture, reorder, signs, joint scale, gripper scale, gains, and left/right CAN assignment. Its mapping is transcribed for documentation in `/home/mh/openarm_act/configs/existing_bridge_mapping.yaml`; the recorder does not recompute it.

## Why CAN targets are recorded

The passive observer decodes the target position already placed in each outgoing MIT CAN frame. This avoids a second home capture and guarantees that the dataset action matches the command the existing bridge actually sent.

It also decodes each motor response frame for the actual follower state. Both values use the existing OpenArm CAN library's native radian convention.

| CAN frame | IDs | Dataset role | Position bytes |
|---|---:|---|---|
| outgoing MIT target | 1–8 | action | data 0–1 |
| motor state response | 17–24 | observation state | data 1–2 |

Each side is ordered as joints 1–7 followed by gripper. The combined LeRobot feature order is left eight followed by right eight, yielding a 16-value state and a 16-value action.

## Cameras

| Key | Model | Mount | Initial ACT stream |
|---|---|---|---|
| `chest` | Intel RealSense D435ICB | chest/workspace overview | RGB 640×480 @ 30 fps |
| `wrist_left` | Intel RealSense D405 | left end effector | RGB 640×480 @ 30 fps |

There is no right-wrist camera. Initial ACT does not use depth.

## CAN interfaces

The current bridge uses left `can1`, right `can0`, Classic CAN at 1 Mbit/s. The two PEAK adapters can enumerate differently after reconnect, so `/home/mh/openarm_act/hardware.env` keeps both values explicit. The passive observer never changes interface state, bitrate, or CAN mode.
