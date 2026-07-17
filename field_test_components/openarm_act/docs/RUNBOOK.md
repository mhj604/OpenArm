# Runbook

Camera and receive-only paths were validated without moving the robot. Any step
that starts the existing bridge remains an explicitly approved physical-motion
operation.

## 1. Fill session values

```bash
cp /home/mh/openarm_act/configs/hardware.env.example /home/mh/openarm_act/hardware.env
```

Fill the numeric D435ICB and D405 serials, the same left/right CAN interfaces used by the working bridge, dataset ID, and task description.

## 2. Start passive CAN telemetry

After a reboot, first restore the documented transient SocketCAN link state in
a sudo terminal:

```bash
sudo /home/mh/openarm_act/scripts/bring_up_can_receive_only.sh
```

This configures Classic CAN at 1 Mbit/s and verifies that application TX packet
counters do not change.

Start this before beginning the demonstration so all 16 target/state channels are available:

```bash
/home/mh/openarm_act/scripts/start_can_telemetry.sh
```

This process only receives CAN frames and publishes localhost UDP telemetry. It has no CAN write call.

## 3. Start the existing control path

Use the exact command that already makes ExoArm-7 drive both OpenArm-v1 arms. Do not substitute a LeRobot follower command. The existing `/home/mh/openarm_teleop/exo_openarm_bridge/exo_to_openarm.py` remains the controller.

## 4. Record one passive episode

```bash
/home/mh/openarm_act/scripts/record_one_episode.sh
```

The recorder opens the two RealSense streams, receives passive state/action telemetry, and writes a local LeRobot dataset. Its robot `send_action()` does nothing, so no duplicate OpenArm command is sent.

Default recording is one 30-second episode, 30 fps, RGB video, local storage, and no Hub upload.

## 5. Inspect the dataset later if requested

```bash
source /home/mh/openarm_act/scripts/env.sh
python /home/mh/openarm_act/scripts/validate_dataset.py \
  --repo-id local/openarm_v1_bimanual_act_TIMESTAMP \
  --root /home/mh/openarm_act/data/recordings/SESSION
```

Expected schema: `chest`, `wrist_left`, 16-state radian vector, and 16-action radian vector.

## 6. Five-episode smoke collection

```bash
/home/mh/openarm_act/scripts/record_five_episodes.sh
```

Run only after the user chooses to validate the one-episode output.

## 7. ACT training

Training does not access hardware:

```bash
export LEROBOT_DATASET_REPO_ID=local/openarm_v1_bimanual_act_TIMESTAMP
export LEROBOT_DATASET_ROOT=/home/mh/openarm_act/data/recordings/SESSION
/home/mh/openarm_act/scripts/train_act.sh
```

Defaults: ACT, CUDA, batch size 1, 100,000 steps, checkpoint every 10,000 steps, WandB off. The conservative batch size reflects the 4 GiB RTX 2050 but was not runtime-tested.

Resume:

```bash
export ACT_TRAIN_CONFIG=/path/to/checkpoint/train_config.json
/home/mh/openarm_act/scripts/resume_act.sh
```

## 8. Rollout

The active ACT path uses the same Classic-CAN interfaces, motor IDs, radian
convention, and MIT gains as the working ExoArm bridge. It requires an explicit
execution variable because it enables and moves both arms:

```bash
OPENARM_ACT_EXECUTE=YES /home/mh/openarm_act/scripts/rollout_act.sh
```

The default rollout has no time limit and runs until `Ctrl-C`. Interrupting sends
a stop packet and the motor endpoint disables both arms. Never run the ExoArm
bridge at the same time.
