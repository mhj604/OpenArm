# Runbook

## 1. Install isolated overlay

```bash
/home/mh/openarm_dp3/scripts/setup_python_overlay.sh
```

## 2. Calibrate the fixed workspace crop

Place the center red rectangle and left red target point within the chest view.
Keep the chest camera fixed after this step.

```bash
/home/mh/openarm_dp3/scripts/calibrate_workspace.sh
```

The script detects one red start rectangle and one compact red target point to
its left from the D435 image. It does not write guessed bounds. Review the saved
PNG under `calibration/`; the cyan outer rectangle must contain the start
region, target point, and their approach space.

The chest D435 profile is RGB/depth 848x480 at 30 fps. Color is aligned to the
wider depth coordinate system; depth is not cropped to the narrower 640x480
color field of view.

## 3. Collect demonstrations

Start the existing Qnbot HMI, ExoArm ROS path, and known-good bridge using the
same commands that already work. Then run:

```bash
/home/mh/openarm_dp3/scripts/record_demonstrations.sh 50
```

The wrapper starts only the existing receive-only telemetry mirror if needed.
For each episode:

1. Randomize cube position/yaw inside the center rectangle.
2. Restore the known start pose with ExoArm teleoperation.
3. Press Enter and begin moving at the cue.
4. Center the cube on the left red point, place it, and release.
5. Press Enter immediately.
6. Accept only a clean success; reject failures.

The target is 50 accepted demonstrations. Rejected or too-short attempts are
preserved for traceability and do not reduce the remaining accepted count.

## 4. Convert without altering raw data

```bash
/home/mh/openarm_dp3/scripts/convert_dataset.sh
```

The command refuses to overwrite an existing processed dataset. To intentionally
replace it while preserving the previous one under a timestamped backup:

```bash
/home/mh/openarm_dp3/scripts/convert_dataset.sh --overwrite
```

## 5. Validate and train

```bash
/home/mh/openarm_dp3/scripts/validate_dataset.sh
/home/mh/openarm_dp3/scripts/train_simple_dp3.sh
```

The RTX 2050 configuration uses batch size 4, gradient accumulation 4, a
reduced UNet, 1,024 points, 16-step horizon, and 8 action steps. W&B logging is
offline. Checkpoints are stored only under `outputs/`.

### Zero-start retraining without changing demonstrations

The accepted raw episodes and the original processed Zarr remain unchanged.
Create a second dataset that prepends a 30-frame linear arm interpolation from
numeric zero to each demonstration's first state/action:

```bash
/home/mh/openarm_dp3/scripts/augment_zero_start_dataset.sh
/home/mh/openarm_dp3/scripts/validate_dataset.sh \
  /home/mh/openarm_dp3/data/processed/openarm_cube_to_point_zero_start.zarr
/home/mh/openarm_dp3/scripts/train_simple_dp3_zero_start.sh
```

The prefix reuses the episode's first chest point cloud, keeps both gripper
channels at their demonstrated open values, and changes only the new derived
dataset. Training writes to a separate `outputs/zero_start_retrain_*` directory.

If PyTorch reports CUDA unavailable after reboot, recreate the existing NVIDIA
device nodes and rerun training:

```bash
/sbin/ub-device-create --verbose
```

## 6. First active DP3 rollout

Stop the ExoArm-to-OpenArm bridge and put the yellow cube inside the center red
rectangle. Bring up the existing CAN interfaces without enabling a motor:

```bash
sudo /home/mh/openarm_act/scripts/bring_up_can_receive_only.sh
```

Then start the bounded rollout:

```bash
cd /home/mh/openarm_dp3
OPENARM_DP3_EXECUTE=YES ./scripts/rollout_dp3.sh
```

The checkpoint and D435 are loaded and a discarded learned inference runs while
CAN remains closed. The D435 must see the yellow cube in the learned center-box
region before the motor handshake is allowed. Type `START` only when the
workspace is clear. D405 is not opened because the first trained model uses only
the fixed D435 point cloud.

## 7. Learned continuous cycles

The active controller uses one explicit OpenArm zero pose for every cycle. Its
seven arm targets on each side are `0.0 rad`, matching `open_home_left` and
`open_home_right` in the existing ExoArm bridge. Grippers are opened at the
recorded safe action bound. It does not select a recorded robot pose and does
not read or replay a demonstration action sequence.

Preview D435 input and one DP3-generated action chunk without opening CAN:

```bash
/home/mh/openarm_dp3/scripts/preview_learned_dp3.sh
```

The output must report `cube_in_start_region: true` and
`recorded_action_replay: false`. For active learned control, use the command in
section 6.

Continuous mode is enabled in `configs/dp3_rollout.env`. After DP3 releases the
cube at a learned goal location for eight stable frames, both arms return to
zero pose. The controller stays alive and sends fresh zero-pose hold commands.
Move the cube from the target back inside the center rectangle. When the D435
sees it in the start region for 15 consecutive control frames, a new learned
DP3 cycle begins. Press Ctrl+C to stop the session and disable both arms.

The 30-second value is a per-cycle failure timeout, not the lifetime of the
continuous process. If a cycle does not produce a verified placement within
that time, the controller stops instead of retrying uncontrolled motion.

At the start of each cycle, zero-pose motion is ramped at no more than 0.03 rad
per arm step. The policy then receives two consecutive zero-pose point clouds.
Each plan uses one deterministic learned diffusion sample and executes the
complete 15-action future suffix. The final operator-tuned arm gain is
0.87890625. The first 0.4 seconds use 0.90 blending and a 0.029296875 rad step
ceiling; later steps use no low-pass attenuation and a 0.05859375 rad policy and
motor ceiling. These transforms never substitute a recorded action sequence.

## 8. Raw demonstration replay diagnostic

Replay is a separate active diagnostic and never runs DP3 inference or opens a
camera. It accepts only a complete, operator-accepted raw episode whose action
metadata is `outgoing OpenArm CAN target position`. Both arms first ramp to the
recorded first state, then the raw action frames are sent at the recorded rate.

```bash
cd /home/mh/openarm_dp3
OPENARM_DP3_REPLAY=YES DP3_REPLAY_EPISODE=1 ./scripts/replay_demonstration.sh
```

Type `REPLAY` only with the ExoArm bridge stopped, both-arm path clear, and the
physical E-stop in hand. Episode 1 lasts about 18.5 seconds at 15 Hz. The motor
endpoint still clamps each target to recorded bounds. Its replay-only arm and
gripper rate limits are computed from the selected episode's largest raw target
transition, so learned-policy tuning does not slow or reshape the recording.

## Stop conditions

- Collector: Ctrl+C preserves the current episode as incomplete and disconnects
  both cameras. It never controls motors.
- Training: Ctrl+C stops optimization; the latest completed checkpoint remains.
- Active policy: Ctrl+C requests stop and the motor process disables both arms.
  Use physical power/E-stop if software stop is not immediate.
- Active replay: Ctrl+C or normal completion sends the stop packet and disables
  both arms.
- Continuous learned policy: after each verified placement it returns to zero
  and remains alive while waiting. Ctrl+C stops inference, zero-pose motion, or
  waiting; sends the stop packet; and disables both arms.
