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
/home/mh/openarm_dp3/scripts/record_demonstrations.sh 40
```

The wrapper starts only the existing receive-only telemetry mirror if needed.
For each episode:

1. Randomize cube position/yaw inside the center rectangle.
2. Restore the known start pose with ExoArm teleoperation.
3. Press Enter and begin moving at the cue.
4. Center the cube on the left red point, place it, and release.
5. Press Enter immediately.
6. Accept only a clean success; reject failures.

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

If PyTorch reports CUDA unavailable after reboot, recreate the existing NVIDIA
device nodes and rerun training:

```bash
/sbin/ub-device-create --verbose
```

## Stop conditions

- Collector: Ctrl+C preserves the current episode as incomplete and disconnects
  both cameras. It never controls motors.
- Training: Ctrl+C stops optimization; the latest completed checkpoint remains.
- Active policy rollout is not started or enabled by any command in this runbook.
