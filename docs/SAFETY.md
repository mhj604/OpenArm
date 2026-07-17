# Safety Boundary

## Processes that cannot move the robot

- `calibrate_workspace.sh` reads the chest RealSense only.
- `record_demonstrations.sh` reads two RealSense cameras and localhost UDP.
- The reused CAN telemetry helper opens receive-only SocketCAN sockets.
- `convert_dataset.sh`, `validate_dataset.sh`, and `train_simple_dp3.sh` are
  offline processes.
- `preview_learned_dp3.sh` reads only the D435, checkpoint, and processed visual
  references. It does not start a motor process or open CAN.

The collector does not import `openarm_can`, call `enable_all`, or transmit a
CAN frame. Motor movement during demonstrations comes only from the user's
already-established ExoArm/OpenArm bridge.

## Active DP3 rollout

`rollout_dp3.sh` is the learned-policy command that can move the robot. It reuses the
existing OpenArm motor endpoint and retains these independent gates:

- `OPENARM_DP3_EXECUTE=YES` is required before either process starts.
- The motor endpoint waits for the policy handshake before opening CAN.
- Enable acknowledgements are drained, and zero-gain queries must produce
  plausible state for both grippers before current-pose hold can begin.
- The policy loads the checkpoint, D435, workspace crop, and one warm-up
  inference before requesting the handshake.
- The operator must type `START` immediately before the handshake.
- Before learned actions, both arms move to the configured zero-radian pose used
  by the existing ExoArm bridge. Initial motion is rejected if any arm joint is
  more than 1.50 rad from zero; zero alignment uses a 0.03 rad ramp and must
  settle within 0.08 rad.
- Commands are clamped to the recorded action range. Arm joints are limited to
  0.05859375 rad per step at 15 Hz, below the demonstrated fast-joint 95th percentile
  of approximately 0.094 rad per step. Grippers retain their separate 0.08 rad
  limit.
- The left gripper remains at its recorded open bound for the first 2.5 seconds
  of each learned cycle. All accepted demonstrations kept it open for at least
  38 frames, so this blocks premature closure without replaying a trajectory.
- Learned arm output uses one deterministic diffusion sample and the complete
  15-action future suffix. Arm excursions use the last operator-tuned gain of
  0.87890625. The first 0.4 seconds use 0.90 blending and a 0.029296875 rad step
  ceiling; later actions use no attenuation, while the independent motor
  endpoint enforces 0.05859375 rad.
- Missing/stale policy commands stop the motor process. Ctrl+C and normal exit
  send a stop packet and disable both arms.

Keep the ExoArm bridge stopped, put the yellow cube inside the center red
rectangle, and keep a hand on the physical power/E-stop. The 30-second limit is
applied to each learned PICK→PLACE attempt. Continuous waiting after a verified
success does not expire.

## Recorded demonstration replay

`replay_demonstration.sh` is a separate active diagnostic guarded by
`OPENARM_DP3_REPLAY=YES` and an exact `REPLAY` terminal confirmation. It does not
load a model or camera. It accepts only complete, operator-accepted raw episodes
recorded as outgoing OpenArm position targets. Both arms visibly ramp to the
episode's first state before replay begins. The ExoArm bridge must be stopped.

The same motor endpoint enforces recorded action bounds, fresh command packets,
verified gripper feedback, and disable-on-exit behavior. Replay-only step limits
are derived from the selected raw episode so its targets are not reshaped by the
learned-policy profile. Normal completion and Ctrl+C both send a stop packet and
disable both arms.

## Continuous learned-policy behavior

Recorded point clouds identify the learned start and goal workspace regions, but
recorded actions are never opened by the learned-policy runtime reference loader.
All learned PICK→PLACE targets come from the loaded DP3 policy.

After the D435 sees the released yellow cube at a learned goal for eight stable
frames, OpenArm moves back to zero pose. It then remains enabled at zero while
waiting for a cube in the start box for 15 stable frames. Only then can another
DP3-generated cycle start.

All targets remain inside recorded action bounds. The learned motor endpoint
limits arm steps to 0.05859375 rad and gripper steps to 0.08 rad. Missing state or action packets,
excessive tracking error, an unverified task timeout, Ctrl+C, or process exit
sends stop and disables both arms. Zero-pose return is visible joint-space
motion; keep the table path clear and physical E-stop in hand.
