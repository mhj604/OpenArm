# OpenArm TestUSD Isaac Lab Tasks

External Isaac Lab tasks for the OpenArm bimanual USD scene committed at:

```bash
assets/openarm_test.usd
```

Set `OPENARM_TEST_USD_PATH` to use a different local USD file.

## Registered Tasks

- `Isaac-Reach-OpenArm-TestUSD-Bi-v0`
- `Isaac-Reach-OpenArm-TestUSD-Bi-Play-v0`
- `Isaac-PickPlace-OpenArm-TestUSD-Bi-v0`
- `Isaac-PickPlace-OpenArm-TestUSD-Bi-FineTune-v0`
- `Isaac-PickPlace-OpenArm-TestUSD-Bi-Play-v0`
- `Isaac-PickPlace-OpenArm-TestUSD-Bi-Stage-v0`
- `Isaac-PickPlace-OpenArm-TestUSD-Bi-Stage-Play-v0`

## Scene

- `assets/openarm_test.usd` is the current default scene.
- The committed asset uses the final local `openarm_test_v2.usd` layout, but keeps the repository filename `openarm_test.usd`.
- The robot and work table are separated.
- The robot is supported by the lower table.
- The basket table is centered in front of the robot and widened for object placement.
- Robot materials/colors are not modified by the task code.

## Pick-and-Place Policy Scope

- The pick-and-place policy controls only the right arm and right gripper.
- The left arm is removed from the policy action space so it stays at its default PD-held pose.
- The right hand is rewarded to approach a pre-grasp point above the object before moving toward the object.
- A right-hand table-clearance penalty discourages low paths that collide with the work table.
- Older pick-and-place checkpoints trained with the previous 15-action policy are not compatible with this task version.

Daily policy/code changes are tracked in Korean under `docs/daily_changes/`.
Selected v18-v30 checkpoints and their resolved training configurations are archived under
`artifacts/checkpoints/openarm_pick_place/`. User-approved good baselines are tracked separately in
`docs/validated_checkpoints.md`; pending training versions are not counted until GUI validation.

## Run

```bash
source /home/mh/isaac-stack-2.3.2/activate_isaaclab_2.3.2.sh
cd /path/to/OpenArm
```

Static GUI check:

```bash
python scripts/environments/static_viewer.py --task Isaac-PickPlace-OpenArm-TestUSD-Bi-v0 --num_envs 1 --device cuda
```

Legacy base-task smoke only; this is not the current Stage policy workflow:

```bash
python scripts/rsl_rl/train.py --task Isaac-PickPlace-OpenArm-TestUSD-Bi-v0 --num_envs 1024 --device cuda --max_iterations 5000 --headless
```

Train v31 completely from scratch. It restores the validated v28/v29 three-control grasp entry while
retaining v30's one-shot ready and physical-contact bonuses. The strict bilateral contact, coupled
object/TCP lift, v29 transport slew, assist handoff, and release hold remain unchanged. The final
release is now authorized only within 2 mm above or 3 mm below the 0.670 m target and below 0.06 m/s.
The Stage learning rate is halved for the long run to reduce post-peak policy drift. Configuration
defaults already select CUDA and 5000 iterations; do not add resume, checkpoint, load-run, or
bootstrap arguments:

```bash
python scripts/rsl_rl/train.py --task=Isaac-PickPlace-OpenArm-TestUSD-Bi-Stage-v0 --num_envs=5120 --headless
```

TensorBoard uses sampled Gaussian actions and is a training diagnostic, not a checkpoint acceptance
test. Select checkpoints only with actor-mean, first-complete-episode evaluation; require physical
`lift_completed` and `validated_transport`, then confirm the same actor mean in the GUI.

Legacy base-task GUI smoke only; use Stage-Play below for checkpoint evaluation:

```bash
python scripts/rsl_rl/train.py --task Isaac-PickPlace-OpenArm-TestUSD-Bi-v0 --num_envs 16 --device cuda --max_iterations 200
```

Play a stage-aligned pick-and-place checkpoint:

```bash
python scripts/rsl_rl/play.py --task=Isaac-PickPlace-OpenArm-TestUSD-Bi-Stage-Play-v0 --num_envs=1 --device=cuda --seed=42 --checkpoint=logs/rsl_rl/openarm_testusd_bi_pick_place/<run>/model_<N>.pt --real-time
```

Reach training remains available:

```bash
python scripts/rsl_rl/train.py --task Isaac-Reach-OpenArm-TestUSD-Bi-v0 --num_envs 1024 --device cuda --max_iterations 5000 --headless
```
