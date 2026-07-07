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
- `Isaac-PickPlace-OpenArm-TestUSD-Bi-Play-v0`

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

## Run

```bash
source /home/mh/isaac-stack-2.3.2/activate_isaaclab_2.3.2.sh
cd /path/to/OpenArm
```

Static GUI check:

```bash
python scripts/environments/static_viewer.py --task Isaac-PickPlace-OpenArm-TestUSD-Bi-v0 --num_envs 1 --device cuda
```

Headless pick-and-place training:

```bash
python scripts/rsl_rl/train.py --task Isaac-PickPlace-OpenArm-TestUSD-Bi-v0 --num_envs 1024 --device cuda --max_iterations 5000 --headless
```

GUI pick-and-place training check:

```bash
python scripts/rsl_rl/train.py --task Isaac-PickPlace-OpenArm-TestUSD-Bi-v0 --num_envs 16 --device cuda --max_iterations 200
```

Play a pick-and-place checkpoint:

```bash
python scripts/rsl_rl/play.py --task Isaac-PickPlace-OpenArm-TestUSD-Bi-Play-v0 --num_envs 16 --device cuda --checkpoint logs/rsl_rl/openarm_testusd_bi_pick_place/<run>/model_<N>.pt --real-time
```

Reach training remains available:

```bash
python scripts/rsl_rl/train.py --task Isaac-Reach-OpenArm-TestUSD-Bi-v0 --num_envs 1024 --device cuda --max_iterations 5000 --headless
```
