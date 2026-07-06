# OpenArm Test USD Bimanual Reach Task

External Isaac Lab task that uses the original USD scene committed at:

```bash
assets/openarm_test.usd
```

Set `OPENARM_TEST_USD_PATH` to use a different local copy of the USD.

Registered tasks:

- `Isaac-Reach-OpenArm-TestUSD-Bi-v0`
- `Isaac-Reach-OpenArm-TestUSD-Bi-Play-v0`

Integrity rules used here:

- The original USD file is not edited.
- Isaac Lab source files are not edited.
- `/World`, `/Environment`, and `/Render` from the original USD are referenced into the task stage.
- The ground plane, packing table, robot base, robot support height, lighting, and render prims come from the original USD.
- No separate transport object is added. This task follows the OpenArm bimanual reach setup.
- The RL structure follows the Isaac Lab OpenArm bimanual reach config and reach MDP.

## Run

```bash
source /home/mh/isaac-stack-2.3.2/activate_isaaclab_2.3.2.sh
cd /path/to/OpenArm
```

Static GUI check:

```bash
python scripts/environments/static_viewer.py --task Isaac-Reach-OpenArm-TestUSD-Bi-v0 --num_envs 1 --device cuda
```

Headless training:

```bash
python scripts/rsl_rl/train.py --task Isaac-Reach-OpenArm-TestUSD-Bi-v0 --num_envs 128 --device cuda --headless --max_iterations 100
```

GUI training check:

```bash
python scripts/rsl_rl/train.py --task Isaac-Reach-OpenArm-TestUSD-Bi-v0 --num_envs 1 --device cuda --max_iterations 100
```

Play a checkpoint:

```bash
python scripts/rsl_rl/play.py --task Isaac-Reach-OpenArm-TestUSD-Bi-Play-v0 --num_envs 1 --device cuda --checkpoint logs/rsl_rl/openarm_testusd_bi_reach/<run>/model_100.pt
```
