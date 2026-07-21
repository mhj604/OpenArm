# ROMO-B + OpenArm Isaac Sim 5.1 model

This directory contains a reviewable, reproducible Isaac Sim model of the ROMO-B mobile base with
the NVIDIA OpenArm bimanual asset mounted toward the front.

The committed scene is [`assets/romo_b_openarm_manual_proxy.usda`](assets/romo_b_openarm_manual_proxy.usda).
It is ASCII USD so every dimension, joint, body relationship, and source annotation can be audited
in Git. The model is generated from
[`config/romo_b_openarm_spec.json`](config/romo_b_openarm_spec.json); do not edit the USDA by hand.

## What is exact, approximate, and unresolved

- Values printed in the provided ROMO-B text manual are copied without additional rounding.
- Values described by the manual as “approximately” are marked `manual_approximate`.
- Derived geometry and simulation-only choices are marked `derived`, `proxy`, or `unverified`.
- OpenArm geometry, links, joints, masses, and inertias come from the NVIDIA Isaac Sim 5.1 OpenArm
  payload. They are not reconstructed from ROMO-B documentation.
- The user-provided 17 cm placement is interpreted as `+0.170 m` from the ROMO-B geometric centre
  along +X. If 17 cm meant a setback from the front edge, change mount X to `+0.208 m` in the JSON
  and regenerate.

The ROMO-B manual does not provide enough information for a calibrated digital twin. Wheel width,
actual wheel-link origins, suspension geometry, inertias, tire parameters, centre of mass, and mount
bracket geometry remain explicitly unverified. See
[`../docs/ISAAC_SIM_ROMOB_OPENARM_KO.md`](../docs/ISAAC_SIM_ROMOB_OPENARM_KO.md) for the complete
boundary.

## Coordinate frame and articulation

- Origin: ROMO-B geometric centre projected onto the ground plane
- +X: forward
- +Y: left
- +Z: up
- OpenArm base pose: `(0.170, 0.000, 0.342) m`, identity rotation

ROMO-B is a floating 4WID-4WIS articulation with four Z-axis steering joints and four Y-axis wheel
joints. The fixed-base `root_joint` from the OpenArm payload is disabled, and
`openarm_body_link` is connected to the ROMO-B chassis by `openarm_mount_joint`. This avoids leaving
the arm fixed to the world or creating two nested articulations.

## Generate and validate

Run the builder and USD validator with an Isaac Sim 5.1 Python environment that exposes `pxr`:

```bash
${ISAAC_SIM_PATH}/python.sh simulation/scripts/build_romo_b_openarm_usda.py
${ISAAC_SIM_PATH}/python.sh simulation/scripts/validate_isaacsim_model.py
```

For the pip/conda Isaac Sim 5.1 installation used on this workstation, activate that environment
and use the included launcher (or set `ISAAC_PYTHON` to its Python executable):

```bash
ISAAC_PYTHON=/path/to/isaac-sim-env/bin/python \
  simulation/scripts/run_usd_tool.sh simulation/scripts/build_romo_b_openarm_usda.py
ISAAC_PYTHON=/path/to/isaac-sim-env/bin/python \
  simulation/scripts/run_usd_tool.sh simulation/scripts/validate_isaacsim_model.py
```

The first validation does not resolve the remote OpenArm payload and therefore works offline. To
also verify the OpenArm body and arm joints, use a machine with the Isaac asset cached or reachable:

```bash
${ISAAC_SIM_PATH}/python.sh simulation/scripts/validate_isaacsim_model.py --resolve-openarm
```

An offline/local OpenArm copy can be substituted for resolution testing without changing the
committed stage:

```bash
${ISAAC_SIM_PATH}/python.sh simulation/scripts/validate_isaacsim_model.py \
  --openarm-asset /path/to/openarm_bimanual.usd
```

Pure-Python source/value contract tests do not require Isaac Sim:

```bash
python -m unittest discover -s simulation/tests -v
```

Run a 120-step integration smoke test with Isaac Sim 5.1:

```bash
python simulation/scripts/run_isaacsim_smoke.py --steps 120
```

Add `--gui` to inspect the model interactively. The smoke test intentionally fails if the NVIDIA
OpenArm payload cannot be resolved. `--openarm-asset /path/to/openarm_bimanual.usd` is also
available for an offline cached copy.

## Files

- `config/romo_b_openarm_spec.json`: numeric source of truth and provenance classifications
- `assets/romo_b_openarm_manual_proxy.usda`: generated Isaac Sim stage
- `scripts/build_romo_b_openarm_usda.py`: deterministic USD builder
- `scripts/run_usd_tool.sh`: USD/PhysX schema launcher for an Isaac Sim pip/conda installation
- `scripts/validate_isaacsim_model.py`: non-rendering structural validator
- `scripts/run_isaacsim_smoke.py`: Isaac Sim integration smoke test
- `tests/test_model_contract.py`: standard-library numeric/source contract tests
