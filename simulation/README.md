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
- The ROMO-B shell, chamfered plan outline, recessed top plate, single circular wheel-house
  openings, and wheel hubs
  are reconstructed from the manual's page 5 orthographic drawing and page 6 photograph. Printed
  envelope dimensions constrain the result; undimensioned contour coordinates are explicitly
  marked `manual_drawing_traced_and_scaled`.
- The recessed top uses the 13-by-9 lattice visible on page 5. Its pitch is 0.050 m in both axes,
  as confirmed by the user. Four 3-by-2 corner groups are occupied by the circular covers, leaving
  93 visible small-hole markers rather than the previous arbitrary 11-by-7 pattern.
- Derived geometry and simulation-only choices are marked `derived`, `proxy`, `traced`, or
  `unverified`.
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

The maximum effective collision mode supported by the Isaac Sim 5.1 articulation is enabled: all
32 colliders and all 32 rigid bodies are explicitly active, articulation self-collision is on, and
no USD filtered-pair or collision-group exclusion is retained. This enables external contacts and
non-adjacent ROMO-B/OpenArm self-collision. PhysX always filters directly joint-connected
parent-child articulation links and ignores per-joint `collisionEnabled` settings; enabling those
warning-only flags would not change contact behavior. Direct-neighbour collision would require
restructuring the model without the current single articulation.

The previous arbitrary 0.450 m blue box has been removed. The visible shell uses the manual-stated
0.597 m overall width; the distinct 0.529 m track locates the wheel centres. The derived 0.068 m
tire width therefore places the outer tire faces at the same 0.597 m envelope. The side panels are
open with exactly one fixed circular wheel house per wheel, matching the page 5 left/right side
elevations. Each circle is centred on the neutral wheel centre and uses a drawing-traced 0.126 m
opening radius. There is no separate inner/outer fender ribbon and therefore no duplicate visible
arch at any wheel.

A four-part compound chassis collider covers the 0.320 m steering-clear core, the full 0.597 m
deck above the tire envelope, and lower front/rear end members. The 361-angle -30-to-+30 degree
wheel sweep is retained only for invisible collision auditing; it never deforms the manual-traced
visible circles. Wheel-house radius, wheel width, suspension travel, and panel thickness are not
dimensioned in the manual, so the traced contour and collision pieces must not be treated as
production CAD. Re-run the integration smoke test after changing any mount, collision proxy, or
joint pose.

The current 320-step `all +30/all -30/manual Pivot/neutral` PhysX steering sweep checks every wheel
against every compound chassis collider in the chassis frame and audits contact reports on every
physics step. All 320 AABB samples passed with a minimum three-dimensional gap of `0.008021 m`,
maximum steering target error of `0.516825 deg`, and no penetrating or impulsive self-contact.

The NVIDIA bimanual zero pose penetrates a non-adjacent base/link collider when unfiltered
self-collision is enabled. The generated stage therefore starts from a PhysX-validated symmetric
pose: left/right joint2 at `-90/+90 deg` and both grippers open at `0.044 m`. The shoulder angles
are simulation start-pose choices, not an official or measured hardware home pose; `0.044 m` is the
payload's driven finger-joint upper limit. A 120-step hold produced no negative OpenArm separation
and no OpenArm contact impulse. Two base/link2 contact-offset pairs remain at positive separation
with zero impulse.

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

Run a 320-step integration smoke test with Isaac Sim 5.1. Add `--steering-sweep` to exercise all
four steering joints through all +30, all -30, the manual Pivot combination, and neutral. The test
verifies that every target is reached within 1 degree and checks every wheel against every compound
chassis collider plus contact reports on every physics step:

```bash
python simulation/scripts/run_isaacsim_smoke.py --steps 320 --steering-sweep
```

Add `--gui` to inspect the model interactively. The smoke test intentionally fails if the NVIDIA
OpenArm payload cannot be resolved. `--openarm-asset /path/to/openarm_bimanual.usd` is also
available for an offline cached copy.

## Files

- `config/romo_b_openarm_spec.json`: numeric source of truth and provenance classifications
- `assets/romo_b_openarm_manual_proxy.usda`: generated Isaac Sim stage
- `scripts/build_romo_b_openarm_usda.py`: deterministic USD builder
- `scripts/romo_b_geometry.py`: pure body-outline and full-steering wheel-envelope calculations
- `scripts/run_usd_tool.sh`: USD/PhysX schema launcher for an Isaac Sim pip/conda installation
- `scripts/validate_isaacsim_model.py`: non-rendering structural validator
- `scripts/run_isaacsim_smoke.py`: Isaac Sim integration smoke test
- `tests/test_model_contract.py`: standard-library numeric/source contract tests
