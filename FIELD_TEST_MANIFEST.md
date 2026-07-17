# OpenArm Field-Test Snapshot

This branch preserves the OpenArm-v1 bimanual field-test environment built on
2026-07-17, from the initial ACT/RealSense/CAN setup through the current DP3
training, learned rollout, and raw-demonstration replay diagnostics.

## Repository layout

- The repository root is the current `openarm_dp3` project.
- `field_test_components/openarm_act` contains the earlier ACT data collection,
  training, RealSense driver, CAN telemetry, and active rollout environment.
- `field_test_components/openarm_imitation` contains the initial modular
  imitation-learning and DP3 collection prototype.
- `field_test_components/exo_openarm_bridge_ros2` contains the ROS 2 bridge
  workspace source used during early ExoArm-to-OpenArm integration.
- `field_test_components/openarm_teleop` is the working source snapshot based on
  upstream `enactic/openarm_teleop@eb2d49338bf70ace95282ea724903849397b7811`,
  including the ExoArm bridge and local OpenArm-v1 launch-path changes.
- `field_test_components/openarm_can` is the working source snapshot based on
  upstream `enactic/openarm_can@4f5d0c107155c85b0775492d62c74bda80c94dd7`.
- `field_test_components/openarm_description` records the URDF/mesh source at
  upstream commit `1daa95b71565a6696242013a42c4d3af689cbdb3`.

## Hardware bindings

The committed configs describe the field machine rather than generic example
values. They include the verified left/right CAN assignments, D435ICB chest
camera serial, D405 left-wrist camera serial, collection schema, workspace
calibration, zero-pose profile, and the final operator-tuned rollout values.

No right-wrist camera is configured.

## Intentionally excluded

The following are local runtime products and are not suitable for normal Git:

- raw and processed datasets;
- model checkpoints and training outputs;
- logs and temporary recordings;
- virtual environments, package caches, build/install trees, and vendored
  third-party repositories;
- timestamped rollback backups;
- generated Python bytecode and test caches.

The source configs keep the exact local paths to those assets so the field
machine can continue using them. See each component's README and RUNBOOK for
collection, training, replay, rollout, safety, and rollback procedures.

## Safety

Active rollout and replay scripts require explicit environment locks and typed
terminal confirmation. They can enable and move both OpenArm arms. Keep the
physical E-stop available and do not run the ExoArm bridge and an autonomous
motor server simultaneously.
