# OpenArm Pick-and-Place Checkpoint Archive

This archive preserves one representative checkpoint and the resolved RSL-RL/Isaac Lab
configuration for each policy version from v18 through v28. Raw training directories remain
ignored because they contain thousands of redundant checkpoints and several gigabytes of logs.

Each version directory contains:

- `checkpoint.pt`: the selected RSL-RL checkpoint.
- `agent.yaml`: the resolved agent configuration saved by the run.
- `env.yaml`: the resolved environment configuration saved by the run.

The original run name, original checkpoint name, file size, SHA-256, and selection rationale are
recorded in `manifest.json`. Verify the binaries from this directory with:

```bash
sha256sum --check SHA256SUMS
```

## Selected Checkpoints

| Version | Original checkpoint | Selection basis |
| --- | --- | --- |
| v18 | `hybrid_place_v18/model_825.pt` | User-requested historical grasp/transport reference |
| v19 | `reachable_place_v19/model_900.pt` | Diagnosed stable grasp/lift with incomplete basket travel |
| v20 | `legacy_transport_assisted_place_v20/model_225.pt` | Strongest documented stable v20 checkpoint |
| v21 | `stabilized_vertical_place_v21/model_400.pt` | Representative final vertical-place policy |
| v22 | `2026-07-10_21-32-18/model_1300.pt` | Nearest saved model to the documented iteration-1304 diagnosis |
| v23 | `2026-07-10_23-12-11/model_500.pt` | Documented grasp/lift/assist learning checkpoint |
| v24 | `2026-07-11_00-21-25/model_1125.pt` | GUI-reviewed checkpoint exposing metric/behavior mismatch |
| v25 | `2026-07-11_02-45-03/model_1000.pt` | Sampled-policy peak used for deterministic validation work |
| v26 | `2026-07-11_05-23-39/model_125.pt` | Nearest saved model to the documented iteration-114 peak |
| v27 | `2026-07-11_19-14-07/model_3650.pt` | Actor-mean checkpoint used to diagnose grasp failure |
| v28 | `2026-07-12_03-16-50/model_4999.pt` | User-validated successful grasp, transport, and placement |

## Source History Boundary

Versions v18-v24 were developed before version-specific source commits were established. Their
exact standalone source trees cannot be reconstructed from Git history; the corresponding changes
first appear cumulatively in commit `4455992`. The archived YAML files preserve the resolved runtime
configuration, but they are not substitutes for a full source snapshot.

Version-specific source boundaries are available from v25 onward:

| Version | Source commit |
| --- | --- |
| v25 | `4455992afa7f72d129a71c4677ed1f6ad6a7d383` |
| v26 | `3856bcacf22a7cb773ab7ddabf1d8bbc4e4f0506` |
| v27 | `816e94301c0edcfefc08355f170b77c4f921d8e7` |
| v28 | `b4bedeaab9f9b4b41bb7d887d10d03c91705481f` |

Do not compare TensorBoard sampled-action success directly with GUI actor-mean behavior. A
checkpoint is accepted only after actor-mean, first-complete-episode physical grasp, lift, transport,
and placement validation.
