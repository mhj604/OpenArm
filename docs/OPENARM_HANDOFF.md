# OpenArm Pick-and-Place Handoff

Last updated: 2026-07-14 KST

## Repository And Tasks

```bash
source /home/mh/isaac-stack-2.3.2/activate_isaaclab_2.3.2.sh
cd /home/mh/.codex_tmp_openarm_remote
```

```text
Train: Isaac-PickPlace-OpenArm-TestUSD-Bi-Stage-v0
GUI:   Isaac-PickPlace-OpenArm-TestUSD-Bi-Stage-Play-v0
```

Current code target is untrained v34. The preserved v33 baseline completed all 5000 iterations; its
selected checkpoint is `2026-07-13_20-04-29_fresh_continuous_assist_v33/model_4999.pt`.

## V19 Result And Exact Cause

- valid run: `2026-07-10_17-55-28_reachable_place_v19`, inspected through `model_900.pt`.
- grasp/lift stayed about 97-98%, but mean actual-goal XY stopped near `0.133m` and object height near
  `0.98m`.
- assist, center, release, and success remained 0; timeout stayed about 98-99%.
- assist required `basket_reached <=0.10m` before its nominal outer radius mattered, so the controller
  never ran. The visible upward motion was the inherited actor, not a reversed controller/Jacobian.
- v18 actor weights were copied exactly, but v19 changed the old input distribution: goal X direction
  reversed, Y error shrank, and two existing Z inputs shifted by about `+0.10m`.
- v19 also copied an obsolete critic and used LR `2e-5`; without downstream events the actor drifted.

## V20 Policy

- actual reachable goal B: center `(0.37, 0.07, 0.33)`, train X `(0.36,0.38)`, Y `(0.06,0.08)`.
- inherited actor sees virtual waypoint `(0.48,0.20,0.33)`, which restores the v18 input range.
- the old path passes within `72.8mm` of the actual goal; all reset/goal corners pass within `86.8mm`.
- actor reaches actual-goal XY `0.14m`, then `assist_engaged` latches.
- strict basket state remains `0.10m`; center/release are `0.04m`; settle success is `0.05m`.
- assist stays active through `0.24m`; exit is `0.20m` with 24-step recovery dwell.
- before centering, vertical target is `min(current object z, 0.78m)`; after centering it is the lower
  sampled release level `0.73/0.68m`. Controller never commands upward motion.
- object-center Jacobian is `Jv - skew(p_object-p_tcp) @ Jw`, followed by 3D damped least squares.

## Basket Position

- physical prim: `/packing_table/container_h20`.
- local translation X: `0.333333 -> 0.283333m`.
- parent transform maps this to world Y `-0.03m`: basket center `(0.48,0.20) -> (0.48,0.17)`.
- actual goal is unchanged, so reachability is unchanged and near-wall clearance grows by `30mm`.
- worst randomized goal cube-edge clearance before release is X `76mm`, Y `68mm`.
- with arbitrary planar cube yaw, release/success guards still keep the cube inside the basket.

## V20 Reward And PPO

Stage reward has 17 terms.

```text
approach/lift/transport progress       +5 / +150 / +150
carry height signed progress to .78   +150
grasp/lift/assist/release events       +10 / +15 / +40 / +25
success / failure / timeout            +200 / -100 / -50
elapsed / gripper phase                -0.5 per sec / -4
carry band / clearance                 -10 / -2
actor-only action rate / joint vel     -1e-4 / -1e-4
```

The assist event is actor-controlled. Controller-generated basket entry has no actor reward. Signed
height potential is positive only while approaching `0.78m`, negative while moving away, and zero when
stationary. PPO is 3072 env, 48 steps, 24 mini-batches, 2 epochs, LR `5e-6`, entropy `1e-4`, gamma
`0.995`.

## V21 Vertical Fix

V20 solved transport entry but its no-rise clamp made Z control a downward-only ratchet. Object height
fell from about `0.645m` to `0.632m`, hold/assist then degraded, and release/success stayed 0.

V21 keeps all 69 observations, 8 actions, 17 rewards, goals, basket transforms, and state thresholds.
Only assist Z control changes:

```text
dz = 0.10 * (target_z - object_z) - 1.5 * object_vz * step_dt
dz clamp = [-0.0015m, +0.00075m] per control step
target_z = 0.78m before centered; sampled 0.73/0.68m after centered
```

This permits a smaller upward recovery only below target while braking upward velocity and retaining
twice the downward authority.

## Bootstrap

Source actor/noise:

```text
2026-07-10_16-30-21_hybrid_place_v18/model_1000.pt
```

Target:

```text
2026-07-10_19-10-00_model1000_actor_v20_bootstrap/model_0.pt
```

- actor first 66 columns and all later actor layers/log-std are exact copy.
- new `place_centered`, center dwell, and `assist_engaged` columns are zero initialized.
- actor/critic input shape `(256,69)`; critic is freshly initialized.
- iteration 0; Adam moments are empty but a valid 17-parameter group at LR `5e-6` is present.

This bootstrap is historical only. V22 must not load it or any other checkpoint.

## V22 Fresh Policy

V21 kept grasp/lift/assist near `98/98/97%` initially, but over about 160 new updates hold declined
from about `80%` to `69%`, basket from `96%` to `33%`, center from `5.6%` to `0.1%`, and object Z
stayed near `0.632-0.633m`. Release and success remained zero.

- remove the inherited virtual waypoint; policy and controller now share actual goal XY `(0.37,0.07)`.
- policy transport target Z is world `0.78m`, matching carry reward and controller.
- observation is 74: previous 69 plus three continuous grasp-quality values and two milestone latches.
- reward is 19: approach potential `30`, ready/candidate/acquired one-shot bonuses `5/10/20`, with the
  remaining actor-owned terms retained.
- failure starts at `-50` and ramps linearly to `-100` between common steps `4,800` and `14,400`
  (iterations 100-300 at a 48-step rollout).
- progress rewards pay only a new episode best while regressions stay negative, so discounted
  back-and-forth cycles cannot repay earlier progress.
- assist captures the object-TCP offset and controls the corresponding TCP target with linear `Jv`.
  DLS remains `0.08`; entry posture is held softly in the nullspace (`gain=.04`, cap `.003rad`).
- XY control is `kp=.12`, `kd=.25`, norm cap `.002m`; Z is `kp=.12`, `kd=.30`, down/up caps
  `.003/.0025m`; object velocity EMA is `.75`.
- every Cartesian command is target-directed and no larger than remaining error. Joint-limit scaling
  preserves the whole DLS vector direction, followed by a predicted TCP-direction check.
- the assist joint target persists to build enough PD effort but may lead current joints by at most
  `.05rad` and cannot exceed remaining Cartesian error. The 12-step handoff holds the prior target
  after hold flicker and current joint position after release.
- centering requires carry-height error `<=.03m`, preventing descent from starting at the old low Z.
- PPO is fresh-policy settings: 3072 env, 48 steps, 24 mini-batches, 2 epochs, noise `.30`, LR `5e-6`,
  entropy `1e-4`, gamma `.98`, maximum 2000 iterations.

## V23 Object Feedback Fix

At v22 iteration 1304, grasp/lift/assist reached about `99/97/97%` and hold `91%`, but basket was about
`19%`, center/release/success stayed zero, object Z was `.6356m`, and dXY was `.1197m`. Assist was active
over 92%, while all requested/applied Z and target-lead diagnostics stayed exactly zero. The fixed
entry object-TCP offset let TCP reach its nominal target while later cube slip disappeared from feedback.

- control error is now `target_object_position - actual_object_position` every step.
- damping uses clamped actual object velocity (`+/-0.25m/s`) with EMA `.80`.
- XY is `.12/.25`, `2.5mm`; Z is `.12/.30`, up/down `2.5/3mm`; deadband is `3mm`.
- predicted residual must shrink; persistent Cartesian lead is capped at XY/up/down `5/5/6mm`.
- fixed offset is diagnostic-only. Slip over `50mm` for 6 steps resets lead, freezes one step, and
  rebases the diagnostic offset.
- controller episode metrics now accumulate control steps, requested/incremental/applied Z, max slip,
  watchdog trips, posture error, and joint-target lead.
- lift/carry band is `.73/.75/.79m`, center is `.75 +/- .025m`, release remains `.73/.68m`.
- observation 74, reward 19, fresh shaping, failure ramp, and Stage PPO are unchanged.

## V24 Gravity, Torque Headroom, And Hold Recovery

V23 model 500 reached roughly `95/92/92%` grasp/lift/assist, but terminal object Z stayed near `.632m`
and center/release/success stayed zero. Its roughly `.9m` accumulated applied-Z metric was repeated
Jacobian prediction, not physical displacement.

- assist target lead was only about `.0208rad`, which at stiffness 400 requests about `8.3Nm` before
  damping and actuator clipping; it could not reliably support the arm.
- the source high-PD config requests disabled robot gravity, but `spawn=None` prevented that spawn
  rigid property from reaching the referenced original USD robot.
- `current_hold` also required object Z `>.630m`; at that boundary assist switched off and failure was
  confirmed after only 3 control steps.
- v24 authors `disableGravity=True` on every referenced robot rigid body while retaining cube gravity.
- assist entry preserves a bounded actor target. Persistent lead is joint `.12rad`, XY `.012m`, up
  `.025m`, down `.012m`, angular `.03rad`; per-step motion caps remain unchanged.
- v18's full 6D spatial DLS is restored: translation is solved together with zero angular motion, so
  the stronger lead cannot exploit the position-only nullspace to rotate the wrist out of the grasp.
- hold is now proximity/closure only, retaining the validated `.25` threshold. During the 12-step loss
  confirmation, object servo pauses at current q; reacquisition starts a fresh assist. Slip watchdog
  is tightened from `50mm/6` to `15mm/3` control steps.
- assist target ownership now includes handoff and hold-freeze, keeping actor-only rewards disabled
  whenever the actor target is overridden.
- actual object/TCP vertical displacement, command-aligned vertical progress, estimated effort-limit
  ratio, and TCP orientation error are logged separately from predicted displacement.
- reward 19, observation 74, PPO, goal geometry, carry `.73/.75/.79m`, center `.75 +/-.025m`, and
  release `.73/.68m` are unchanged because v23's fresh grasp/lift learning was healthy.

## Historical V24 Training Command (Superseded)

```bash
python scripts/rsl_rl/train.py --task Isaac-PickPlace-OpenArm-TestUSD-Bi-Stage-v0 --headless --num_envs 5120 --device cuda --max_iterations 2000 --run_name fresh_gravity_compensated_place_v24
```

This creates actor, critic, optimizer, and iteration counter from scratch. Do not add `--resume`,
`--load_run`, `--checkpoint`, actor transfer, or an iteration-zero bootstrap.

## Future Training Rule

V21 was the final run allowed to resume an earlier checkpoint. V22 through V24 follow the fresh-training rule.

## Decision Points

```text
model 25: grasp-ready/candidate/acquired events must be nonzero
model 100: grasp must exceed about 10% and trend upward
model 250: lift/assist >20%; command-aligned object/TCP Z progress must be positive
model 500: grasp >85%, lift/assist >75%, object Z >.70m, center nonzero
model 750: basket rising and one of ready/authorized/released nonzero
model 1000: success nonzero; otherwise stop and redesign
early stop: at model 500 center/release=0 and command-aligned Z progress <=0, or object Z <.68m
```

## Validation And Git

- basket transform and actual-goal reachability remain unchanged from v20.
- manager params have no missing/unknown required args; observation 74; reward 19.
- 20k randomized 6D Jacobian tests passed joint-step, joint-target, XY/Z lead, angular lead, and joint
  limit bounds; 19,979 nonzero cases remained target-directed.
- `py_compile` and `git diff --check` passed.
- command metrics can omit the terminal control step because IsaacLab resets the action manager before
  the command manager; this is at most a one-step diagnostic error and does not affect control/reward.
- the first v24 launch created only `model_0.pt`, hit a startup patch-buffer overflow, and then ended;
  no usable v24 training checkpoint exists yet.
- Historical v24 note: remote `main` was then at `deebda9`; the current repository state is listed below.

## V24 PhysX Patch Buffer Follow-up

The first 5120-env launch reported a required rigid patch capacity of `199728`, above the default
`163840`. `gpu_max_rigid_patch_count` is now `10 * 2**15 = 327680`, matching IsaacLab's existing
large-environment precedent. The affected launch must not be resumed because missing contact patches
can corrupt grasp/contact transitions; restart the same v24 command from scratch.

## V26 Failure Result

V25 model1000 reported roughly 97% grasp and 83% success in asynchronous stochastic training logs,
but deterministic first-complete-episode evaluation measured only 10.94% grasp and 5.47% success.
The log counted fast successful resets repeatedly, while training sampled Gaussian actions with
learned std about .148 and GUI play used the actor mean.

Measured causes and v26 corrections:

- the old .32 proximity gate could close with a finger body origin about 116 mm from the cube;
- the actor moved the TCP about 60 mm during the roughly 9-step finger closure;
- a missed close stayed latched and never reopened;
- v26 closes only in measured TCP-local XYZ and finger-symmetry bounds, freezes the arm target during
  closure, confirms contact for 3 steps, and verifies object/TCP coupling with a bounded 18 mm lift;
- failed closure or verification reopens and retries, with at most 3 failed attempts per episode;
- attempt keep proximity .38 is separate from the unchanged post-grasp transport hold .25;
- approach reward and the 3 grasp observations now use centered TCP-local geometry;
- Stage PPO initial noise is .05. Transport, PlaceAssist, basket, release, and success logic are kept.

Validation on the v25 model1000 with the new state machine, despite the changed meaning of three
observations: 128 deterministic first-complete episodes reached 80.47% ready, 53.91% contact candidate,
42.19% verified grasp, 39.84% lift, and 39.06% real transport. Failed termination was 2.34%. This is a
state-machine regression check, not a v26 policy result; v26 must train fresh.

V26 completed 5000 iterations, but grasp peaked at 2.083% at iteration 114 and then collapsed to zero.
The final policy avoided the grasp basin: timeout was 89.09% and the TCP-local offset moved about
20 cm away from its target. No v26 checkpoint should be resumed.

## Current V28 Mean-Robustness Candidate

V27 TensorBoard was not a fake physical-success signal: its sampled actions really moved the cube.
The acceptance interpretation was wrong. `model_3650.pt` learned Gaussian action std averaged `.1429`; sampled
rollouts reached roughly `97%` grasp while a 128-env actor-mean first-complete evaluation measured
ready/candidate/grasp/lift-transport/success `85.16/82.03/81.25/78.13/71.09%`. The main loss was before
the centered ready gate, not in bilateral contact or coupling validation.

- Cube nominal position is `(0.40,-0.15,0.625)`, 5 cm toward the right arm. Worst randomized cube-face
  clearance from the basket wall grows from about `9.6mm` to `59.6mm`.
- The right ready pose is IK-adjusted by the same world `-Y 50mm`; original TCP orientation and
  TCP-local grasp offset are preserved. Moving the cube without this pose correction is invalid.
- Best-only approach potential is replaced by current alignment quality `+5/s` and consecutive dwell
  up to `+1/s`. Together with pregrasp `-7.5/s` and elapsed `-0.5/s`, hovering is still negative.
- Raw `last_action` is removed because sampled train actions differ from actor-mean play and are also
  overwritten during close/retry/assist. Joint state, object/TCP geometry, velocity, and FSM state remain.
- Stage PPO starts with learnable noise std `.05` and entropy `0`; policy observation/action/reward is
  `75/8/20`. The learned std must still be monitored because `.05` is not frozen.
- V27 strict 3-step alignment, bilateral stopped-finger contact, actor-owned object/TCP coupling lift,
  retry, transport, PlaceAssist, release, and physical success conditions remain unchanged.
- The new observation and geometry contracts are incompatible with every older checkpoint. Train fresh.
- These changes target mean robustness; no v28 checkpoint exists yet, so deterministic improvement is
  unproven until the acceptance protocol below passes.

## Current Training Command

Run this as one line from `/home/mh/.codex_tmp_openarm_remote`:

```bash
python scripts/rsl_rl/train.py --task=Isaac-PickPlace-OpenArm-TestUSD-Bi-Stage-v0 --num_envs=5120 --device=cuda --max_iterations=5000 --headless --run_name=fresh_clearance_mean_grasp_v28
```

Do not add `--resume`, `--load_run`, `--checkpoint`, or bootstrap arguments.

## V28 Decision And Acceptance Rules

```text
protocol: actor-mean, corruption off, first complete episode per env
intermediate: seed 42 x 128 env at model 250 and 500
final: seeds 42/43/44/45 x 128 env = 512 episodes
final overall: lift_completed and validated_transport >=90%, success >=80%, timeout <=15%
final per-seed: lift/transport >=85%
final reset-Y bins: [-.165,-.1575), [-.1575,-.150), [-.150,-.1425), [-.1425,-.135]m
final per-Y-bin: each n>=100 and lift/transport >=85%
final GUI: Stage-Play `--seed=42`, 10 consecutive episodes, physical lift/transport >=9/10, empty transport 0
report: learned Gaussian std and same-protocol sampled/mean gap
sampled TensorBoard: training health and phase-opening diagnostics only; never checkpoint acceptance
model 100: centered alignment and candidate object/TCP lift must both be nonzero
model 250: run 128-env actor-mean first-complete evaluation; physical lift/transport must trend upward
model 500: repeat actor-mean evaluation and GUI; do not infer GUI behavior from sampled metrics
early stop: model 100-250 has candidates but zero paired object/TCP lift, or mean alignment moves away
```

## Current V31 Policy (Supersedes The V28 Current Sections Above)

User-approved references remain v18 `model_825.pt`, v28 `model_4999.pt`, and v29 `model_2100.pt`.
V29 is the direct behavior baseline because its strict 3-control grasp, transport slew, assist handoff,
descent, and release were all observed in the GUI. V18 uses weaker historical grasp validation and must
not replace the current bilateral contact/coupling contract.

V30 changed alignment confirmation from 3 to 4 controls and dwell shaping from 1 to 1.5. Its rolling-50
best was grasp/success `74.13/49.51%` at iteration 1781, then every task milestone became zero and timeout
became 100% from iteration 1950. PPO was identical to v29; the narrower 4-control gate and stronger dwell
combination is the strongest isolated regression cause.

V31 changes:

```text
alignment confirmation             4 -> 3 controls
persistent dwell shaping           1.5 -> 1.0
one-shot ready/contact rewards     keep 8 / 15
Stage fixed learning rate          5e-6 -> 2.5e-6
PlaceAssist XY gain                .12 -> .16
XY maximum step                    keep .0025m/control
low release target world/root      .674/.324 -> .670/.320m
release height gate                +/-40mm -> 2mm above / 3mm below
release maximum object speed       .12 -> .06m/s
controller position deadband       3 -> 2mm
```

Vertical gain/damping `.12/.30`, down cap `3mm/control`, joint slew, strict stopped bilateral contact,
object/TCP `12/10mm` coupled lift, assist handoff, release hold, high target `.730m`, and settle `.650m`
are unchanged. Release command and actual open height/error/speed are now latched as six metrics. Since
non-release episodes retain zero, divide command metrics by `task_release_commanded` and actual-open
metrics by `task_released` when reading conditional means.

Run fresh as one line from `/home/mh/.codex_tmp_openarm_remote`:

```bash
python scripts/rsl_rl/train.py --task=Isaac-PickPlace-OpenArm-TestUSD-Bi-Stage-v0 --num_envs=5120 --headless
```

The Stage config supplies CUDA and 5000 iterations. Do not add resume/load-run/checkpoint/bootstrap.
With the halved learning rate, inspect grasp by model 500-1000, lift/transport near model 2500, and
release/success mainly over model 3500-4500. Every 25 iterations is saved, so a later decline does not
overwrite an earlier candidate. Final selection still requires actor-mean 4 seeds x 128 env plus GUI
10 episodes; sampled TensorBoard values alone never establish a good checkpoint.

## Current V32 Policy (Supersedes V31)

V31 run `2026-07-13_01-57-04`, actor-mean GUI `model_2500.pt` grasped, lifted, and transported well,
but paused above the basket and took too long to descend and align. Rolling-50 showed
grasp/transport `94.77/94.33%`, center `80.62%`, release authorization `54.78%`, success `52.46%`,
and timeout `47.05%`. The direct cause was the binary `.750m carry -> .670m release` target switch
only after XY/height/speed centering dwell.

V32 keeps all strict grasp/release checks and all motion caps. Its deterministic assist target is:

```text
XY 120 -> 40mm: Z carry .750m -> min(carry, release + 20mm), cubic smoothstep
XY  40 ->  2mm: Z preplace    -> exact release, cubic smoothstep
descent progress: monotonic latch; XY rebound cannot command an upward bounce
```

At the low target, the 40mm transition leaves `16.46mm` cube-bottom clearance above the rim. Maximum
target-Z advance for a capped 2.5mm XY control is `2.812mm`, inside the unchanged 3mm downward cap.
Maximum Cartesian steps, damping, joint step, DLS, target leads, 2mm/3mm release height guard, and
`.06m/s` speed guard are unchanged. XY/Z gains increase `.16/.12 -> .20/.16` only to shorten the
small-error proportional tail. Center and release confirmation decrease `5 -> 3` controls. Center is
planar-only; exact height and full speed are checked once by release-ready. New metrics are
`task_descent_progress` and `task_descent_target_height`.

Terminal failure/timeout `2x` now applies only after verified grasp and before `assist_engaged`, while
the learned actor still owns transport. Once deterministic assist owns final alignment, the cost is the
normal `1x`; this removes the v31 incentive to avoid grasp after controller-side timeouts.

Run fresh as exactly one shell line:

```bash
python scripts/rsl_rl/train.py --task=Isaac-PickPlace-OpenArm-TestUSD-Bi-Stage-v0 --num_envs=5120 --headless --run_name=fresh_coupled_descent_v32
```

Do not resume or bootstrap. Check verified grasp at model 500, lift/transport at model 1000, then compare
center/release authorization/success/timeout every 500 models from 1500. Use actor-mean and GUI for the
final decision; sampled TensorBoard success is diagnostic only.

## Current V33 Policy (Supersedes V32)

V32 `2026-07-13_12-22-03/model_2875.pt` preserved grasp and transport but used about 248 assist
controls (`8.27s`) per episode. V33 keeps the `75/8/20` policy contract, 16-second episode, PPO,
strict grasp/coupling/release/success gates, and all existing speed/joint/lead limits.

V33 changes only the verified bottlenecks:

- A physically empty, stopped close is confirmed for 3 controls and retried early; valid near-contact
  at the original 30-control deadline receives a latched 12-control grace period. Strict contact and
  coupled object/TCP lift requirements are unchanged.
- Only automatic closing/reopening pose holds are excluded from actor pregrasp delay and centering
  shaping. Verification and bounded handoff remain actor-owned and therefore keep their time incentive.
- Assist carries measured goal-directed object velocity into a 12-control quintic handoff. A
  jerk-aware stopping envelope brakes to the 2mm deadband without an end-point projection jump.
- Active requested XY/Z rows are solved first. Inactive translation and orientation are corrected in
  the active-task nullspace. Joint-limited hard-safe candidates are compared by actual progress; a
  posture candidate is used only when it preserves at least 95% of the fastest candidate.
- Target lead uses a fully checked chain: new bounded lead, previous bounded lead, one-step lead, then
  current-pose hold. Entry seeding and every moving fallback apply lead/soft-limit bounds first and the
  previous-target `.03rad/control` bound last, then revalidate translation, angular lead and finiteness
  without rescaling away continuity. A kinematically incompatible 1mm inactive-leak case may still hold
  for safety and is counted explicitly rather than silently throttling another axis.
- Train and play now share the physical world release target `.670m`; exact descent begins by XY 5mm,
  while release XY 4cm, success XY 5cm, height `+2/-3mm`, and speed `.06m/s` remain strict.

Run fresh as exactly one shell line from `/home/mh/.codex_tmp_openarm_remote`:

```bash
python scripts/rsl_rl/train.py --task=Isaac-PickPlace-OpenArm-TestUSD-Bi-Stage-v0 --num_envs=5120 --device=cuda --max_iterations=5000 --headless --run_name=fresh_continuous_assist_v33
```

Do not resume/bootstrap. Watch empty-close/grace counters, phase timeout one-hot metrics, assist active
progress ratio, low-progress controls, incremental safety stops, and each lead fallback level. Static
tests cannot replace actor-mean and GUI validation against the actual PhysX Jacobian.

### V33 Completed Result

The fresh run `2026-07-13_20-04-29_fresh_continuous_assist_v33` completed all 5000 iterations and
saved `model_4999.pt`. Its final rolling-50 sampled metrics were grasp `97.30%`, validated transport
`97.24%`, center `82.91%`, release authorization `69.22%`, success `68.98%`, timeout `31.04%`, and
failure `0.0074%`.

The user reviewed `model_4999.pt` in the GUI and judged the overall result good enough to preserve.
It is a strong v33 candidate, not a formal acceptance pass: `90.2%` of its remaining timeouts occur
in assist-to-center or center-to-release, and the GUI exposed intermittent basket-rim interference
near the final target. Preserve its grasp, transport, strict outcome gates, and continuous handoff as
the v34 baseline; fix final basket-relative clearance and assist convergence without weakening the
success definition.

## Current V34 Policy (Untrained)

V33 already reaches rolling-50 grasp/transport `97.30/97.24%`; changing the grasp rewards, policy
observations, goal, or success gates would confound the remaining final-assist problem. V34 therefore
keeps `75 observations / 8 actions / 20 rewards`, PPO, the 16-second episode, the `.670m` release
target, and every strict grasp/release/success condition.

The verified controller faults and fixes are:

- V33 constrained current lead and joint limits, then applied previous-target continuity last. The
  final continuity correction could reopen the `.12rad` lead bound. V34 projects directly into the
  component-wise intersection of soft joint limits, current `+/- .12rad`, and previous target
  `+/- .03rad`.
- The steady target no longer accumulates `previous_target + delta`. It is a current-relative
  three-control lookahead, `current + 3 * safe_delta`, while output continuity remains `.03rad` per
  control.
- Joint-limit scaling is relative to measured current joints instead of a stale target. An empty
  constraint intersection explicitly holds/rebases to current pose and clears Cartesian velocity and
  acceleration before the next control.
- Entry seeding is the final target for the first assist control. The steady lead path starts on the
  next control, preventing the old possible `.03 + .03rad` same-control jump.
- Invalid DLS candidates retain their invalid state; a zeroed candidate can no longer be revived as a
  valid zero-motion lead. Preferred, previous, incremental, and hold fallbacks now share the same
  repair and validation path.
- XY and Z stop independently at `5mm` and `2mm`. The release gate remains XY `40mm`, success XY
  `50mm`, height `+2/-3mm`, and speed `.06m/s`; no success condition was weakened.

Low-cost training telemetry now separates predicted XY/Z progress from actual next-control object
tracking, records lead rebases/backlog and current/target joint-limit proximity, and labels the
pre-clip implicit-actuator value as `effort_demand_ratio`. Direct basket ContactSensor processing is
not enabled in the 5120-env training task because its FPS and PhysX-buffer cost has not been measured.
Broad domain randomization is also deferred until this nominal controller passes actor-mean and GUI
acceptance.

Run fresh as exactly one shell line from `/home/mh/.codex_tmp_openarm_remote`:

```bash
python scripts/rsl_rl/train.py --task=Isaac-PickPlace-OpenArm-TestUSD-Bi-Stage-v0 --num_envs=5120 --device=cuda --max_iterations=5000 --headless --run_name=fresh_current_relative_assist_v34
```

Do not add resume, load-run, checkpoint, or bootstrap arguments. Compare model 500 for grasp, model
1000 for lift/transport and first center/release events, then models 2000, 3500, and 4500 for success
and timeout. Stop and redesign if model 1000 has grasp/lift below 80% or zero center/release events.
Final acceptance still requires actor-mean multi-seed evaluation and ten GUI episodes; sampled
TensorBoard success alone is not acceptance.
