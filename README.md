# OpenArm TestUSD Isaac Lab 작업

다음 경로에 저장된 OpenArm 양팔 USD 장면을 위한 외부 Isaac Lab 작업 모음입니다.

```bash
assets/openarm_test.usd
```

다른 로컬 USD 파일을 사용하려면 `OPENARM_TEST_USD_PATH`를 설정하십시오.

## 등록된 작업

- `Isaac-Reach-OpenArm-TestUSD-Bi-v0`
- `Isaac-Reach-OpenArm-TestUSD-Bi-Play-v0`
- `Isaac-PickPlace-OpenArm-TestUSD-Bi-v0`
- `Isaac-PickPlace-OpenArm-TestUSD-Bi-FineTune-v0`
- `Isaac-PickPlace-OpenArm-TestUSD-Bi-Play-v0`
- `Isaac-PickPlace-OpenArm-TestUSD-Bi-Stage-v0`
- `Isaac-PickPlace-OpenArm-TestUSD-Bi-Stage-Play-v0`

## 장면

- 현재 기본 장면은 `assets/openarm_test.usd`입니다.
- 저장소의 자산은 최종 로컬 `openarm_test_v2.usd` 배치를 사용하지만 파일명은 `openarm_test.usd`로 유지합니다.
- 로봇과 작업대는 분리되어 있습니다.
- 로봇은 아래쪽 받침대 위에 놓입니다.
- 바스켓 테이블은 로봇 앞 중앙에 있으며 물체를 놓을 수 있도록 폭을 넓혔습니다.
- 작업 코드는 로봇의 재질과 색상을 바꾸지 않습니다.

## 집기·놓기 정책 범위

- 집기·놓기 정책은 오른팔과 오른쪽 그리퍼만 제어합니다.
- 왼팔은 정책의 동작 공간에서 제외되며 PD 제어가 유지하는 기본 자세에 머뭅니다.
- 오른손이 물체로 바로 향하지 않고 먼저 물체 위의 파지 준비점에 접근하도록 보상합니다.
- 오른손이 작업대와 부딪힐 만큼 낮은 경로를 택하면 테이블 여유 높이 페널티를 부과합니다.
- 이전 15개 동작 정책으로 학습한 과거 집기·놓기 체크포인트는 이 작업 버전과 호환되지 않습니다.

날짜별 정책과 코드 변경 내역은 `docs/daily_changes/`에 한국어로 기록합니다.
선별한 v18-v30 체크포인트와 학습 당시 확정된 설정은
`artifacts/checkpoints/openarm_pick_place/`에 보관합니다. 사용자가 승인한 우수 기준은
`docs/validated_checkpoints.md`에서 따로 관리하며, 학습 중인 버전은 GUI 검증을 마치기 전까지 집계하지 않습니다.
현재 작업 인계서는 `docs/OPENARM_HANDOFF.md`에 있습니다. 과거 장면을 한 번만 구성할 때 사용한
스크립트는 `scripts/scene_authoring/legacy/`에 보존하며, 실행 중인 작업 코드에는 포함되지 않습니다.

## 실행

```bash
source /home/mh/isaac-stack-2.3.2/activate_isaaclab_2.3.2.sh
cd /path/to/OpenArm
```

정적 GUI 확인:

```bash
python scripts/environments/static_viewer.py --task Isaac-PickPlace-OpenArm-TestUSD-Bi-v0 --num_envs 1 --device cuda
```

과거 기본 작업의 간단한 실행 확인에만 사용하십시오. 현재 Stage 정책 학습 절차가 아닙니다.

```bash
python scripts/rsl_rl/train.py --task Isaac-PickPlace-OpenArm-TestUSD-Bi-v0 --num_envs 1024 --device cuda --max_iterations 5000 --headless
```

v32는 처음부터 새로 학습합니다. v31의 엄격한 양쪽 파지 조건, 물체와 TCP를 연동한 들기,
운반 속도 변화 제한, 정확한 놓기 게이트, 동작 상한은 그대로 유지합니다. PlaceAssist는 이제 한 방향으로만 진행되는
두 단계 smoothstep을 따릅니다. 바스켓 위에서 기다렸다가 수직으로 내려가지 않고 XY 이동을 마치면서
함께 하강합니다. 종료 페널티는 학습된 정책망이 파지 후 운반을 맡은 동안에만 두 배로 유지됩니다.
제어기가 맡는 최종 정렬 구간에서는 PPO가 파지를 피하도록 학습되지 않습니다. 기본 설정에서 이미
CUDA와 5000회 반복을 선택하므로 재개, 체크포인트, 실행 불러오기, 부트스트랩 인수는 추가하지 마십시오.

```bash
python scripts/rsl_rl/train.py --task=Isaac-PickPlace-OpenArm-TestUSD-Bi-Stage-v0 --num_envs=5120 --headless --run_name=fresh_coupled_descent_v32
```

TensorBoard는 가우시안 분포에서 표본을 뽑은 동작을 사용하므로 학습 진단용이며 체크포인트 승인
시험이 아닙니다. 체크포인트는 정책망의 평균 동작으로 처음 완료된 에피소드를 평가해 선별하십시오.
물리적으로 `lift_completed`와 `validated_transport`가 확인돼야 하며, 같은 정책망 평균 동작을 GUI에서도
확인해야 합니다.

과거 기본 작업의 GUI 실행 확인에만 사용하십시오. 체크포인트 평가는 아래 Stage-Play를 사용합니다.

```bash
python scripts/rsl_rl/train.py --task Isaac-PickPlace-OpenArm-TestUSD-Bi-v0 --num_envs 16 --device cuda --max_iterations 200
```

Stage 설정에 맞는 집기·놓기 체크포인트 실행:

```bash
python scripts/rsl_rl/play.py --task=Isaac-PickPlace-OpenArm-TestUSD-Bi-Stage-Play-v0 --num_envs=1 --device=cuda --seed=42 --checkpoint=logs/rsl_rl/openarm_testusd_bi_pick_place/<run>/model_<N>.pt --real-time
```

도달 작업 학습도 계속 사용할 수 있습니다.

```bash
python scripts/rsl_rl/train.py --task Isaac-Reach-OpenArm-TestUSD-Bi-v0 --num_envs 1024 --device cuda --max_iterations 5000 --headless
```
