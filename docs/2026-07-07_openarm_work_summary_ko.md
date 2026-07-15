# 2026-07-07 OpenArm 작업 정리

이 문서는 2026년 7월 7일 OpenArm Isaac Lab 작업 중 GitHub에 반영한 핵심 변경만 정리한다.

## 1. USD 기본 장면 정리

- 로컬에서 최종 확정한 `openarm_test_v2.usd`를 GitHub의 기본 USD 파일인 `assets/openarm_test.usd`로 반영했다.
- GitHub에는 `openarm_test_v2.usd`라는 별도 이름을 추가하지 않았다.
- 저장소의 기본 장면은 계속 `assets/openarm_test.usd`다.
- 장면 구성은 로봇과 작업 테이블을 분리한 형태로 정리했다.
- 로봇은 낮은 받침 테이블 위에 두고 바스켓이 있는 높은 테이블은 로봇 앞쪽 중앙에 배치했다.
- 높은 테이블은 바스켓과 물체 배치 공간을 확보하도록 가로 폭을 넓혔다.
- 로봇 색상과 로봇 자산 자체는 작업 코드에서 수정하지 않았다.

## 2. 도달 튜토리얼 확인

- `Isaac-Reach-OpenArm-Bi-v0`는 집기·놓기가 아니라 양팔 말단 작동부가 목표 자세를 따라가도록 학습하는 도달 작업이다.
- 이 작업은 OpenArm 양팔 관절체, 관절 동작, 보상 구성이 정상인지 확인하는 선행 튜토리얼로 사용했다.
- 5000회 반복 학습 후 TensorBoard에서 최종 평균 보상이 약 `-0.03` 수준으로 수렴하는 것을 확인했다.
- `Isaac-Lift-Cube-OpenArm-v0`와 보상 척도가 다르므로 두 그래프의 절대 보상값을 직접 비교하면 안 된다.

## 3. 집기·놓기 작업 추가

새 작업을 추가했다.

- `Isaac-PickPlace-OpenArm-TestUSD-Bi-v0`
- `Isaac-PickPlace-OpenArm-TestUSD-Bi-Play-v0`

추가된 주요 파일은 다음과 같다.

- `source/openarm_testusd_task/openarm_testusd_task/tasks/testusd_pick_place/__init__.py`
- `source/openarm_testusd_task/openarm_testusd_task/tasks/testusd_pick_place/testusd_pick_place_env_cfg.py`
- `source/openarm_testusd_task/openarm_testusd_task/tasks/testusd_pick_place/mdp.py`
- `source/openarm_testusd_task/openarm_testusd_task/tasks/testusd_pick_place/agents/rsl_rl_ppo_cfg.py`
- `source/openarm_testusd_task/openarm_testusd_task/tasks/__init__.py`

현재 PAP 작업은 오른손 중심의 첫 학습 버전이다.

- 오른팔 관절 7개만 관절 위치 동작으로 제어한다.
- 오른손 그리퍼만 이진 열기·닫기 동작으로 제어한다.
- 왼팔은 정책 동작 공간에서 제거해 기본 자세를 PD 제어로 유지하도록 했다.
- 이전 15차원 동작 PAP 체크포인트는 현재 8차원 동작 작업과 호환되지 않는다.
- 학습 물체는 안정성 확인을 위해 큐브 1종으로 제한했다.

## 4. 보상과 종료 조건

PAP 보상은 `Isaac-Reach-OpenArm-Bi-v0`를 그대로 복사하지 않았다. 도달 튜토리얼은 양팔 말단 작동부가 목표 자세를 따라가는지 확인하는 용도다. PAP에는 물체 상태와 그리퍼 닫기, 들기, 목표 추적이 필요하다. Isaac Lab 들기 계열 MDP의 물체 이송 보상 방식을 참고해 현재 `openarm_test.usd`의 양팔 OpenArm 구조에 맞는 오른손 중심 작업으로 다시 구성했다.

현재 보상 구성은 다음과 같다.

- 오른손 TCP와 물체 위 파지 전 지점 사이 거리 보상
- 오른손 TCP와 물체 파지 지점 사이 거리 보상
- 오른쪽 두 손가락 강체와 물체 사이 평균 거리 보상
- 물체 근처에서 오른쪽 그리퍼를 닫는 보상
- 그리퍼가 닫힌 상태에서 물체 위 들기 목표로 올라가는 보상
- 낮은 높이에서 먼저 들어 올리는 `lifting_object_early` 보상
- 최종 들기 보상
- 물체가 목표 위치에 가까워지는 목표 추적 보상
- 오른손이 테이블 높이 아래의 낮은 경로를 지나갈 때 적용하는 테이블 이격 벌점
- 동작 변화량 벌점
- 관절 속도 벌점

종료 조건은 다음과 같다.

- 에피소드 시간 초과
- 물체가 너무 낮게 떨어지는 경우
- 물체가 목표 위치에 들어온 경우

기존 들기 MDP의 말단 작동부 거리 계산은 현재 USD 구조와 맞지 않는 부분이 있었다. 이에 `mdp.py`에 현재 OpenArm 강체를 직접 읽는 보상 함수를 추가했다. 주요 추가 함수는 `object_body_offset_distance()`, `object_bodies_average_distance()`, `gripper_closed_near_object_bodies()`, `closed_gripper_lift_target()`, `body_height_below_minimum()`이다.

## 5. PPO 안정화

초기 학습 중 `value_function loss: inf`와 `normal expects all elements of std >= 0.0` 오류가 발생했다.

같은 오류가 반복되지 않도록 PPO 설정을 안정화했다.

- 경험적 정규화 활성화
- 동작 잡음을 로그 표준편차 방식으로 변경
- 학습률을 `3e-5`로 낮춤
- 고정 학습률 조정 방식 사용
- 엔트로피 계수 축소
- 미니배치와 에포크 설정 조정

이후 100회 반복 화면 없는 검증은 정상 종료했다.

## 6. 현재 학습 상태

PAP 작업의 학습 절차는 정상 동작한다.

초기 5000회 반복 결과는 아직 집기·놓기 성공으로 보기 어려웠다.

- 평균 보상은 약 `0.13` 수준으로 안정화됐다.
- `reaching_object` 보상은 생겼다.
- `lifting_object`, `object_goal_tracking`, `object_reached_goal`은 거의 0이었다.

초기 5000회 반복 결과를 보면 “오른손이 물체에 접근하는 초기 정책은 잡히지만, 집기와 운반은 아직 보상 설계와 초기 자세를 더 다듬어야 하는 단계”였다.

그 이후 확인된 문제는 다음과 같았다.

- 오른팔이 낮은 경로로 올라가며 테이블에 걸려 물체를 집지 못했다.
- 왼팔이 정책 동작에 포함되어 있어 뒤나 옆으로 불필요하게 움직였다.

이 문제를 해결하려고 PAP 작업을 수정했다.

- 왼팔 관절을 동작·관측·보상 벌점 대상에서 제거했다.
- `arm_action`은 오른팔 관절 7개만 제어한다.
- `gripper_action`은 오른손 그리퍼의 이진 동작 1개만 사용한다.
- 오른손 위치를 관측에 추가했다.
- 물체 중심으로 바로 들어가는 보상 대신 물체 위 파지 전 지점과 근접 파지 지점을 분리했다.
- 오른손이 물체 x 좌표 쪽으로 전진하도록 `right_hand_forward_alignment` 보상을 추가했다.
- 오른손이 로봇 기준 `x=0.40 m` 이상 전방으로 나가도록 `right_hand_forward_progress` 보상을 수정했다.
- 오른손이 로봇 기준 `z=0.43 m` 이상으로 올라가도록 `right_hand_lift_to_pre_grasp` 보상을 수정했다.
- 오른손이 로봇 기준 x 음수 방향으로 뒤로 빠지는 동작은 `right_hand_backward_guard` 벌점으로 약하게 막았다.
- 오른손이 `z=0.62 m` 아래로 내려가는 낮은 경로는 `right_hand_table_clearance` 벌점으로 막았다.
- 오른팔 준비 자세를 FK 탐색으로 찾은 물체 위 파지 전 지점 근처 관절값으로 변경했다.
- 오른팔 동작 배율을 `1.0`에서 `0.5`로 낮춰 준비 자세 주변의 과도한 충돌을 줄였다.

추가로 21:30 학습 중간 점검에서 좌표계 오류를 확인했다. 기존 전진·후퇴 보상은 월드 좌표로 계산됐다. 이 때문에 `num_envs=1024` 복제 환경의 위치에 따라 실제로 후퇴하지 않았는데도 큰 후퇴 벌점이 들어갔다. 이를 로봇 기준 좌표 함수로 바꿨고 20회 반복 간이 검사에서 `right_hand_backward_guard=0`, `right_hand_table_clearance=0`, 평균 보상이 양수가 된 것을 확인했다.

이후 21:50 학습 중간 점검에서는 파지 전 보상이 포화되는 문제를 확인했다. 손은 안전하게 물체 위에 머물렀지만 `lifting_object`가 계속 0이었다. `PRE_GRASP_OFFSET_W`와 `RIGHT_HAND_TARGET_HEIGHT_B`를 낮추고 파지 전·높이 보상 가중치를 줄였다. 실제 파지 지점 접근과 들기를 강화하려고 `reaching_object`와 `lifting_object` 가중치는 높였다.

이후 300회 반복 검증에서 손 위치 보상은 상승했지만 `object_reached_goal`은 계속 0이었다. 손 프레임이 물체에 대략 접근하는 보상만으로는 부족하다고 판단해 더 좁은 파지 지점 보상과 물체 근처에서 오른쪽 그리퍼를 닫는 보상을 추가했다. 왼팔의 의도하지 않은 움직임을 억제하려고 왼팔 관절 위치·속도 벌점도 추가했다.

마지막으로 22:30 검증에서 오른손 기준 강체를 `openarm_right_hand`에서 `openarm_right_ee_tcp`와 오른쪽 손가락 강체로 분리했다. 물체도 Isaac Sim 기본 `DexCube` USD로 바꾸고 초기 물체 위치와 오른팔 준비 자세를 다시 맞췄다. 초기화 직후 TCP와 물체 거리는 약 `4 cm`, 두 손가락과 물체 거리는 약 `5-7 cm` 수준이었다. 사전에 정한 닫기·들기 동작에서는 물체 z가 약 `0.96 m`까지 올라가므로 현재 USD 물리와 그리퍼는 물체를 들어 올릴 수 있다.

수정 후 100회 반복 화면 없는 검증 결과는 다음과 같다.

- 실행: `logs/rsl_rl/openarm_testusd_bi_pick_place/2026-07-07_22-29-23`
- 체크포인트: `model_99.pt`
- `Mean reward: 523.60`
- `Episode_Reward/lifting_object_early: 11.4702`
- `Episode_Reward/lifting_object: 18.7431`
- `Episode_Reward/object_goal_tracking: 1.4204`
- `Episode_Termination/object_reached_goal: 0.0000`

현재 판단은 다음과 같다.

- 집기·들기 학습 신호는 살아났다.
- 배치 성공은 아직 확인되지 않았다.
- 현재 최적 체크포인트는 “들기 검증용 중간 모델”이며 완전한 PAP 모델은 아니다.
- 이 수정본으로 5000회 반복 장기 학습을 다시 수행해야 한다. 현재 속도 기준 예상 시간은 약 2시간 전후다.

## 7. 실행 명령

화면 없는 학습:

```bash
source /home/mh/isaac-stack-2.3.2/activate_isaaclab_2.3.2.sh
cd /path/to/OpenArm
python scripts/rsl_rl/train.py \
    --task Isaac-PickPlace-OpenArm-TestUSD-Bi-v0 \
    --num_envs 1024 \
    --device cuda \
    --max_iterations 5000 \
    --headless
```

GUI 확인:

```bash
source /home/mh/isaac-stack-2.3.2/activate_isaaclab_2.3.2.sh
cd /path/to/OpenArm
python scripts/rsl_rl/train.py \
    --task Isaac-PickPlace-OpenArm-TestUSD-Bi-v0 \
    --num_envs 16 \
    --device cuda \
    --max_iterations 200
```

체크포인트 실행:

```bash
source /home/mh/isaac-stack-2.3.2/activate_isaaclab_2.3.2.sh
cd /path/to/OpenArm
python scripts/rsl_rl/play.py \
    --task Isaac-PickPlace-OpenArm-TestUSD-Bi-Play-v0 \
    --num_envs 16 \
    --device cuda \
    --checkpoint logs/rsl_rl/openarm_testusd_bi_pick_place/<run>/model_<N>.pt \
    --real-time
```

## 8. 첨부 이미지

오늘 캡처한 TensorBoard 이미지를 함께 첨부했다.

### OpenArm 양팔 도달 학습 결과

![OpenArm 양팔 도달 학습 TensorBoard 평균 보상](screenshots/2026-07-07_reach_bimanual_tensorboard_mean_reward.png)

### Isaac 큐브 들기 비교 그래프

![Isaac 큐브 들기 TensorBoard 평균 보상](screenshots/2026-07-07_lift_cube_tensorboard_mean_reward.png)

### OpenArm TestUSD 집기·놓기 학습 결과

![OpenArm TestUSD 집기·놓기 TensorBoard 평균 보상](screenshots/2026-07-07_pick_place_tensorboard_mean_reward.png)

## 9. GitHub에 포함하지 않은 것

다음 항목은 저장소에 넣지 않았다.

- Isaac Lab 원본 소스 수정분
- Isaac Sim 설치 파일
- ROS2 작업 공간
- 학습 로그 전체
- TensorBoard 이벤트 파일
- 체크포인트 `.pt` 파일
- Python 캐시 파일
- 임시 검사 스크립트
