# 2026-07-07 OpenArm 작업 정리

이 문서는 2026년 7월 7일 OpenArm Isaac Lab 작업 중 GitHub에 반영한 핵심 변경만 정리한다.

## 1. USD 기본 장면 정리

- 로컬에서 최종 확정한 `openarm_test_v2.usd`를 GitHub의 기본 USD 파일인 `assets/openarm_test.usd`로 반영했다.
- GitHub에는 `openarm_test_v2.usd`라는 별도 이름을 추가하지 않았다.
- 즉, 저장소 기준 기본 장면은 계속 `assets/openarm_test.usd`다.
- 장면 구성은 로봇과 작업 테이블을 분리한 형태로 정리했다.
- 로봇은 낮은 받침 테이블 위에 위치하고, 바스켓이 있는 높은 테이블은 로봇 앞쪽 중앙에 배치했다.
- 높은 테이블은 바스켓과 물체 배치 공간을 확보하도록 가로 폭을 넓혔다.
- 로봇 색상과 로봇 asset 자체는 task 코드에서 수정하지 않았다.

## 2. Reach 튜토리얼 확인

- `Isaac-Reach-OpenArm-Bi-v0`는 pick-and-place가 아니라 양팔 end-effector가 목표 pose를 따라가도록 학습하는 reach task다.
- 이 task는 OpenArm bimanual articulation, joint action, reward 구성이 정상인지 확인하는 선행 튜토리얼로 사용했다.
- 5000 iteration 학습 후 TensorBoard에서 최종 mean reward가 약 `-0.03` 수준으로 수렴하는 것을 확인했다.
- `Isaac-Lift-Cube-OpenArm-v0`와 reward scale이 다르므로 두 그래프의 절대 reward 값을 직접 비교하면 안 된다.

## 3. Pick-and-place task 추가

새로운 task를 추가했다.

- `Isaac-PickPlace-OpenArm-TestUSD-Bi-v0`
- `Isaac-PickPlace-OpenArm-TestUSD-Bi-Play-v0`

추가된 주요 파일은 다음과 같다.

- `source/openarm_testusd_task/openarm_testusd_task/tasks/testusd_pick_place/__init__.py`
- `source/openarm_testusd_task/openarm_testusd_task/tasks/testusd_pick_place/testusd_pick_place_env_cfg.py`
- `source/openarm_testusd_task/openarm_testusd_task/tasks/testusd_pick_place/mdp.py`
- `source/openarm_testusd_task/openarm_testusd_task/tasks/testusd_pick_place/agents/rsl_rl_ppo_cfg.py`
- `source/openarm_testusd_task/openarm_testusd_task/tasks/__init__.py`

현재 PAP task는 오른손 중심의 첫 학습 버전이다.

- 양팔 joint position action을 사용한다.
- 오른손 gripper만 binary open/close action으로 제어한다.
- 왼손 gripper는 현재 PAP reward에 직접 연결하지 않았다.
- 학습 물체는 안정성 확인을 위해 큐브 1종으로 제한했다.

## 4. Reward와 종료 조건

Reward는 Isaac Lab lift MDP를 기반으로 구성했다.

- 오른손과 물체 사이 거리 보상
- 물체 lifting 보상
- 물체가 목표 위치에 가까워지는 goal tracking 보상
- action 변화량 penalty
- joint velocity penalty

종료 조건은 다음과 같다.

- episode timeout
- 물체가 너무 낮게 떨어지는 경우
- 물체가 목표 위치에 들어온 경우

기존 lift MDP의 end-effector 거리 계산은 현재 USD 구조와 맞지 않는 부분이 있어, `mdp.py`에 `object_body_distance()`를 추가해 `openarm_right_hand` body pose를 직접 읽도록 했다.

## 5. PPO 안정화

초기 학습 중 `value_function loss: inf`와 `normal expects all elements of std >= 0.0` 오류가 발생했다.

이를 막기 위해 PPO 설정을 안정화했다.

- empirical normalization 활성화
- action noise를 log std 방식으로 변경
- learning rate를 `3e-5`로 낮춤
- fixed schedule 사용
- entropy coefficient 축소
- mini-batch와 epoch 설정 조정

이후 100 iteration headless 검증은 정상 종료했다.

## 6. 현재 학습 상태

PAP task의 학습 파이프라인은 정상 동작한다.

다만 5000 iteration 결과는 아직 pick-and-place 성공으로 보기 어렵다.

- 평균 reward는 약 `0.13` 수준으로 안정화됐다.
- `reaching_object` 보상은 생긴다.
- `lifting_object`, `object_goal_tracking`, `object_reached_goal`은 아직 0에 가깝다.

즉, 현재 상태는 “오른손이 물체에 접근하는 초기 정책은 잡히지만, 집기와 운반은 아직 보상 설계와 초기 자세를 더 다듬어야 하는 단계”다.

## 7. 실행 명령

Headless 학습:

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

Checkpoint 실행:

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

### OpenArm bimanual reach 학습 결과

![OpenArm bimanual reach TensorBoard mean reward](screenshots/2026-07-07_reach_bimanual_tensorboard_mean_reward.png)

### Isaac Lift Cube 비교 그래프

![Isaac Lift Cube TensorBoard mean reward](screenshots/2026-07-07_lift_cube_tensorboard_mean_reward.png)

### OpenArm TestUSD pick-and-place 학습 결과

![OpenArm TestUSD pick-and-place TensorBoard mean reward](screenshots/2026-07-07_pick_place_tensorboard_mean_reward.png)

## 9. GitHub에 포함하지 않은 것

다음 항목은 저장소에 넣지 않았다.

- Isaac Lab 원본 소스 수정분
- Isaac Sim 설치 파일
- ROS2 workspace
- 학습 로그 전체
- TensorBoard event 파일
- checkpoint `.pt` 파일
- Python cache 파일
- 임시 검사 스크립트

