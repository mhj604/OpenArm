# ROMO-B/OpenArm Isaac Sim 구현 명세

## 결론

이 구현은 제공된 ROMO-B 매뉴얼의 숫자를 그대로 보존하고, 매뉴얼에 없는 항목을 별도
표시한 **manual-constrained simulation proxy**다. 따라서 “매뉴얼에 적힌 값이 정확히
들어간 모델”이라고는 할 수 있지만, 실측하지 않은 질량중심·관성·타이어·현가까지 정확한
디지털 트윈이라고 주장해서는 안 된다.

OpenArm 자체는 NVIDIA Isaac Sim 5.1의 bimanual USD를 payload로 사용한다. 팔의 링크,
관절, collision, 질량 및 관성은 이 공식 시뮬레이션 자산에 맡기고 ROMO-B 매뉴얼에서
임의로 재구성하지 않았다.

## 모델에 반영한 ROMO-B 값

| 항목 | 적용값 | 근거 상태 |
|---|---:|---|
| 구조 | 4WID-4WIS | 매뉴얼 명시 |
| 전체 길이 | 0.756 m | 매뉴얼 명시 |
| 전체 너비 | 0.597 m | 매뉴얼 명시 |
| 전체 높이 | 0.342 m | 매뉴얼 명시, 높이 기준면은 불명확 |
| 지상고 | 0.062 m | 매뉴얼 명시 |
| 축거 | 0.323 m | 매뉴얼 권장값; 다른 페이지의 0.324 m도 설정에 보존 |
| 일반 사양 윤거 | 0.529 m | 매뉴얼 명시, 실제 wheel centre 여부는 실측 필요 |
| 조향 너클 중심 간격 | 0.390 m | Ackermann 계산용 매뉴얼 명시값 |
| 휠 지름 | 약 0.206 m | 매뉴얼 근삿값 |
| 플랫폼 총질량 | 55.0 kg | 매뉴얼 명시, 질량 분포는 불명 |
| 허용 payload | 100.0 kg | 매뉴얼 명시, 위치·CoM 조건은 불명 |
| 최고 플랫폼 속도 | 1.5 m/s | 매뉴얼 명시 |
| 휠 모터 정격 회전속도 | 200 rpm | 매뉴얼 명시 |
| 휠 모터 최대 회전속도 | 300 rpm | 매뉴얼 명시 |
| 개별 조향 한계 | ±30° | 매뉴얼 명시 |
| 2WIS 등가 명령 한계 | ±22° | 매뉴얼 명시 |
| 4WIS 등가 명령 한계 | ±18° | 매뉴얼 명시 |

`0.529 m`와 `0.390 m`는 서로 바꾸어 쓰지 않았다. 전자는 휠 시각 중심의 조건부 윤거,
후자는 조향축 배치에 각각 사용했다. 매뉴얼의 `323 mm`를 기준으로 조향축 X 좌표는
`±0.1615 m`, `390 mm`를 기준으로 Y 좌표는 `±0.195 m`다. 휠 중심 Y 좌표는 529 mm를
윤거로 해석한 `±0.2645 m`지만, 매뉴얼 자체가 실제 휠 중심을 실측하도록 요구하므로
`derived_from_unverified_wheel_track_interpretation`으로 표시했다.

## 조향 명령 부호와 USD 관절 부호

매뉴얼 조향 명령은 **음수=좌조향, 양수=우조향**이다. 반면 이 모델의 +X 전방,
+Y 좌측, +Z 상방 좌표계에서 오른손 법칙을 따르는 USD +Z revolute joint의 양의
회전은 +X를 +Y 쪽, 즉 좌측으로 돌린다. 두 부호가 반대이므로 명령 변환은 다음과
같다.

```text
usd_target_position_deg = manual_steering_command_deg * (-1)
```

예를 들어 매뉴얼 명령 `-30°` 좌조향은 USD target `+30°`, 매뉴얼 명령 `+30°`
우조향은 USD target `-30°`가 된다. 이 계수는
`romo_b.steering_command_convention.manual_to_usd_joint_multiplier=-1`로 보존한다.
이는 매뉴얼의 명령 부호와 선언한 좌표계 사이의 변환 계약이며, 실제 휠 피드백 부호와
FL/FR/RL/RR 채널 대응을 실물에서 확인했다는 뜻은 아니다.

## OpenArm 장착 위치

좌표계는 차체 기하 중심의 지면 투영점을 원점으로 하며 +X 전방, +Y 좌측, +Z 상방이다.
현재 적용 pose는 다음과 같다.

```text
translation = (0.170, 0.000, 0.342) m
rotation    = identity
```

- `x=+0.170 m`: 사용자가 말한 “앞쪽 약 17 cm”를 **차체 중심 기준 전방 오프셋**으로 해석했다.
- `z=0.342 m`: 매뉴얼의 전체 높이를 상판 장착면으로 사용한 배치 가정이다.
- 만약 17 cm가 전면 끝에서 뒤로 잰 거리라면 `x=0.756/2-0.170=+0.208 m`가 맞다.

17 cm의 기준점과 브래킷 높이는 매뉴얼에 없다. 충돌·전복·하중이동 실험 전에 실제
base flange의 `xyz/rpy`를 측정하여 JSON의 `openarm_mount`를 갱신해야 한다.

## 물리 구조

ROMO-B는 다음 순서의 단일 floating articulation으로 구성했다.

```text
ROMOBOpenArm (ArticulationRootAPI)
├── Chassis (rigid body)
├── FrontLeft/Right, RearLeft/Right
│   ├── Knuckle (rigid body, chassis와 Z축 steering joint)
│   └── Wheel (rigid body, knuckle과 Y축 roll joint)
└── OpenArm payload
    └── openarm_body_link (chassis와 fixed joint)
```

NVIDIA OpenArm 자산의 원래 `root_joint`는 팔을 월드에 고정하므로 override에서
비활성화했다. 그 대신 `openarm_mount_joint`가 ROMO-B chassis와
`OpenArm/openarm_body_link`를 연결한다. 따라서 AMR가 움직일 때 팔도 함께 움직이며,
별도의 nested articulation으로 남지 않는다.

원래 OpenArm `root_joint`에 있던 PhysX 설정도 결합 root로 옮겼다. self-collision은
`false`, solver position iteration은 32, velocity iteration은 1이다. root joint만 끄고
이 값을 버릴 때 생기는 팔 안정성 차이를 방지한다. 새 fixed joint에는
`excludeFromArticulation=false`를 명시했다. 여기서 `self-collision=false`는 OpenArm 내부에만
한정되지 않고 **결합 articulation 전체**에 적용되므로 팔↔AMR 및 ROMO-B 자체 링크 사이의
self-collision도 자동 검사하지 않는다. 실제 간섭·안전 검증용 모델에서는 collision filtering과
self-collision 정책을 별도로 설계하고 검증해야 한다.

플랫폼 총질량 55 kg을 중복 계상하지 않기 위해 시뮬레이션 질량을 다음처럼 배분했다.

- 휠/인휠 모터 링크: 매뉴얼 근삿값 4.000 kg × 4
- 조향 knuckle/servo 링크: 매뉴얼 근삿값 0.765 kg × 4
- chassis 잔여 질량: `55 - 4 × (4 + 0.765) = 35.940 kg`

합계는 정확히 55.000 kg이지만 **실측 질량 분포는 아니다**. OpenArm 질량은 NVIDIA
payload의 링크별 질량을 그대로 사용하며 ROMO-B의 55 kg에 포함하지 않는다.

## 의도적으로 확정하지 않은 값

매뉴얼에 다음 정보가 없으므로 임의의 “정확값”을 만들지 않았다.

- Dual A-arm 현가 링크 위치, travel, 강성 및 감쇠
- 플랫폼 CoM 및 링크별 관성 텐서
- 휠 실제 폭, 타이어 마찰·강성·감쇠
- caster, camber, trail, steering-axis inclination
- 상판 홀 패턴과 OpenArm 브래킷 치수
- LiDAR·카메라·IMU·GNSS의 정확한 pose
- 실물 모터 배선과 FL/FR/RL/RR 채널 대응
- 조향 position drive와 휠 velocity drive의 제어 gain

chassis 폭 0.450 m와 휠 폭 0.068 m는 충돌 형상을 만들기 위한 proxy다. 특히 휠 폭은
`0.597-0.529`에서 유도한 값이지 실측값이 아니다. 조향 knuckle body의
`0.045 × 0.045 × 0.070 m` 크기(X×Y×Z)도 매뉴얼이나 실측에서 얻은 값이 아니라,
형상을 보이기 위한 **unverified simulation proxy**다. 현재 builder에서는
knuckle의 시각 형상과 collision 치수에 함께 사용한다. 이 값은
`simulation_proxy.knuckle_body_size_m`에 보존하며 clearance, 간섭, 관성 또는 제작
근거로 사용해서는 안 된다. 이 항목들은 모두 JSON에 `derived` 또는 `unverified`로
표시되어 있다.

조향축이 무부하 상태에서 쓰러지지 않고 휠 velocity target이 작동하도록 baseline drive
gain을 넣었지만, 이 gain은 매뉴얼 값이 아니다. 실제 step response를 측정하기 전에는
동역학 결과의 근거로 사용하지 않는다.

플랫폼 최고 속도 `1.5 m/s`는 휠 반지름 `0.103 m`를 사용해
`(1.5 / 0.103) rad/s = 약 834.4 deg/s`로 환산하고, 휠 revolute joint의
PhysX `maxJointVelocity`로 적용한다. 따라서 휠 관절 속도는 제한하지만, 슬립이
생길 때 차체의 선속도를 `1.5 m/s`로 직접 clamp하는 규칙은 아니다. 조향
서보의 `0.37 s/60°`도 `60 / 0.37 = 약 162.2 deg/s`로 환산해 조향
관절에 같은 방식으로 적용한다. 반면 휠 모터 `200/300 rpm`은 부품 사양
provenance를 보존하는 custom metadata이며, 현재 런타임 제한값으로 쓰지 않는다.
제어기 명령 생성단에서도 같은 한계로 clamp해 과도한 target을 보내지 않아야 한다.

## 재현과 검증

수치와 근거 분류는 `simulation/config/romo_b_openarm_spec.json` 한 곳에서 관리한다.
USD는 `build_romo_b_openarm_usda.py`로 재생성하고, 다음 세 단계로 검증한다.

1. 표준 Python 계약 테스트: 원문 보존, 치수, 파생식, 질량 합계, 17 cm 해석
2. pxr 정적 검사: stage 단위, prim, articulation, joint 축·한계·body 관계, payload와 mount
3. Isaac Sim smoke test: OpenArm payload 해석 후 120 physics step 동안 stage 유지

정적 검사는 payload를 불러오지 않아 오프라인으로 수행할 수 있다. payload 해석 검사와
smoke test에는 Isaac Sim 5.1 자산 캐시 또는 네트워크가 필요하다.
