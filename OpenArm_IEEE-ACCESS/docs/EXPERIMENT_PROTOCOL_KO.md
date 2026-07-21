# KALO-H 하드웨어 실험 프로토콜

이 프로토콜은 Pivot의 구조적 조향 불일치와 2WIS의 조건 의존 궤적 residual을 분리하고,
OpenArm 양팔이 실제로 바꿀 수 있는 authority만 검증하기 위한 최소 설계입니다.

## 1. 안전 선행조건

실험 전에 다음이 문서로 확인되어야 합니다.

- ROMO-B 100 kg payload 정격 안에서 실제 총 탑재질량
- OpenArm pedestal, 프레임, 체결부의 정적·동적 허용하중
- arms의 허용 속도·가속도·토크·온도
- payload 고정과 낙하 방지
- 독립 E-stop, 경계구역, tether 또는 물리적 보호장치
- 제조사 승인 또는 내부 안전검토를 통과한 속도·가속도 범위

임의의 torque/temperature 비율을 안전기준으로 만들지 않습니다. wheel unloading,
mount loosening, resonance, payload displacement, CAN/PCU fault가 나타나면 즉시 중단합니다.

## 2. 필수 계측

### 2.1 최소 장비

- 외부 motion capture 또는 동급 6-DoF ground truth
- 고속 chassis IMU
- RS-232 raw command/feedback bytes와 host timestamps
- 양팔 CAN/상태 로그
- 네 wheel steering feedback과 RPM/speed feedback
- component mass, 3D CoM, wheel radius, steering-axis geometry
- 네 모서리 정적 scale

### 2.2 강하게 권장되는 장비

- 네 바퀴 synchronized dynamic normal-load sensors
- 또는 네 접촉점을 동시에 포착하는 force-plate 구성
- actual wheel-plane angle 광학 측정
- hardware timestamp 또는 공통 trigger

20 Hz RS-232만으로 tire traction force, lateral force, 고주파 slip, normal force,
마찰계수, arm-induced ground impulse 또는 직접 dynamic stability margin을 주장할 수 없습니다.

## 3. 좌표계와 동기화

다음을 trial manifest에 고정합니다.

- world, chassis, steering-axis, wheel, arm-base, end-effector frame
- 좌·우·전·후 부호와 steering sign
- yaw 및 CW/CCW 부호
- wheel radius와 RPM sign
- 각 장치 clock offset, drift, interpolation 방법
- 데이터 exclusion과 abort rule

모션캡처·IMU·RS-232·CAN의 동기 오차가 50 ms cycle과 비슷하면 transient mechanism을
분리할 수 없습니다. 저주파 분석이라도 synchronization uncertainty를 prediction interval에
포함합니다.

## 4. 첫 주 falsification plan

| 일자 | 작업 | 통과 기준 또는 실패 의미 |
|---|---|---|
| Day 1 | steering-axis, wheel plane, track, wheel radius 광학 측정 | 매뉴얼 기하 사용 가능성 판정 |
| Day 2 | 2WIS/4WIS/Pivot actual angle, latency, hysteresis 측정 | steering realization model 확보 |
| Day 3 | arm 고정, 매우 저속 Pivot 반복 | ICR gap과 geometry floor 검증 |
| Day 4 | arm 고정, 2WIS 일정 곡률과 두 acceleration profile | rear-slip 가설의 기초 증거 |
| Day 5 | arm posture별 정적 corner load와 전체 CoM | arm load-shift authority 확인 |
| Day 6 | 동일 packet에서 fixed/naive/gated 저속 matched trial | 효과 부호와 크기 예비확인 |
| Day 7 | session-blocked model과 Go/No-Go memo | 논문 claim 동결 |

## 5. 실험 조건

### 5.1 Payload 조건

최종 수치는 안전검토 후 정합니다.

| 조건 | 설계 목적 |
|---|---|
| P0: nominal front-heavy installation | 실제 KALO-H 기준 |
| P1: 같은 위치, 다른 질량 | mass-ratio 일반화 |
| P2: 비슷한 질량, 다른 전후 offset | eccentricity 일반화 |
| P3: 비슷한 질량·offset, 다른 높이 | stability sensitivity; 완전 held-out 권장 |

### 5.2 Surface 조건

- S0: 실험실 기준 바닥
- S1: 안전하게 고정된 다른 마찰 특성 표면

마찰계수를 직접 측정하지 않으면 “high/low friction coefficient”라고 쓰지 않고 surface
condition으로만 보고합니다. surface variation은 first-week pilot 이후 안전할 때만 추가합니다.

### 5.3 Base profile

2WIS primary:

- 일정 곡률 원호
- 저속과 중속
- 완만한 acceleration, 강한 acceleration, braking profile
- 동일 scalar SPEED/STEER packet을 비교조건에서 재사용

Pivot negative control:

- 매우 낮은 yaw command부터 단계적 증가
- CW/CCW 균형
- 같은 packet duration과 acceleration ramp
- 구조 잔차를 제거하는 성공 사례가 아니라 residual redistribution 확인

4WIS는 9월 원고에서는 exploratory 또는 제외합니다.

## 6. Arm 조건과 비교군

### 6.1 2WIS primary comparison

| 기호 | 조건 | 분리되는 효과 |
|---|---|---|
| H | picking pose에서 hold | arm intervention 없음 |
| Q | best fixed low/rear compensated posture | quasi-static load shift |
| N | 동일 picking-to-stow path, nominal timing | naive concurrent |
| A | N과 동일 path, authority-gated timing | 제안법 |
| O | offline oracle timing | achievable upper reference |

Primary comparison은 A 대 N입니다. 동일한 arm 시작·종료 자세, geometric path와 base packet을
사용합니다. A 대 H만 비교하면 concurrency, posture 및 timing 효과가 섞입니다.

### 6.2 반드시 포함할 비-arm baseline

- scalar speed derating
- acceleration ramp 완화
- scalar curvature precompensation
- Pivot 대신 4WIS multi-arc 또는 2WIS multi-point maneuver

이 기준들이 arm intervention보다 우수하면 “왜 팔을 써야 하는가”에 답할 수 없습니다.

### 6.3 Ablation

1. posture-only: 팔을 옮긴 뒤 정지
2. slow-versus-fast: 같은 path를 다른 duration으로 실행
3. phase shift: path·duration은 같고 base 대비 start offset만 변경
4. common-versus-differential arm motion: aggregate CoM과 yaw reaction 분리
5. uncertainty gate off: point estimate scheduler 비교
6. timing shuffled: 세션 drift나 우연 배제

## 7. 평가량

### 7.1 Wheel residual

각 휠에 대해 다음 시계열을 산출합니다.

- longitudinal residual \(e_{\parallel,i}\)
- normalized slip-ratio proxy \(\kappa_i\)
- lateral rolling residual \(e_{\perp,i}\)
- kinematic slip-angle proxy \(\alpha_i\)
- measured steering command/feedback/optical angle difference

### 7.2 Geometry와 excess 분해

\[
J_{\mathrm{geom}}=\min_{v_x,v_y}\sum_i e_{\perp,i}^2,
\qquad
J_{\mathrm{excess}}=\sum_i e_{\perp,i}^2-J_{\mathrm{geom}}
\]

시간적 primary는 전체 arm-return cycle에서 적분한 \(J_{\mathrm{excess}}\)입니다.

### 7.3 Effective ICR

\(|\dot\psi|>\epsilon_\psi\)에서 모션캡처 twist에 의한 effective ICR은

\[
p_{\mathrm{ICR}}=
\begin{bmatrix}
-v_y/\dot\psi\\
v_x/\dot\psi
\end{bmatrix}
\]

입니다. Pivot 평가량은 desired center와의 RMS 거리와 최종 translation drift입니다.

### 7.4 Trajectory

- cross-track error \(e_y\)
- heading error \(e_\psi\)
- actual curvature \(\kappa_{actual}=\dot\psi/\sqrt{v_x^2+v_y^2}\)
- final position/yaw error

### 7.5 CoM과 stability

부품 질량과 arm state로 전체 CoM을

\[
r_G(q_L,q_R)=\frac{\sum_jm_jr_j(q_L,q_R)}{\sum_jm_j}
\]

로 계산합니다. direct dynamic corner load가 있으면 CoP와 최소 wheel load를 보고합니다.

\[
r_{CoP}=\frac{\sum_iF_{z,i}r_i}{\sum_iF_{z,i}},
\qquad
F_{z,min}=\min_iF_{z,i}
\]

normal-load sensor만으로 longitudinal traction force를 측정했다고 주장하지 않습니다.

## 8. Session·holdout·통계

### 8.1 Identification

- 독립 3 sessions
- P0-P2
- 2WIS와 Pivot
- arm condition당 최소 4-5 randomized matched repetitions

### 8.2 Confirmatory holdout

- 독립 2 sessions
- held-out P3
- 2WIS primary
- A/N/Q 조건
- profile당 최소 8 matched blocks
- 최종 표본수는 pilot paired variance와 practical effect로 재계산

권장 mixed-effects 구조:

- session random intercept
- payload/condition random slope
- 사전등록된 primary A-N contrast
- effect estimate, one-sided confidence bound, prediction interval과 empirical coverage

## 9. Go/No-Go 기준

### 9.1 Geometry gate

- \(|\theta^*-\theta_{pivot}|>3\sigma_\theta\)
- \(\sqrt{J_{geom}}>3\sigma_{e_\perp}\)
- optical angle에서도 front/rear virtual ICR non-concurrence 확인

실패하면 structural Pivot residual 주장을 중단합니다.

### 9.2 2WIS diagnosis gate

command+steering-only model에 wheel residual과 payload/load 변수를 추가했을 때 held-out
trajectory error가 \(\max(10\%,3\sigma_{CV})\) 이상 줄어야 합니다.

실패하면 rear-slip decomposition은 과잉 모델링입니다.

### 9.3 Arm-authority gate

- discovery 3/3 sessions에서 같은 효과 부호
- held-out one-sided 95% lower confidence bound > 0
- improvement > \(\max(10\%,3\sigma_{repeat})\)
- support/stability endpoint 악화 없음
- full arm-return payback 포함

### 9.4 Prediction gate

- held-out payload에서 effect sign 유지
- useful/negligible/harmful class 정확도 목표 80% 이상
- false-safe 0회
- 사전 목표에 맞는 prediction interval coverage

## 10. 결과별 논문 전환 규칙

| 결과 | 해석과 다음 행동 |
|---|---|
| geometry floor와 arm excess reduction 모두 확인 | 권장 IEEE Access 원고 진행 |
| geometry floor만 확인 | platform diagnosis; arm claim 삭제 |
| arm effect가 speed derating보다 약함 | mode/maneuver supervisor로 전환 |
| chassis transient만 있고 whole-system gain 없음 | transient-to-task conversion failure 논문 검토 |
| arm effect가 noise 이하 | positive improvement claim 중단 |
| 세션 간 효과 부호 변경 | 불확실성 원인 식별 전 제출 중단 |

Null result를 제출하려면 단순 \(p>0.05\)가 아니라 smallest effect of interest에 대한
equivalence/upper bound, 충분한 excitation, 여러 payload/session과 공개 raw data가 필요합니다.
