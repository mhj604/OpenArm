# ROMO-B Ackermann/Pivot 오차와 양팔 하중이동 연구의 엄격한 분석

기준일: 2026-07-21
목표 저널: IEEE Access
목표 제출일: 2026-09-15

## 1. 분석 대상 질문

> 그리고 또한 중요한 점은 애커먼 모드의 경우 뒷바퀴 슬립으로 인해 실제 하드웨어
> 수치(조향각, 페이로드, 가속도 등)의 수치적인 궤적 계산과는 다른 궤적으로 가고,
> 피벗 모드 또한 가장 이상적인 조향각은 4바퀴 모두 45도 인데 ROMO-B의 조향각을
> 보면 3n도 정도로 알고 있기 때문에 정확한 원을 그리지 못하는 문제가 있어.
> 그렇기에 필요한 연구인 것 같은데 웹 ChatGPT Pro와 엄격하게 분석해줘.

이 문서는 위 질문에서 출발한 수치 검증, 물리적 반증 가능성, 선행연구 충돌, 실험 설계,
저널 적합성 판단을 하나의 재현 가능한 기록으로 정리합니다.

## 2. 최종 판정

연구 필요성은 분명합니다. 그러나 다음 세 문장은 서로 다른 수준의 주장입니다.

1. ROMO-B의 실제 궤적이 무슬립 기구학 예측과 다르다: 관측 가능하고 개연성이 높음.
2. 그 원인이 2WIS 후륜 슬립이다: 현재는 검증되지 않은 가설.
3. 양팔을 움직이면 Pivot 기하 오차가 해결된다: 물리적으로 틀림.

IEEE Access에서 방어 가능한 과학적 질문은 다음입니다.

> 폐쇄형 4WID-4WIS 베이스에서 조향 한계가 만드는 제거 불가능한 구조적 rolling
> residual과 payload, 하중이동, 마찰 및 embedded PCU가 만드는 조건 의존 residual을
> 계측적으로 분리하고, 양팔이 후자에 제공하는 authority의 불확실성 경계를 식별할 수 있는가?

이 분해와 held-out 검증이 없으면 연구는 특정 ROMO-B의 calibration exercise입니다.

## 3. 매뉴얼에서 확정되는 사실

근거는 [ROMO-B 전체 압축본](../source_files/hardware/ROMO-B_manual_complete_compressed.pdf)과
[핵심 페이지 이미지](../evidence/manual_pages/README.md)에 보존되어 있습니다.

| 항목 | 매뉴얼 값 | 근거 페이지 |
|---|---:|---:|
| 구동 구조 | 4WID-4WIS | 1 |
| 플랫폼 질량 | 55 kg | 4 |
| 정격 payload | 100 kg | 4 |
| 축거 | 0.323 m 기본값(6회); 0.324 m 원문 이상치(1회) | 2, 3, 5, 26, 35 |
| 실제 윤거 | 0.529 m | 3, 26 |
| 조향 너클 회전중심 간격 | 0.390 m | 3, 26 |
| 개별 조향 한계 | -30° to +30° | 3 |
| 2WIS scalar command 범위 | -22° to +22° | 3, 36 |
| 4WIS scalar command 범위 | -18° to +18° | 3, 36 |
| Pivot 조향 패턴 | FL/RR +30°, FR/RL -30° | 3, 31 |
| RS-232 cycle | 50 ms, 20 Hz | 38 |
| 외부 명령 | scalar SPEED, STEER, MODE 등 | 36, 38 |
| 휠별 값 | 속도·조향각 feedback | 37, 38 |
| 피드백 표기 오차 | 속도 ±5%, 조향 ±1° | 38 |

외부 인터페이스에는 문서화된 휠별 속도·조향·토크 명령, 휠 전류, 수직하중,
접지력 또는 time-tagged execution이 없습니다. 따라서 새 wheel-level allocation 또는
직접 4WIS 제어를 제안했다고 주장할 수 없습니다.

## 4. Pivot 기하의 독립 검산

### 4.1 45°와 39.63°

매뉴얼 31쪽은 중심 Pivot의 이상각을 ±45°라고 적습니다. 하지만 같은 매뉴얼의
축거와 조향 너클 간격을 적용하면 정확한 각도는 다릅니다.

차체 중심을 원점으로 하고 조향축 좌표를

\[
r_i=(\pm L/2,\pm W_k/2)
\]

로 놓으면 중심 순수회전에서 wheel plane은 조향축 위치의 접선속도와 나란해야 합니다.
필요 조향각의 크기는

\[
\theta^*=\tan^{-1}\left(\frac{L}{W_k}\right)
=\tan^{-1}\left(\frac{0.323}{0.390}\right)
=39.6317^\circ
\]

입니다. 45°는 \(L=W_k\)인 정사각형 배치에서만 성립합니다. 따라서 매뉴얼의 45°는
단순화되었거나 잘못된 설명이고, ROMO-B의 실제 고정 ±30°와의 차이는

\[
\Delta\theta=39.6317^\circ-30^\circ=9.6317^\circ
\]

입니다.

여기서는 PDF 전체에서 `323 mm` 또는 `0.323 m`가 6회(R2 1회, R5 1회,
R26 3회, R35 1회), `324 mm` 또는 `0.324 m`가 1회(R3) 나타나는 점을 근거로
`L=0.323 m`를 계산·시뮬레이션 기본값으로 확정했다. `0.324 m`는 원문 보존과
실측 대조를 위한 고립 이상치로만 남긴다.

### 4.2 공통 ICR 부재

대칭 ±\(\theta\) cross steering에서 앞바퀴 쌍과 뒷바퀴 쌍의 가상 ICR 종방향 위치는

\[
x_F=\frac L2-\frac{W_k}{2}\tan\theta,
\qquad
x_R=-\frac L2+\frac{W_k}{2}\tan\theta
\]

입니다. \(\theta=30^\circ\)일 때

\[
x_F=+0.04892\ \mathrm m,
\qquad
x_R=-0.04892\ \mathrm m
\]

로서 두 가상 ICR은 97.83 mm 떨어집니다. 따라서 비영 yaw rate에서 네 휠이 동시에
zero-slip 조건을 만족하는 단일 rigid-body ICR은 존재하지 않습니다.

실제 장비는 다음 현상으로 불일치를 분배합니다.

- tire lateral deformation과 scrub
- 휠별 longitudinal/lateral slip
- steering compliance
- 차체 translation과 effective ICR drift
- yaw-rate attenuation
- 숨겨진 PCU wheel-speed allocation
- 하중이 작은 바퀴로 residual 집중

### 4.3 이론적 rolling-constraint residual

조향 너클 중심 반경은

\[
r_k=\sqrt{(L/2)^2+(W_k/2)^2}=0.25319\ \mathrm m
\]

입니다. 차체 중심 yaw를 강제할 때 휠당 횡방향 rolling-constraint residual의 크기는

\[
|e_{\perp,i}|=r_k|\dot\psi|\sin|\Delta\theta|
=0.04236|\dot\psi|\ \mathrm{m/s}
\]

입니다. 예를 들어 \(|\dot\psi|=0.5\ \mathrm{rad/s}\)이면 약 0.0212 m/s입니다.

이 수치는 측정된 tire slip ratio, lateral force 또는 scrub power가 아닙니다. 강제된
중심회전과 매뉴얼 기하 아래에서 예상되는 횡방향 속도 제약 잔차입니다.

### 4.4 왜 0.529 m가 아니라 0.390 m인가

매뉴얼은 조향각 계산에 실제 윤거 0.529 m가 아니라 조향 너클 회전중심 간격 0.390 m를
사용합니다. 일반적인 조향축-접촉점 횡방향 offset은 lateral rolling constraint에서
소거될 수 있으므로 0.390 m를 사용하는 것이 조건부로 타당합니다.

그러나 실제 steering-axis 위치, caster/trail, wheel-plane angle, 하중에 따른 변형은
확인되지 않았습니다. 첫 실험에서 광학적으로 측정해야 하며, 매뉴얼 숫자만으로 실제
타이어 접촉기하를 확정해서는 안 됩니다.

## 5. 2WIS/Ackermann의 후륜 슬립 진단

### 5.1 확정 사실과 가설

| 진술 | 상태 |
|---|---|
| 2WIS rear steer는 0° | 매뉴얼로 확정 |
| scalar steer가 앞바퀴 Ackermann-like 각으로 변환됨 | 매뉴얼로 확정 |
| PCU가 네 drive motor를 내부 제어함 | 매뉴얼로 확정 |
| 실제 경로가 이상 기구학 경로와 다름 | 외부 ground truth로 계량 필요 |
| rear lateral slip이 주요 원인 | 가설 |
| rear longitudinal slip이 주요 원인 | 가설 |
| payload가 corner load를 바꿈 | 높은 개연성, 동적 측정 필요 |
| hidden speed allocation이 경로 오차를 만듦 | 가설 |

경로 오차만 보고 rear-wheel slip을 결론내릴 수 없습니다. 조향 편차, latency,
hysteresis, 휠 반경, 타이어 변형, 바닥 마찰 및 hidden PCU saturation도 같은 현상을 만듭니다.

### 5.2 필요한 바퀴별 잔차

모션캡처와 IMU에서 body-frame twist를

\[
\xi=[v_x,v_y,\dot\psi]^\top
\]

로 얻습니다. 바퀴 \(i\)의 차체 좌표가 \(r_i=[x_i,y_i]^\top\)이면 wheel 위치의
rigid-body velocity는

\[
v_i=
\begin{bmatrix}
v_x-\dot\psi y_i\\
v_y+\dot\psi x_i
\end{bmatrix}
\]

입니다. 실제 조향각 \(\delta_i\)에 대해 rolling 방향과 횡방향을

\[
t_i=[\cos\delta_i,\sin\delta_i]^\top,
\qquad
n_i=[-\sin\delta_i,\cos\delta_i]^\top
\]

로 정의합니다.

휠 유효반경 \(R_i\), 각속도 \(\Omega_i\)에 대한 종방향 잔차와 slip-ratio proxy는

\[
e_{\parallel,i}=R_i\Omega_i-t_i^\top v_i
\]

\[
\kappa_i=
\frac{R_i\Omega_i-t_i^\top v_i}
{\max(|R_i\Omega_i|,|t_i^\top v_i|,v_\epsilon)}
\]

입니다. 횡방향 잔차와 kinematic slip-angle proxy는

\[
e_{\perp,i}=n_i^\top v_i
\]

\[
\alpha_i=\operatorname{atan2}(e_{\perp,i},t_i^\top v_i)
\]

입니다. 이는 tire force가 아니며 저속에서 정규화가 불안정합니다. 하중별 effective
rolling radius와 센서 지연을 보정해야 합니다.

rear residual이 front보다 크고 그 변화가 held-out trajectory error를 설명해야
“후륜 슬립이 주요 원인”이라고 주장할 수 있습니다.

## 6. 구조적 항과 조건 의존 항의 분해

측정 yaw rate와 실제 조향각을 고정했을 때 가능한 차체 translation 중 최소 횡잔차를

\[
J_{\mathrm{geom}}(\dot\psi,\delta)=
\min_{v_x,v_y}\sum_{i=1}^{4}w_i e_{\perp,i}^2
\]

로 정의합니다. 이는 steering geometry와 angle limit가 만드는 제거 불가능한 floor입니다.

실제 twist에서 측정되는 값은

\[
J_{\mathrm{kin}}=\sum_i e_{\perp,i}^2
\]

이고, 조건 의존 excess를

\[
J_{\mathrm{excess}}=J_{\mathrm{kin}}-J_{\mathrm{geom}}\ge0
\]

로 정의할 수 있습니다. 단, \(J_{\mathrm{excess}}\)에는 마찰만이 아니라 다음이 모두
섞이므로 곧바로 “friction residual”이라고 부르면 안 됩니다.

- hidden wheel-speed allocation
- payload와 load transfer
- surface friction과 tire compliance
- arm posture와 reaction
- PCU delay와 saturation
- unmodeled geometry와 timing error

nominal steering map과 실제 steering feedback/광학각의 차이는 별도의 steering-realization
residual로 분리해야 합니다.

## 7. 양팔이 바꿀 수 있는 것과 없는 것

양팔은 \(L\), \(W_k\), 조향 한계 및 네 wheel normal line의 비동시성을 바꾸지 못합니다.
따라서 Pivot의 구조적 \(J_{\mathrm{geom}}\)을 제거할 수 없습니다.

양팔이 바꿀 수 있는 항목:

- corner normal-load distribution
- 전체 CoM 위치·높이와 yaw inertia
- 순간 chassis reaction
- 실제 yaw rate와 least-squares ICR compromise
- 어떤 바퀴가 더 많은 scrub을 담당하는지
- support margin과 endpoint drift

전방 편심 설치물은 이상 Pivot 조향각을 바꾸지는 않지만, 플랫폼 중심 회전에서 전체
시스템 CoM이 원운동하게 만들어 추가 횡가속과 관성모멘트를 발생시킵니다. 팔을 후방·저상
자세로 옮기면 이 동적 부담을 줄일 가능성이 있으나 기하 불일치는 남습니다.

## 8. 선행연구 충돌과 남는 신규성

다음 영역은 이미 동료심사 문헌이 점유했습니다.

- mobile-manipulator tire-slip dynamics와 tracking control: Chung and Velinsky,
  [10.1017/S0263574798000873](https://doi.org/10.1017/S0263574798000873)
- 팔 자세로 wheel vertical force를 바꾸는 slip avoidance: Wang et al.,
  [10.1109/ROBIO.2011.6181521](https://doi.org/10.1109/ROBIO.2011.6181521)
- manipulator motion·reaction·joint acceleration의 slip analysis: Song et al.,
  [10.1115/1.4037287](https://doi.org/10.1115/1.4037287)
- 대곡률 4WIS ICR tracking: Chu et al.,
  [10.1109/TITS.2024.3432885](https://doi.org/10.1109/TITS.2024.3432885)
- dynamic weight shifting과 rollover prevention: Storms and Tilbury,
  [10.1115/1.4033841](https://doi.org/10.1115/1.4033841)
- arm posture/base speed 기반 tip-over avoidance: Ding et al.,
  [10.1109/ACCESS.2019.2915115](https://doi.org/10.1109/ACCESS.2019.2915115)

따라서 arm-based slip avoidance, traction improvement, weight shifting, ICR control 일반론,
4WIS path tracking 또는 tip-over avoidance 자체는 신규 기여가 아닙니다.

남는 좁지만 방어 가능한 기여는 다음입니다.

> 휠별 명령이 없는 20 Hz scalar interface에서 structural steering infeasibility와
> condition-dependent excess residual을 분리하고, 양팔이 바꿀 수 있는 closed-loop
> authority를 불확실성과 함께 held-out 조건에서 예측하는 방법.

관련 문헌 목록은 [SOURCES.md](../references/SOURCES.md)와
[references.bib](../references/references.bib)에 있습니다. arXiv-only 및 MDPI 문헌은
증거 기반에서 제외했습니다.

## 9. 논문 범위와 모드 역할

| 모드 | 논문 역할 |
|---|---|
| 2WIS/Ackermann | primary intervention mode |
| Pivot | structural negative control 및 falsification mode |
| 4WIS | 2026-09-15 원고에서는 제외 또는 exploratory |

2WIS는 이상 steering geometry가 대체로 feasible하므로 payload, speed allocation,
friction 및 arm effect를 검증하기 적합합니다. Pivot는 팔이 구조적 residual을 제거하지
못한다는 음성대조군입니다.

두 모드를 모두 “팔로 성능 개선” 사례로 쓰면 기하적 원인과 동적 원인이 섞여 논리가
무너집니다.

## 10. 권장 제목·주장·Primary endpoint

권장 제목:

> KALO-H: Separating Structural Steering Scrub from Bimanual Load-Transfer Authority
> in a Closed-Interface 4WID-4WIS Mobile Manipulator

한 문장 주장:

> 동일 scalar base packet에서 외부 운동, 휠 속도·조향 feedback, payload 질량특성과
> arm state를 동기화하여 조향 기하가 부과하는 제거 불가능한 rolling residual과 조건
> 의존 excess residual을 분리하고, 양팔 posture와 timing은 후자에 대해서만 적용한다.

Primary endpoint는 held-out 2WIS 조건에서

\[
\Delta J_{\mathrm{primary}}=
J_{\mathrm{excess}}^{\mathrm{Naive}}
-J_{\mathrm{excess}}^{\mathrm{Authority\text{-}gated}}
\]

로 둡니다. 전체 arm-return cycle을 포함하고, support margin의 non-inferiority를 동시에
만족해야 합니다. cross-track error, Pivot center drift, 작업 완료시간은 secondary입니다.

## 11. 실제 supervisor가 할 수 있는 조치

| 조치 | 가능 여부 | 비고 |
|---|---:|---|
| scalar speed derating | 가능 | yaw rate 감소로 구조 잔차 규모 감소 |
| acceleration ramp 완화 | 가능 | 20 Hz SPEED profile 사용 |
| mode selection | 가능 | 2WIS/4WIS/Pivot 선택 |
| arm posture·timing | 가능 | load distribution과 inertia 변경 |
| 2WIS/4WIS curvature precompensation | 조건부 | 반복 가능한 scalar mapping 필요 |
| Pivot 내부 curvature 변경 | 문서상 불가 | Pivot angle 고정 |
| per-wheel speed/steer/torque allocation | 문서상 불가 | 외부 명령 없음 |
| Pivot 대신 multi-arc maneuver | 가능, 강한 baseline | 팔 사용 필요성 검증에 필수 |

팔보다 speed derating이나 multi-arc가 우수하면, 논문의 중심은 arm scheduler가 아니라
mode/maneuver supervisor가 되어야 합니다.

## 12. 저널 판정

| 범위 | IEEE Access | RA-L | T-RO/IJRR |
|---|---|---|---|
| “팔로 slip·Pivot 개선” | Reject 가능성 높음 | Reject | Reject |
| geometry/excess residual 분해 + held-out authority | 조건부 Competitive | 현재 Reject | Reject |
| 다중 payload·직접 하중·cross-mode gray-box model | Competitive | 조건부 Competitive | Borderline 이하 |
| 일반 이론·보장·두 번째 플랫폼 | Competitive | Competitive | T-RO 검토 가능 |

Pivot의 9.63° 불일치는 논문을 자동으로 강하게 만드는 발견이 아닙니다. 팔로 고칠 수 없는
항과 고칠 가능성이 있는 항을 분리하도록 강제하는 falsification device입니다.

## 13. 적대적 Reviewer 2 관점

> ROMO-B의 고정 ±30° cross steering이 매뉴얼 기하에서 zero-slip centered Pivot과
> 양립하지 않는다는 관찰은 맞지만, 이는 특정 플랫폼의 조향 한계이지 새로운 모바일
> 매니퓰레이션 원리가 아니다. 팔은 네 wheel normal을 한 점에 모을 수 없으므로 구조적
> scrub을 제거하지 못한다. 직접 force 증거, 동일 base packet, held-out payload/session
> 예측 및 구조/제어 가능 residual의 사전등록된 분해가 없다면 이 원고는 한 상용 플랫폼의
> 보정 실험에 불과하다.

## 14. 한계와 정직한 표현

- Google Scholar의 모든 문헌을 완전하게 확인했다고 주장하지 않습니다.
- 학교 구독 논문 원문은 저장소에 재배포하지 않습니다.
- 웹 ChatGPT/Oracle 결과는 자문이며 매뉴얼·수식·동료심사 원문보다 우선하지 않습니다.
- 현재 55 kg 전방 설치물의 정확한 질량, 3D CoM, 장착 강성 및 동적 안전한계는
  실측 전 확인 필요 항목입니다.
- 실제 wheel plane, steering axis와 접촉점 기하는 광학 측정 전까지 매뉴얼 기반 추정입니다.
- 20 Hz feedback으로 tire force, 순간 고주파 slip, 마찰계수 또는 직접 안정성 여유를
  주장할 수 없습니다.

구체적 하드웨어 시험은 [EXPERIMENT_PROTOCOL_KO.md](EXPERIMENT_PROTOCOL_KO.md),
허용·금지 주장은 [VENUE_AND_CLAIM_BOUNDARIES_KO.md](VENUE_AND_CLAIM_BOUNDARIES_KO.md)를
따릅니다.
