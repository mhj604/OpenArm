# 저널 적합성 및 주장 경계

## 1. SCI/SCIE와 상위 저널의 구분

IEEE Access는 SCIE 등재 저널이면서 applied engineering, 새로운 measurement technique,
rigorous negative result를 폭넓게 다룹니다. RA-L은 concise하고 innovative한 robotics 결과,
T-RO는 high-impact state-of-the-art advance를 요구합니다.

따라서 “SCI 논문 가능”과 “분야 상위 로보틱스 저널급”을 같은 의미로 사용하지 않습니다.

## 2. 현재 범위의 판정

| 원고 범위 | IEEE Access | RA-L | T-RO/IJRR |
|---|---|---|---|
| 여러 모드에서 팔로 slip·가속·안정성 개선 | Reject 가능성 높음 | Reject | Reject |
| 단일 ROMO-B lookup scheduler | Borderline 이하 | Reject | Reject |
| 구조/excess residual 분해, direct/validated measurement, held-out authority | 조건부 Competitive | 아직 부족 | Reject |
| 다중 payload·2개 mode·직접 하중·gray-box cross-prediction | Competitive | 조건부 Competitive | Borderline 이하 |
| 일반 이론·formal guarantee·두 번째 플랫폼 | Competitive | Competitive | T-RO 검토 가치 |

## 3. IEEE Access에서 허용 가능한 주장

- 한 measured KALO-H installation에 대한 결과
- ROMO-B 매뉴얼 기하에서 ±30° Pivot의 structural non-concurrence
- 동기화된 외부 pose와 wheel feedback으로 계산한 kinematic residual proxy
- 동일 base packet과 동일 arm path 아래의 causal A-N comparison
- held-out session/payload에서 불확실성과 함께 예측된 excess residual authority
- safety non-inferiority와 full arm-return payback
- 효과가 없을 때 usable authority upper bound

## 4. 금지하거나 추가 계측이 필요한 주장

다음 표현은 현재 자료만으로 사용하지 않습니다.

- “팔이 Pivot geometry mismatch를 해결한다”
- “후륜 slip이 경로 오차의 원인이다” - 바퀴별 residual과 인과 증거 전까지 금지
- “traction force가 향상됐다” - 접지력/토크/전류 증거 전까지 금지
- “새로운 4WIS 또는 wheel-level controller” - 외부 wheel command가 없어 금지
- “일반적인 모바일 매니퓰레이터 안정성 개선” - single-platform에서 금지
- “모든 바닥과 payload에 일반화된다” - held-out/cross-platform 전까지 금지
- “45°가 ROMO-B의 정확한 이상각이다” - 0.323 m 기본 축거를 적용한 매뉴얼 기하에서는 약 39.63°
- “20 Hz feedback으로 순간 tire slip과 force를 직접 측정했다”

권장 용어:

- measured wheel-kinematics residual/proxy
- structural steering infeasibility 또는 rolling-residual floor
- condition-dependent excess residual
- uncertainty-gated authority
- effective ICR drift

## 5. IEEE Access 원고의 최소 기여

1. Steering-limit floor, steering-realization error, condition-dependent excess residual의 분해
2. 외부 body twist, individual wheel feedback, payload mass property, arm state의 동기 계측
3. 동일 base packet과 동일 arm path에서 randomized matched A-N comparison
4. 학습에 쓰지 않은 session 또는 payload에서 sign/magnitude class 예측
5. sensor/model/timing uncertainty를 포함한 prediction interval
6. raw packet, calibration, trial manifest, code와 실패 trial 공개

Scheduler는 식별한 authority envelope의 데모여야 하며, 단독 contribution이면 플랫폼별
parameter sweep으로 보입니다.

## 6. RA-L로 올리기 위한 추가사항

- synchronized dynamic corner loads 또는 force-plate evidence
- 질량, offset, 높이가 다른 3-4 payload configurations
- 2WIS 외 held-out mode
- quasi-static와 dynamic residual을 분리하는 gray-box model
- 최소 5개 독립 sessions와 confirmatory holdout
- best fixed posture, speed derating, multi-arc, offline oracle baseline
- calibrated prediction interval과 full-cycle return/payback
- 핵심 mechanism을 영상이 아닌 본문 수치로 증명

RA-L의 페이지 제약에서는 “cross-payload authority prediction과 gated scheduling” 한 주장으로
좁히고 Pivot는 mechanism negative control로만 사용해야 합니다.

## 7. T-RO 수준의 추가사항

- bounded steering geometry에서 irreducible residual의 일반 정리
- hidden wheel controller 아래 authority identifiability와 persistent excitation 조건
- 20 Hz feedback에서 residual observability 분석
- payload, timing, sensor/model uncertainty에 대한 robust 또는 chance guarantee
- 두 번째 상용 플랫폼과 다른 steering architecture
- mass ratio, geometry ratio, steering-limit ratio의 dimensionless scaling
- cross-platform useful/negligible/harmful prediction

2026-09-15까지 이 범위를 완성하는 것은 현실적이지 않습니다.

## 8. 권장 투고문

제목:

> KALO-H: Separating Structural Steering Scrub from Bimanual Load-Transfer Authority
> in a Closed-Interface 4WID-4WIS Mobile Manipulator

핵심 주장:

> We separate the irreducible rolling residual imposed by steering geometry from the
> condition-dependent excess residual under identical scalar base packets, and gate
> bimanual posture and timing only on the latter.

Primary:

\[
J_{excess}^{Naive}-J_{excess}^{Authority\text{-}gated}
\]

held-out one-sided lower confidence bound가 0을 초과하고 stability endpoint의
non-inferiority를 동시에 만족해야 합니다.
