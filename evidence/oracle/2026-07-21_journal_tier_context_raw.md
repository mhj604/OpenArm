# 편집자 총평

SCI/SCIE 등재는 서지 데이터베이스에 포함되었다는 뜻이지, 분야 최상위 저널이라는 뜻이 아닙니다. IEEE Access는 SCIE 등재 저널이지만 공식적으로 응용 중심 연구, 새로운 측정기법, 실험적 공학 해법, negative results까지 폭넓게 수용합니다. 반면 RA-L은 짧고 시의성 있는 혁신적 로보틱스 결과를, T-RO는 state of the art의 major advance를, IJRR은 시간이 지나도 다른 연구자가 구축할 수 있는 archival value를 요구합니다. 네 저널을 단순히 “SCI 논문”이라는 한 등급으로 묶으면 투고 전략을 잘못 세우게 됩니다. ([IEEE Access][1])

이 주제의 가장 강한 선행연구 충돌은 명확합니다.

* 양팔과 바퀴를 협조하는 aided locomotion: Ben Salem et al., AIM 2003, DOI 10.1109/AIM.2003.1225100
* CoM·반력 변경에 의한 traction control: Abeykoon and Ohnishi, INDIN 2005, DOI 10.1109/INDIN.2005.1560468
* manipulator를 이용한 hardware dynamic weight shifting과 rollover prevention: Storms and Tilbury, ASME JDSMC 2016, DOI 10.1115/1.4033841
* manipulator posture에 의한 stability compensation range: Huang et al., Advanced Robotics 1998, DOI 10.1163/156855399X00892
* force-angle tip-over margin: Papadopoulos and Rey, Vehicle System Dynamics 2000, DOI 10.1076/0042-3114(200001)33:1;1-5;FT029
* 실물 모바일 매니퓰레이터 안정성 측정: Bostelman and Hong, NIST TN 1955, DOI 10.6028/NIST.TN.1955
* 팔 자세 또는 차량 속도를 조정하는 실물 tip-over avoidance: Ding et al., IEEE Access 2019, DOI 10.1109/ACCESS.2019.2915115

따라서 “팔을 움직여 안정성 또는 가속 성능을 개선한다”는 아이디어 자체는 신규하지 않습니다. 남아 있는 좁은 틈은 closed commercial base의 숨겨진 제어루프를 통과한 뒤 실제로 남는 authority를 식별·예측하고, 그 authority가 검증된 영역에서만 필수 stow motion을 배치하는 것입니다. 이 판단은 기존 Google Scholar 기반 dossier의 결론과도 일치합니다. ([ResearchGate][2]) 

아래 readiness score는 acceptance probability가 아닙니다. 각 원고 범위가 계획대로 완결되었을 때의 상대적인 편집 준비도를 0–100으로 비교한 것입니다. B와 C의 직접 하중 계측 및 held-out 데이터는 아직 없으므로, 2026년 7월 21일 현재의 실제 제출 준비도는 표보다 낮습니다.

---

# 1. A/B/C별 readiness와 저널 적합성

## 1.1 상대적 readiness score

| 원고 범위                                | 학술적 신규성 | 방법론적 깊이 | 실험 증거 | 일반화 가능성 | 종합 readiness |
| ------------------------------------ | ------: | ------: | ----: | ------: | -----------: |
| A. 다중모드 가속·Ackermann·4WIS·Pivot 개선   |      15 |      18 |     8 |      10 |           13 |
| B. 2WIS 제동·단일 플랫폼·lookup scheduler   |      46 |      55 |    58 |      32 |           48 |
| C. 다중 payload·다중 mode·직접 force·물리 모델 |      74 |      84 |    88 |      72 |           80 |

### A의 해석

A는 범위가 넓어서 강해 보이지만, 실제로는 서로 다른 네 기동에 이미 알려진 weight shifting 원리를 각각 적용하는 구성입니다. “가속 성능”, “Ackermann stability”, “4WIS stability”, “Pivot stability”가 하나의 일반 모델이나 보장으로 연결되어 있지 않기 때문에, breadth는 contribution이 아니라 미완성된 네 개의 응용 실험으로 보입니다.

### B의 해석

B는 인과 비교가 훨씬 정교합니다. 동일 base packet, 동일 start/end pose와 path, Sequential–Naive–Authority-gated 비교는 단순 concurrency와 scheduler effect를 분리합니다. 그러나 reconstructed (D_{\min}), empirical lookup, 단일 플랫폼, 단일 mode라는 네 가지 한계 때문에 transferable robotics method라고 보기 어렵습니다.

### C의 해석

C는 SCI/SCIE 논문 여부를 넘어 제대로 된 robotics/mechatronics 연구 범위입니다. 직접 force evidence, 여러 payload geometry, 여러 mode, reduced-order model, held-out prediction, uncertainty, 공개 데이터가 결합되면 시스템 구축이 아니라 식별·예측 방법론이 됩니다. 다만 C조차 single-platform이면 T-RO/IJRR에서는 “강한 실물 논문”이지 반드시 “general robotics advance”는 아닙니다.

## 1.2 venue별 적합성

| 범위 | IEEE Access | RA-L        | T-RO       | IJRR       | Mechatronics | Robotics and Autonomous Systems | Journal of Field Robotics |
| -- | ----------- | ----------- | ---------- | ---------- | ------------ | ------------------------------- | ------------------------- |
| A  | Reject      | Reject      | Reject     | Reject     | Reject       | Reject                          | Reject                    |
| B  | Borderline  | Reject      | Reject     | Reject     | Borderline   | Reject                          | Reject                    |
| C  | Competitive | Competitive | Borderline | Borderline | Competitive  | Competitive                     | Reject¹                   |

¹ C를 실제 물류·산업 현장이나 대표적 비정형 환경에서 장기간 검증하면 JFR은 Borderline까지 올라갈 수 있습니다. 현재의 실내 시험장 단기 실험은 JFR의 unstructured/dynamic field environment 및 extended field validation 성격과 맞지 않습니다. ([Wiley Online Library][3])

Mechatronics는 물리 모델·식별·제어와 full-scale mechatronic apparatus의 실험 결과를 명시적으로 중시하므로 C와 매우 잘 맞습니다. Robotics and Autonomous Systems는 autonomous systems의 theoretical, computational, experimental development를 요구하므로, 단순 load shifting이 아니라 predictive authority model과 autonomous gating이 중심일 때 C가 적합합니다. ([ScienceDirect][4])

RA-L은 6쪽, 유료 초과분을 포함해 최대 8쪽이므로 C 전체를 그대로 담기는 어렵습니다. RA-L에 내기 위해서는 “cross-payload authority prediction과 gated stow scheduling”이라는 한 주장만 남기고, Pivot나 추가 mode는 supplementary video가 아니라 후속 논문으로 분리해야 합니다. RA-L은 동영상이나 멀티미디어를 본문 증거와 page-limit의 대체물로 사용할 수 없다고 명시합니다. ([IEEE RAS][5])

---

# 2. 가장 강한 desk reject 또는 Reviewer 2 reject 문장

## A

> This manuscript combines four platform-specific demonstrations of already established manipulator-based weight shifting and tip-over compensation, but provides neither a new general model nor a causal mechanism that unifies the modes.

한국어로는 다음과 같습니다.

> 이미 확립된 팔 기반 무게중심 이동과 전복 보상을 네 기동에 적용했을 뿐이며, mode를 관통하는 새로운 모델·보장·인과 메커니즘이 없다.

A는 결과가 모두 positive여도 신규성 문제가 해결되지 않습니다.

## B

> The proposed lookup scheduler is a calibration table for one ROMO-B operating condition, not a transferable robotics contribution, because the claimed stability margin is reconstructed rather than directly validated and is not predicted across payloads or modes.

한국어로는 다음과 같습니다.

> 직접 검증되지 않은 reconstructed margin을 이용해 한 ROMO-B 조건의 timing을 조정한 calibration에 불과하며, payload나 mode를 넘어서는 예측 가능성이 없다.

B에서 Reviewer 2가 공격할 지점은 scheduler가 단순하다는 것이 아닙니다. 식별 대상과 uncertainty boundary가 다른 조건에서 재사용되지 않는다는 점입니다.

## C

> Despite extensive instrumentation, the work remains a high-quality revalidation of known manipulator weight shifting unless its reduced-order model and safety bounds transfer to conditions or platforms not used for identification.

한국어로는 다음과 같습니다.

> 계측은 강하지만, 모델과 안전 경계가 식별에 쓰이지 않은 조건이나 플랫폼으로 이전되지 않으면 기존 weight-shifting 원리의 정교한 재검증에 머문다.

---

# 3. IEEE Access에서조차 단순 시스템 통합으로 보이지 않게 할 최소 기여

최소한 다음 하나가 비가역적 scientific contribution으로 남아야 합니다.

## Uncertainty-calibrated closed-interface stability-authority envelope

이는 단순한 “좋은 timing을 찾았다”가 아니라 다음 관계를 의미합니다.

[
\mathcal E:
\left(
u_b,,
m_p,,
r_p,,
\text{mode},,
q_a(\cdot)
\right)
\mapsto
\left[
\Delta D_{\min}^{L},
\Delta D_{\min}^{U}
\right],
]

여기서

* (u_b): 실제 전송된 scalar base packet sequence
* (m_p,r_p): 측정된 payload mass와 CoM
* mode: 2WIS, 4WIS 또는 Pivot
* (q_a(\cdot)): 동일한 필수 arm path의 posture·timing
* ([\Delta D_{\min}^{L},\Delta D_{\min}^{U}]): uncertainty-calibrated authority interval

입니다.

이 contribution은 최소한 다음 조건을 만족해야 합니다.

| 요구사항    | 최소 기준                                                                               |
| ------- | ----------------------------------------------------------------------------------- |
| 인과 분리   | 동일 base packet, 동일 arm start/end pose, 동일 geometric path                            |
| 측정 타당성  | 직접 corner-load/force 또는 직접 계측으로 검증된 (D_{\min}) reconstruction                       |
| 예측성     | 학습에 사용하지 않은 session 또는 payload에서 effect sign과 magnitude class 예측                    |
| 불확실성    | repeatability, sensor error, model residual, timing error를 포함한 prediction interval  |
| null 대응 | 효과가 없을 때도 usable authority의 상한을 제시                                                  |
| 재현성     | raw packet, arm states, IMU, external pose, mass/CoM, calibration, analysis code 공개 |

scheduler는 이 envelope의 응용 데모여야 합니다. scheduler 자체가 중심 contribution이면 B는 “parameter sweep 후 최적 timing 선택”으로 축소됩니다.

이 authority envelope는 positive, transient-only, null 결과 모두에서 남습니다. 따라서 이를 “비가역적” 기여라고 부를 수 있습니다. 반면 “(T_{\mathrm{ready}})가 3% 감소했다”는 결과는 다른 바닥이나 payload에서 재현되지 않으면 사라지는 가역적 결과입니다.

IEEE Access가 applied engineering, measurement technique, negative result까지 수용한다는 사실은 낮은 신규성을 허용한다는 뜻이 아닙니다. Access에서도 original, technically correct, clearly presented라는 기준을 만족해야 하며, 단순 통합 대신 새로운 측정·식별 object가 필요합니다. ([IEEE Access][6])

---

# 4. RA-L에서 Competitive가 되기 위한 정확한 추가사항

B를 그대로 RA-L에 내면 Reject가 더 타당합니다. Competitive로 올리려면 아래가 모두 필요합니다.

## 4.1 직접 하중 증거

다음 중 하나가 main confirmatory experiment 전체에 있어야 합니다.

* 네 corner의 synchronized dynamic normal-load measurement
* 또는 네 바퀴 접촉을 동시에 포착하는 synchronized force-plate arrangement

일부 정적 scale data로만 (D_{\min})을 보정하고, 주행 중에는 IMU와 reconstructed CoM만 쓰는 구성은 RA-L에서 부족합니다. 이는 “stability improvement”의 핵심 변수가 직접 검증되지 않았기 때문입니다.

영상에서 차체가 덜 흔들리는 것은 다음을 증명하지 못합니다.

* inner wheel unload가 감소했는지
* rear axle load reserve가 증가했는지
* support-margin minimum이 실제로 개선됐는지
* transient 뒤 payback이 있었는지

## 4.2 payload 조건

제조사와 구조 안전 검토를 통과한 범위에서 최소 네 조건이 필요합니다.

| 조건                                   | 목적                       |
| ------------------------------------ | ------------------------ |
| P0: nominal front-heavy payload      | 기준                       |
| P1: 같은 위치, 다른 mass                   | 질량비 일반화                  |
| P2: 비슷한 mass, 다른 longitudinal offset | 편심 일반화                   |
| P3: 비슷한 mass와 offset, 다른 height      | tip-over sensitivity 일반화 |

P0–P2로 모델을 식별하고 P3를 완전 held-out으로 두는 정도가 최소입니다. 최종 수치는 실제 안전 승인 이후 정해야 하며, 현재 mass·offset·height 값은 unknown입니다.

## 4.3 mode 조건

main paper에는 두 mode만 넣는 것이 적절합니다.

1. 2WIS straight braking
2. 4WIS constant-curvature turning

Pivot는 exploratory video나 appendix figure로도 충분하지 않습니다. Pivot를 claim하려면 별도의 identification과 held-out evaluation이 필요하므로, RA-L의 페이지 제약 안에서는 삭제하는 편이 낫습니다.

## 4.4 모델

pure lookup table을 다음 gray-box model로 바꿔야 합니다.

[
D_{\min}
========

D_{\mathrm{qs}}
\left(
m_p,r_p,q_{\mathrm{hold}},\text{mode}
\right)
+
G_{\mathrm{dyn}}
\left(
\dot h_a,\ddot h_a,u_b,\text{mode}
\right)
+
\varepsilon.
]

* (D_{\mathrm{qs}}): measured mass/CoM geometry로 계산하는 quasi-static component
* (G_{\mathrm{dyn}}): PCU를 포함한 empirical dynamic residual
* (\varepsilon): session, floor, timing, sensor uncertainty

중요한 것은 모델이 복잡한가가 아니라, posture effect와 timing/reaction effect를 분리하고 held-out condition을 예측하는가입니다.

## 4.5 session과 holdout

최소 설계는 다음입니다.

* 독립된 5개 session
* session 1–3: identification과 hyperparameter selection
* session 4–5: 완전 confirmatory holdout
* 각 held-out mode마다 session당 최소 8 matched randomized blocks
* primary A–N 비교당 최소 32 held-out matched pairs
* 최종 표본수는 pilot paired variance와 preregistered minimum practical effect를 이용해 다시 계산
* session random intercept와 condition random slope를 포함한 mixed-effects model

32가 충분하다는 보장은 없습니다. power calculation에서 더 큰 표본이 나오면 큰 값을 따라야 합니다. 32는 RA-L의 실험적 minimum floor이지 보편적인 통계 기준이 아닙니다.

## 4.6 필수 비교군

| 비교군                        | 목적                                     |
| -------------------------- | -------------------------------------- |
| Sequential                 | concurrency 자체의 이득                     |
| Naive-concurrent           | scheduler의 직접 비교군                      |
| Best fixed low-CoM posture | 기존 quasi-static weight-shifting 원리와 비교 |
| Offline oracle timing      | proposed scheduler와 가능한 최대 성능 사이 gap   |
| Authority-gated            | 제안 방법                                  |

Fixed picking pose 하나만 baseline으로 두는 것은 약합니다. Storms–Tilbury와 Huang 등의 원리를 고려하면, 최적에 가까운 static weight-shift baseline을 이겨야 합니다. 그렇지 않으면 새로운 dynamic scheduler가 아니라 단순히 팔을 낮추는 것이 정답일 수 있습니다. ([ResearchGate][7])

## 4.7 필수 ablation

1. posture-only: 팔을 보상 자세로 옮긴 뒤 고정
2. timing-only: 같은 path·duration에서 offset만 변경
3. slow-versus-fast: 같은 path와 timing에서 time scaling만 변경
4. common-versus-differential: CoM shift와 reaction/yaw effect 분리
5. uncertainty gate off: point-estimate scheduler와 비교

## 4.8 RA-L primary evidence

RA-L용 가장 방어적인 primary는 다음입니다.

[
\Delta D_{\min}^{A-N}
=====================

## D_{\min}^{\mathrm{Authority}}

D_{\min}^{\mathrm{Naive}}.
]

요구되는 결과는 단순 평균 차이가 아니라 다음 세 가지입니다.

* held-out condition에서 one-sided lower confidence bound가 0 초과
* safety endpoint에서 non-inferiority 만족
* prediction interval의 empirical coverage가 사전 목표 범위와 일치

(T_{\mathrm{ready}})는 secondary로 두는 것이 맞습니다.

RA-L 공식 scope는 significant theoretical findings뿐 아니라 중요한 application case study도 허용하지만, “significant”가 빠지지 않습니다. 한 플랫폼의 lookup 성능만으로는 이 문턱을 넘기 어렵습니다. ([IEEE RAS][8])

---

# 5. T-RO와 IJRR 수준에 필요한 것

C를 완성해도 T-RO와 IJRR에서는 Borderline입니다. Competitive로 올리려면 최소 세 축이 더 필요합니다.

## 5.1 일반 이론

다음과 같은 interface-class theory가 필요합니다.

> wheel-level inputs를 노출하지 않고 scalar speed/steer command만 받는 안정한 hidden base controller와 외부 manipulator가 결합될 때, 어떤 arm excitation이 closed-loop support-margin response에서 식별 가능한가?

구체적으로는 다음이 필요합니다.

* authority identifiability condition
* persistent excitation 조건
* base-loop bandwidth와 arm-loop bandwidth 비에 따른 authority attenuation
* arm-to-total mass ratio, CoM stroke/wheelbase ratio, momentum ratio 등의 무차원 scaling
* mode change가 authority sign에 미치는 조건
* hidden PCU dynamics와 timing uncertainty에서 구분 가능한 model class

ROMO-B에 fit된 FIR coefficient를 제시하는 것은 이론이 아닙니다.

## 5.2 보장

최소 하나의 명시적인 보장이 필요합니다.

예를 들면

[
\Pr
\left[
D_{\min}\ge d_{\mathrm{safe}}
\right]
\ge
1-\alpha
]

를 payload, timing, model residual, sensor uncertainty 아래에서 만족하는 chance-constrained scheduler입니다.

또는 bounded uncertainty에 대해

[
\underline D_{\min}
\left(
q_a,\tau,T
\right)
\ge d_{\mathrm{safe}}
]

를 보장하는 robust lower-bound governor가 필요합니다.

단순히 validation set에서 사고가 없었다는 것은 safety guarantee가 아닙니다.

## 5.3 cross-platform validation

현실적으로 최소 두 개의 서로 다른 interface class가 필요합니다.

예:

* ROMO-B: scalar speed/steer/mode, hidden 4WIS controller
* 다른 상용 base: differential, omnidirectional 또는 다른 steering architecture의 velocity-controlled base

중요한 것은 두 로봇 모두에서 성능이 좋아야 한다는 것이 아닙니다. 동일한 이론이

* authority의 sign
* useful/negligible/harmful classification
* required calibration amount
* uncertainty coverage

를 설명해야 합니다.

두 번째 플랫폼에서 전체 모델을 처음부터 다시 fit해야 한다면 cross-platform theory가 아니라 플랫폼별 calibration 두 개입니다.

T-RO는 공식적으로 major advances in the state of the art를 요구합니다. IJRR은 original·solid뿐 아니라 다른 연구자가 장기간 구축할 수 있는 archival value를 요구합니다. C에 두 번째 플랫폼과 위 이론·보장이 없으면 이 기준에 미달할 가능성이 큽니다. ([IEEE RAS][9])

## 현실성

2026년 9월 15일까지는 불가능합니다.

현재 남은 기간에 현실적으로 가능한 일은

* 한 플랫폼 mass/CoM calibration
* 한 mode의 load measurement
* 저차원 empirical model
* session holdout
* same-path scheduler evaluation

정도입니다.

새로운 일반 이론, formal guarantee, 두 번째 상용 플랫폼 통합, cross-platform validation을 같은 기간에 수행하면 어느 것도 충분히 검증되지 않습니다. T-RO/IJRR를 목표로 한다면 9월 제출 목표를 버리는 것이 맞습니다.

---

# 6. single-platform empirical lookup scheduler는 충분한가

## 단독 contribution으로는 충분하지 않습니다

single-platform lookup scheduler 하나만 있으면 학술적으로 다음 중 하나입니다.

* robot commissioning 결과
* operating table
* application-specific calibration
* technical report
* benchmark의 baseline method

다음 형태라면 논문 contribution이 될 수 있습니다.

[
\text{Identification protocol}
+
\text{uncertainty-qualified authority envelope}
+
\text{held-out validation}
+
\text{scheduler demonstration}.
]

즉, scheduler는 최종 네 번째 요소입니다.

### venue별 판정

| venue        | lookup scheduler 단독                            |
| ------------ | ---------------------------------------------- |
| IEEE Access  | Reject 또는 약한 Borderline                        |
| RA-L         | Reject                                         |
| T-RO         | Reject                                         |
| IJRR         | Reject                                         |
| Mechatronics | 모델·직접 계측이 없으면 Reject                           |
| RAS          | 일반화 가능한 autonomous decision method가 없으면 Reject |

Access에서 살릴 수 있는 경우는 scheduler를 주 기여로 쓰지 않고 다음처럼 쓰는 경우입니다.

> We first identify and validate an uncertainty-qualified stability-authority boundary for a closed-interface mobile manipulator; a lookup scheduler is then used only to demonstrate how the identified boundary can be exploited.

첨부 dossier도 optimizer의 복잡성보다 식별과 prediction을 contribution으로 두어야 한다고 지적합니다. 

---

# 7. null 또는 transient-only 결과가 살아남기 위한 조건

## 7.1 IEEE Access

IEEE Access는 공식적으로 negative results와 새로운 measurement technique을 scope에 포함합니다. 따라서 null result 자체는 결격 사유가 아닙니다. 그러나 “유의하지 않았다”만으로는 논문이 아닙니다. ([IEEE Access][6])

필요한 것은 다음입니다.

### 통계 자산

* equivalence test 또는 smallest effect of interest 기반 upper bound
* (p>0.05)가 아니라

[
|\Delta D_{\min}|
<
\delta_{\mathrm{practical}}
]

라는 결론

* session 및 condition을 포함한 hierarchical uncertainty
* sensor·mass·CoM·timing·model uncertainty budget
* 사전등록된 exclusion과 abort rule

### excitation sufficiency

* 실제 arm mass와 CoM range
* 실현된 arm momentum와 acceleration spectrum
* actuator saturation 여부
* PCU command와 feedback bandwidth
* tested region이 가능한 authority region을 충분히 덮었는지

이를 제시하지 않으면 Reviewer는 “팔을 충분히 움직이지 않아 효과를 못 본 것”이라고 반박합니다.

### 조건 범위

최소한

* 여러 session
* 둘 이상의 payload/CoM 조건
* 둘 이상의 arm timing family
* 가능하면 2WIS와 4WIS

가 필요합니다.

단일 날, 단일 payload, 단일 궤적에서 null이면 논문이 아니라 failed pilot입니다.

### 공개 자산

* synchronized RS-232 packet bytes와 timestamps
* 양쪽 CAN logs
* external pose와 chassis IMU
* direct corner-load 또는 force data
* component mass·CoM·geometry
* arm trajectory와 clamp state
* trial manifest
* 실패·abort·재시도 이유
* calibration code
* preprocessing 및 statistics code
* CAD/URDF와 coordinate definitions

IEEE Access가 code reproducibility initiative를 운용한다는 점도 이 방향과 맞습니다. ([IEEE Access][10])

## 7.2 RA-L

RA-L에서 single-platform null result는 거의 항상 Reject입니다.

살아남으려면 null이 일반적이고 예상 밖인 결론이어야 합니다. 예를 들면 다음입니다.

> Hidden commercial velocity loops attenuate or reverse arm-induced support-margin authority in a predictable mode-dependent manner, despite measurable open-loop chassis reactions.

이를 방어하려면

* direct load evidence
* 여러 payload와 mode
* held-out prediction
* model이 설명하는 sign reversal 또는 attenuation
* full return payback
* 다른 플랫폼 또는 최소한 다른 embedded-controller condition

이 필요합니다.

단순히 “ROMO-B에서 팔을 흔들었지만 (D_{\min})이 개선되지 않았다”는 RA-L 논문이 아닙니다.

## 7.3 transient-only 결과

transient-only 결과의 핵심은 다음 차이를 계량화하는 것입니다.

[
\text{chassis-level transient}
\neq
\text{whole-system support-margin gain}.
]

필수 결과는

* chassis pose에는 차이가 있음
* direct/reconstructed whole-system margin에서는 차이가 작거나 반전됨
* arm return 이후 payback 크기
* closed PCU가 transient를 억제·증폭하는 frequency range
* usable authority upper bound

입니다.

이 경우 scheduler를 억지로 유지하면 논문이 약해집니다. “왜 apparent benefit이 task-level stability benefit으로 전환되지 않는가”가 논문이어야 합니다.

---

# 8. 투고 전략 한 개씩

## 2026년 9월 15일을 유지하는 경우

### 추천 venue

IEEE Access

### 추천 범위

B를 다음처럼 수정한 원고입니다.

> 2WIS 직선 제동에서, 전방 편심 모바일 매니퓰레이터의 필수 picking-to-stow motion이 closed-interface base를 통과해 제공하는 support-margin authority를 불확실성과 함께 식별하고, 그 하한이 양수인 timing만 선택한다.

### 추천 제목

> KALO-H: Uncertainty-Calibrated Bimanual Stability-Authority Identification on a Front-Heavy Closed-Interface Mobile Manipulator

### 허용 claim

* 한 measured front-heavy installation
* 한 documented mode
* same-base-packet causal comparison
* uncertainty-qualified authority
* same-path gated scheduling
* full-return hardware evaluation

### 금지 claim

* general stability improvement
* acceleration improvement
* traction improvement
* new 4WIS control
* universal bimanual control
* cross-mode generalization

B를 그대로 두고 direct load validation이나 held-out prediction 없이 제출하면 Access에서도 Borderline 이하입니다.

## 제출을 미루는 경우

### 추천 venue

IEEE Transactions on Robotics

### 필요한 확장

* C 전체
* 두 번째 상용 mobile base
* general interface-class authority model
* formal identifiability 또는 robust safety bound
* cross-platform prediction
* direct force validation
* 공개 benchmark

### 추천 제목

> Interface-Limited Stability Authority of Mobile Manipulators: Identification, Safety Guarantees, and Cross-Platform Validation

IJRR보다 T-RO가 먼저인 이유는 이 연구가 순수 이론보다 hardware dynamics, system identification, control, integration의 결합에 가깝기 때문입니다. T-RO 역시 major advance를 요구하지만, 강한 이론과 두 플랫폼의 실물 검증을 갖추면 scope가 자연스럽습니다. ([IEEE RAS][9])

---

# 9. 최종 판정

| 질문                                       | 판정             |
| ---------------------------------------- | -------------- |
| 지금 당장 분야 상위 SCI/SCIE 저널급 주제인가            | NO             |
| 현재 A가 IEEE Access급인가                     | NO             |
| 수정된 B가 IEEE Access급인가                    | YES, 조건부       |
| 현재 B가 RA-L급인가                            | NO             |
| C가 RA-L급인가                               | YES, 조건부       |
| C가 곧바로 T-RO/IJRR급인가                      | NO, Borderline |
| C에 일반 이론·보장·cross-platform이 추가되면 T-RO급인가 | YES, 검토 가치 있음  |

한 문장 결론은 다음과 같습니다.

> 이 주제는 현재 상위 로보틱스 저널급은 아니며, IEEE Access급으로 살아남으려면 “팔로 안정성을 개선했다”가 아니라 직접 또는 추적 가능하게 검증된 closed-interface stability-authority envelope, held-out prediction, 동일 path·packet 인과 비교, safety non-inferiority, 공개 재현성 자산을 제출해야 합니다.

[1]: https://ieeeaccess.ieee.org/about/bibliometrics/ "https://ieeeaccess.ieee.org/about/bibliometrics/"
[2]: https://www.researchgate.net/publication/4031159_Performance_study_of_aided_locomotion_for_an_outdoor_mobile_robot_with_arms "https://www.researchgate.net/publication/4031159_Performance_study_of_aided_locomotion_for_an_outdoor_mobile_robot_with_arms"
[3]: https://onlinelibrary.wiley.com/page/journal/15564967/homepage/productinformation.html "https://onlinelibrary.wiley.com/page/journal/15564967/homepage/productinformation.html"
[4]: https://www.sciencedirect.com/journal/mechatronics "https://www.sciencedirect.com/journal/mechatronics"
[5]: https://www.ieee-ras.org/publications/ra-l/faq/ "https://www.ieee-ras.org/publications/ra-l/faq/"
[6]: https://ieeeaccess.ieee.org/about/ "https://ieeeaccess.ieee.org/about/"
[7]: https://www.researchgate.net/publication/303889789_Dynamic_Weight-Shifting_for_Improved_Maneuverability_and_Rollover_Prevention_in_High-Speed_Mobile_Manipulators "https://www.researchgate.net/publication/303889789_Dynamic_Weight-Shifting_for_Improved_Maneuverability_and_Rollover_Prevention_in_High-Speed_Mobile_Manipulators"
[8]: https://www.ieee-ras.org/publications/ra-l/ "https://www.ieee-ras.org/publications/ra-l/"
[9]: https://www.ieee-ras.org/publications/t-ro/ "https://www.ieee-ras.org/publications/t-ro/"
[10]: https://ieeeaccess.ieee.org/news/ieee-access-reproducibility-initiative/ "https://ieeeaccess.ieee.org/news/ieee-access-reproducibility-initiative/"
