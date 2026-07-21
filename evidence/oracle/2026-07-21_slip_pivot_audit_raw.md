# 총괄 판정

핵심 기하학 진단은 대체로 맞습니다. 다만 두 표현을 수정해야 합니다.

1. (0.253\ \mathrm{m})는 엄밀히 말해 Wk = 0.390 m로 계산한 조향 너클 중심의 반경이지, track = 0.529 m로 정의되는 무조향 상태의 wheel-center 반경이 아닙니다.
2. (0.0428|\dot\psi|\ \mathrm{m/s})는 강제된 중심 회전에서 발생하는 wheel-plane 횡방향 rolling-constraint residual입니다. 측정된 tire slip ratio, lateral tire force, scrub power가 아닙니다.

그럼에도 표준적인 조향축–휠 중심 offset이 휠 횡방향을 따라 존재한다고 보면 그 offset 항이 lateral constraint에서 소거되므로, (W_k)를 사용한 39.72°와 (0.0428|\dot\psi|) 계산은 물리적으로 타당합니다. 반면 “팔을 움직여 이 pivot 기하 오차를 고친다”는 가설은 틀렸습니다. 팔은 기하학적 비동시성 자체를 제거할 수 없습니다.

새 관찰은 IEEE Access 원고를 강화할 수 있지만, 그 강화점은 “팔로 traction·pivot 성능을 개선한다”가 아닙니다. 가장 강한 연구 질문은 다음입니다.

> 폐쇄형 4WID–4WIS 베이스에서 조향 한계로 인한 제거 불가능한 구조적 rolling residual과, payload·하중이동·마찰·embedded PCU 때문에 생기는 조건 의존 residual을 실험적으로 분해하고, 양팔이 실제로 바꿀 수 있는 후자에 대해서만 authority를 식별할 수 있는가?

ROMO-B 매뉴얼은 외부 입력이 20 Hz scalar SPEED/STEER/MODE와 안전 관련 필드로 제한되고, 네 바퀴 속도·조향각은 feedback이라는 점을 명확히 보여 줍니다. 따라서 새 4WIS 제어나 wheel-level allocation 논문으로 쓰는 것은 불가능합니다. 

---

# A. 사용자의 물리 진단은 맞는가

## A-1. Pivot 기하학

차체 좌표계를 (x) 전방, (y) 좌측으로 두고 조향 너클 중심을

[
\begin{aligned}
r_{FL}&=(+L/2,+W_k/2),\
r_{FR}&=(+L/2,-W_k/2),\
r_{RL}&=(-L/2,+W_k/2),\
r_{RR}&=(-L/2,-W_k/2)
\end{aligned}
]

로 놓습니다.

중심에서 순수 yaw하는 경우

[
\xi_c=[v_x,v_y,\dot\psi]^\top=[0,0,\dot\psi]^\top
]

이고 각 조향축 위치의 속도는

[
v_i=\dot\psi
\begin{bmatrix}
-y_i\
x_i
\end{bmatrix}
]

입니다.

휠 rolling 방향을

[
t_i=
\begin{bmatrix}
\cos\delta_i\
\sin\delta_i
\end{bmatrix},
\qquad
n_i=
\begin{bmatrix}
-\sin\delta_i\
\cos\delta_i
\end{bmatrix}
]

로 놓으면 zero lateral rolling residual 조건은

[
n_i^\top v_i=0
]

입니다.

대칭 중심 pivot에서 필요한 조향각 크기는

[
\theta^\star
============

# \tan^{-1}\left(\frac{L}{W_k}\right)

# \tan^{-1}\left(\frac{0.324}{0.390}\right)

39.7188^\circ.
]

따라서 45°는 (L=W_k)일 때만 맞습니다. ROMO-B의 ±30° 고정 pivot은

[
\Delta\theta
============

# 39.7188^\circ-30^\circ

9.7188^\circ
]

의 구조적 불일치를 가집니다.

조향 너클 중심 반경은

[
r_k
===

\sqrt{\left(\frac L2\right)^2+
\left(\frac {W_k}2\right)^2}
============================

0.25351\ \mathrm{m}.
]

중심 yaw를 강제하면 각 휠의 lateral rolling residual 크기는

[
|e_{\perp,i}|
=============

r_k|\dot\psi|
\sin|\Delta\theta|
==================

0.04280|\dot\psi|\ \mathrm{m/s}.
]

따라서 연구자가 제시한 (0.0427|\dot\psi|)는 반올림 오차 내에서 맞습니다.

예를 들어 실제 yaw rate가 (0.5\ \mathrm{rad/s})라면 기하학적 residual 규모는 휠당 약

[
0.0214\ \mathrm{m/s}
]

입니다. 단, 이것은 tire force도 slip ratio도 아닙니다.

## A-2. 왜 track 0.529 m가 아니라 (W_k=0.390) m인가

여기가 가장 혼동하기 쉬운 부분입니다.

조향축에서 실제 wheel center 또는 contact patch까지의 offset이 휠 횡방향 (n_i)을 따라 (d_i)만큼 있다고 하면

[
r_{c,i}=r_{k,i}+d_i n_i.
]

그러면 중심 yaw에서

[
n_i^\top\dot\psi J r_{c,i}
==========================

n_i^\top\dot\psi J r_{k,i}
+
d_i\dot\psi n_i^\top Jn_i.
]

그런데

[
n_i^\top Jn_i=0
]

이므로

[
n_i^\top\dot\psi J r_{c,i}
==========================

n_i^\top\dot\psi J r_{k,i}.
]

즉, 일반적인 kingpin lateral offset은 lateral rolling constraint에서 소거됩니다. 그래서 단순히 track 0.529 m를 넣어

[
\tan^{-1}(0.324/0.529)=31.5^\circ
]

라고 계산하는 것은 오히려 잘못될 수 있습니다.

다만 다음은 아직 unknown입니다.

* 실제 steering-axis 위치
* caster/trail 또는 longitudinal offset
* steering linkage와 tire compliance
* 30° feedback이 실제 wheel plane 30°와 일치하는지
* 하중에 따른 조향 구조 변형

따라서 첫 실험은 manual geometry를 그대로 믿는 것이 아니라 actual wheel-plane angle과 steering-axis geometry를 광학적으로 측정하는 것이어야 합니다.

## A-3. 네 wheel normal이 한 ICR에 모이는가

대칭 ±(\theta) cross steering에서 front pair의 virtual ICR은

[
x_F
===

\frac L2-\frac{W_k}{2}\tan\theta
]

에 있고 rear pair는

[
x_R
===

-\frac L2+\frac{W_k}{2}\tan\theta
]

에 있습니다.

(\theta=30^\circ)이면

[
x_F=+0.04942\ \mathrm{m},
\qquad
x_R=-0.04942\ \mathrm{m}.
]

따라서 front와 rear의 virtual ICR 사이에는

[
|x_F-x_R|=0.09883\ \mathrm{m}
]

의 간격이 있습니다.

이는 중요한 결론입니다.

> 비영 yaw rate에서 모든 휠의 zero-slip 조건을 동시에 만족하는 단일 rigid-body ICR가 존재하지 않습니다.

차체 center drift가 발생해도 모든 residual을 0으로 만들 수는 없습니다. 실제 시스템은 다음 중 하나 또는 여러 개를 통해 불일치를 분배합니다.

* tire lateral deformation과 scrub
* 휠별 다른 longitudinal slip
* steering compliance
* chassis translation과 effective ICR drift
* yaw rate 감소
* 하중이 작은 휠에서 더 큰 residual 허용
* hidden PCU wheel-speed allocation

대칭 하중과 대칭 마찰이면 least-squares 최적 translation은 대체로 0에 가깝습니다. 전방 편심·좌우 하중 비대칭이 크면 실제 ICR가 이동하여 하중이 큰 휠의 residual을 줄이고 다른 휠에 더 많은 residual을 넘길 수 있습니다.

## A-4. 2WIS에서 “rear-wheel slip”이 원인인가

현재는 증명되지 않은 가설입니다.

매뉴얼로 확인되는 것은 다음뿐입니다.

* 2WIS에서 rear steer는 0°
* scalar steering command가 front Ackermann-like angles로 변환됨
* 네 휠 속도와 조향각 feedback이 제공됨
* PCU가 네 drive motor를 내부적으로 제어함
* 외부 사용자는 wheel-level speed ratio를 직접 명령하지 못함

실제 경로 오차는 최소 네 원인으로 분리해야 합니다.

| 원인                                            | 현재 상태        |
| --------------------------------------------- | ------------ |
| front/rear lateral tire slip angle            | 가설           |
| rear 또는 전체 wheel의 longitudinal slip ratio     | 가설           |
| steering angle calibration·hysteresis·latency | 미측정          |
| hidden PCU wheel-speed allocation과 saturation | 미측정          |
| payload에 의한 corner load 변화                    | 높은 개연성이나 미측정 |
| 실제 geometry와 manual model 불일치                 | 미측정          |
| tire radius 차이·공기압·바닥 편차                      | 미측정          |

특히 rear wheel은 조향되지 않았지만 4WID로 구동됩니다. PCU가 이상적인 path radius에 맞는 네 휠 속도 비를 정확히 생성하지 않으면 longitudinal residual이 생길 수 있습니다. 그러나 “rear-wheel slip 때문에 path가 틀어진다”라고 쓰려면 rear wheel residual이 front보다 크고, 그 residual이 trajectory error를 설명한다는 데이터를 보여야 합니다.

Chung과 Velinsky는 이미 Dugoff tire model을 포함한 mobile-manipulator slip dynamics와 tracking degradation을 다뤘습니다. 따라서 slip 자체를 모델링하는 것은 신규성이 아닙니다. DOI 10.1017/S0263574798000873. ([Cambridge University Press & Assessment][1])

### A의 최종 판정

| 주장                                     | 판정       |                           |         |
| -------------------------------------- | -------- | ------------------------- | ------- |
| 이상적 중심 pivot angle이 45°다               | 틀림       |                           |         |
| 이상적 중심 pivot angle이 39.72°다            | 조건부로 맞음  |                           |         |
| 30° pivot에는 구조적 kinematic mismatch가 있다 | 맞음       |                           |         |
| (0.0428                                | \dot\psi | )가 횡 rolling residual 규모다 | 조건부로 맞음 |
| 이것이 측정된 tire slip ratio다               | 틀림       |                           |         |
| center drift가 residual을 재분배할 수 있다      | 맞음       |                           |         |
| center drift가 모든 휠 residual을 제거할 수 있다  | 틀림       |                           |         |
| 2WIS 경로 오차가 rear-wheel slip 때문이다       | 아직 가설    |                           |         |
| 팔을 움직이면 pivot geometry mismatch가 사라진다  | 틀림       |                           |         |

---

# B. 이 사실이 기존 front-heavy stability 원고를 실질적으로 강화하는가

## venue별 엄격 판정

| venue       |     강화 여부 | 이유                                                                                |
| ----------- | --------: | --------------------------------------------------------------------------------- |
| IEEE Access |  YES, 조건부 | 구조적으로 제거 불가능한 residual과 arm이 바꿀 수 있는 load-dependent residual을 분리하면 측정·식별 논문이 됩니다. |
| IEEE RA-L   | NO, 현재 수준 | 단일 플랫폼 geometry observation과 lookup 보정만으로는 transferable robotics method가 아닙니다.    |
| IEEE T-RO   |        NO | 일반 이론, 보장, cross-platform validation이 없습니다.                                       |
| IJRR        |        NO | 플랫폼 독립적 archival theory와 장기적 일반성이 없습니다.                                           |

IEEE Access에서 YES가 되는 것은 다음 프레이밍으로 바꿀 때뿐입니다.

> 팔로 pivot을 개선한다.

가 아니라

> pivot의 제거 불가능한 geometry floor를 먼저 식별하고, 팔이 변경 가능한 excess residual의 authority만 별도로 추정한다.

여야 합니다.

## 신규성 충돌

이미 점유된 영역은 다음과 같습니다.

* tire slip을 포함한 mobile manipulator dynamics와 tracking control: Chung–Velinsky, Robotica 1998, DOI 10.1017/S0263574798000873. ([Cambridge University Press & Assessment][1])
* 팔 자세로 wheel vertical force를 바꿔 slip을 회피: ROBIO 2011, DOI 10.1109/ROBIO.2011.6181521. ([CoLab][2])
* manipulator motion·reaction force·joint acceleration에 따른 slip analysis와 실험: Song et al., ASME JDSMC, DOI 10.1115/1.4037287. ([OUCI][3])
* 4WIS large-curvature에서 ICR tracking control: Chu et al., IEEE T-ITS 2024, DOI 10.1109/TITS.2024.3432885. ([IEEE Xplore][4])
* manipulator dynamic weight shifting과 rollover prevention: Storms–Tilbury, DOI 10.1115/1.4033841. ([ResearchGate][5])
* arm posture 또는 base speed에 의한 physical tip-over avoidance: Ding et al., DOI 10.1109/ACCESS.2019.2915115. ([Directory of Open Access Journals][6])

따라서 다음은 논문 기여가 될 수 없습니다.

* arm-based slip avoidance
* arm-based traction improvement
* arm-based weight shifting
* ICR control 일반론
* 4WIS path tracking
* tip-over avoidance 일반론

남는 기여는 매우 좁습니다.

> wheel-level command가 없는 20 Hz scalar interface에서, structural steering infeasibility와 payload/load/friction residual을 계측적으로 분리하고, 팔이 바꿀 수 있는 closed-loop residual의 불확실성 경계를 정하는 것.

앞선 Scholar 기반 dossier 역시 broad traction·weight-shifting claim은 이미 점유되었으며, 남는 방향은 closed-interface authority identification이라고 결론냈습니다. 다만 이 literature screening이 모든 Google Scholar 논문을 완전하게 포함한다고 보장할 수는 없습니다. 

---

# C. 가장 강한 통합 scientific contribution

## 제안: geometry-aware residual budget

각 wheel의 lateral rolling residual을

[
e_{\perp,i}
===========

n_i(\delta_i)^\top
\begin{bmatrix}
v_x-\dot\psi y_i\
v_y+\dot\psi x_i
\end{bmatrix}
]

로 둡니다.

이를 세 층으로 분리합니다.

### 1. 구조적 kinematic floor

측정된 yaw rate (\dot\psi)와 steering angles (\delta_i)를 고정하고, 가능한 body translation 중 가장 작은 residual을 구합니다.

[
J_{\mathrm{geom}}(\dot\psi,\delta)
==================================

\min_{v_x,v_y}
\sum_{i=1}^4
w_i
e_{\perp,i}^2.
]

순수 geometry floor를 평가할 때는

[
w_i=1
]

을 사용합니다.

이 값은 steering geometry와 angle limit 때문에 제거할 수 없는 부분입니다.

### 2. steering realization residual

nominal command mapping과 실제 steering feedback 또는 optical wheel-plane angle의 차이를 구분합니다.

[
\Delta J_{\mathrm{steer}}
=========================

## J_{\mathrm{geom}}(\dot\psi,\delta_{\mathrm{actual}})

J_{\mathrm{geom}}(\dot\psi,\delta_{\mathrm{nominal}}).
]

여기에는

* static calibration bias
* left/right asymmetry
* latency
* hysteresis
* steering compliance

가 들어갑니다.

### 3. excess dynamic residual

실제 측정 twist로 계산한 residual은

[
J_{\mathrm{kin}}
================

\sum_i e_{\perp,i}^2
]

이고,

[
J_{\mathrm{excess}}
===================

## J_{\mathrm{kin}}

J_{\mathrm{geom}}(\dot\psi,\delta_{\mathrm{actual}})
\ge0
]

로 정의할 수 있습니다.

(J_{\mathrm{excess}})에는 다음이 섞여 있습니다.

* hidden wheel-speed allocation
* payload/load transfer
* surface friction과 tire compliance
* arm posture와 arm reaction
* PCU delay·saturation
* unmodeled geometry
* measurement timing error

따라서 이를 곧바로 “friction residual”이라고 부르면 안 됩니다. payload·surface·corner load·arm 조건을 바꾼 실험으로 추가 분해해야 합니다.

## Ackermann과 Pivot을 한 논문에 넣을 것인가

한 논문에 넣을 수 있지만 역할을 동등하게 두면 안 됩니다.

| mode           | 논문 내 역할                                          |
| -------------- | ------------------------------------------------ |
| 2WIS/Ackermann | primary intervention mode                        |
| Pivot          | structural negative control 및 falsification mode |
| 4WIS           | 2026-09-15 원고에서는 제외하거나 exploratory               |

2WIS를 primary로 두는 이유는 이상적인 steering geometry 자체는 대체로 feasible하고, payload·wheel-speed allocation·friction에 따른 residual을 팔이 일부 바꿀 가능성이 있기 때문입니다.

Pivot는 “팔이 구조적 residual을 제거할 수 없다”는 negative control로 사용해야 합니다. Pivot에서 arm motion으로 (J_{\mathrm{geom}}) 자체가 사라졌다고 나온다면, 발견이라기보다 wheel-angle calibration 또는 모델 오류를 먼저 의심해야 합니다.

두 mode를 모두 “팔로 성능 향상” 사례로 쓰면 범위가 무너집니다.

---

# D. 필요한 식과 측정량

## D-1. Body twist와 wheel velocity

mocap과 IMU에서 body-frame twist를

[
\xi=
\begin{bmatrix}
v_x&v_y&\dot\psi
\end{bmatrix}^\top
]

로 얻습니다.

wheel (i)의 위치를

[
r_i=
\begin{bmatrix}
x_i&y_i
\end{bmatrix}^\top
]

라 하면 wheel 위치의 rigid-body velocity는

[
v_i=
\begin{bmatrix}
v_x-\dot\psi y_i\
v_y+\dot\psi x_i
\end{bmatrix}.
]

## D-2. Longitudinal rolling residual

wheel effective radius (R_i), wheel angular speed (\Omega_i)를 사용하면

[
u_i=R_i\Omega_i
]

이고,

[
v_{\parallel,i}=t_i^\top v_i.
]

longitudinal residual은

[
e_{\parallel,i}
===============

u_i-v_{\parallel,i}.
]

정규화 slip-ratio proxy는 부호 규약을 명시한 뒤

[
\kappa_i
========

\frac{u_i-v_{\parallel,i}}
{\max(|u_i|,|v_{\parallel,i}|,v_\epsilon)}
]

로 둘 수 있습니다.

주의점은 다음과 같습니다.

* 저속에서는 분모가 작아져 불안정합니다.
* effective rolling radius를 하중별로 보정해야 합니다.
* 20 Hz wheel-speed feedback은 고주파 transient slip을 포착하지 못할 수 있습니다.
* 이 값은 tire force가 아닙니다.

## D-3. Lateral rolling residual과 slip-angle proxy

[
v_{\perp,i}=n_i^\top v_i
]

가 lateral rolling residual입니다.

kinematic slip-angle proxy는

[
\alpha_i
========

\tan^{-1}
\left(
\frac{v_{\perp,i}}{v_{\parallel,i}}
\right)
]

또는 수치적으로

[
\alpha_i=\operatorname{atan2}
(v_{\perp,i},v_{\parallel,i})
]

를 사용합니다.

이 역시 실제 tire constitutive slip angle과 lateral force 관계를 직접 측정한 것은 아닙니다. tire lateral stiffness, saturation, contact deformation을 포함하지 않습니다.

## D-4. Effective ICR

[
|\dot\psi|>\epsilon_\psi
]

일 때 mocap twist로 얻는 effective ICR은

[
p_{\mathrm{ICR}}=
\begin{bmatrix}
-v_y/\dot\psi\
v_x/\dot\psi
\end{bmatrix}.
]

휠 steering lines만 사용한 geometry ICR는

[
\hat p_{\mathrm{ICR,geom}}
==========================

\arg\min_p
\sum_i
w_i
\left[
t_i^\top(p-r_i)
\right]^2.
]

geometry inconsistency는

[
E_{\mathrm{ICR,geom}}
=====================

\sqrt{
\frac{1}{4}
\sum_i
\left[
t_i^\top(\hat p-r_i)
\right]^2
}.
]

Pivot에서 (E_{\mathrm{ICR,geom}}>0)이면 wheel normal들이 한 점에 모이지 않는다는 뜻입니다.

## D-5. Reconstructed total CoM

component mass와 arm state로

[
r_G(q_L,q_R)
============

\frac{
\sum_j m_j r_j(q_L,q_R)
}{
\sum_jm_j
}
]

를 계산합니다.

이는 mocap으로 측정한 chassis pose와 다릅니다.

* chassis pose: rigid base frame의 위치·자세
* reconstructed total CoM: base, payload, arms, pedestal를 합한 질량 중심
* wheel center velocity: rigid-body twist로 계산한 wheel 위치 속도

이 셋을 혼용하면 mechanism 분석이 붕괴합니다.

## D-6. Support margin

reconstructed total CoM와 IMU를 이용한 ZMP approximation은

[
F_z=M(g+a_z)
]

[
x_Z
===

x_G-
\frac{z_GMa_x+\dot H_{G,y}}{F_z}
]

[
y_Z
===

y_G+
\frac{\dot H_{G,x}-z_GMa_y}{F_z}
]

입니다.

support margin은

[
D(t)
====

\operatorname{sdist}
\left(
r_Z(t),
\partial\mathcal P(t)
\right)
]

로 둡니다.

직접 dynamic corner load가 있으면

[
r_{\mathrm{CoP}}
================

\frac{\sum_iF_{z,i}r_i}
{\sum_iF_{z,i}}
]

와

[
F_{z,\min}
==========

\min_i F_{z,i}
]

를 함께 사용해야 합니다.

단, normal-load sensor만으로 longitudinal traction force가 측정되는 것은 아닙니다.

## D-7. Trajectory metrics

2WIS에서는

[
e_y(t)
======

\text{reference path에 대한 cross-track error}
]

[
e_\psi(t)
=========

\psi(t)-\psi_{\mathrm{ref}}(t)
]

[
\kappa_{\mathrm{actual}}
========================

\frac{\dot\psi}
{\sqrt{v_x^2+v_y^2}}
]

[
e_\kappa
========

## \kappa_{\mathrm{actual}}

\kappa_{\mathrm{reference}}
]

를 사용합니다.

Pivot에서는

[
E_{\mathrm{center}}
===================

\sqrt{
\frac1T
\int_0^T
|
p_{\mathrm{ICR}}(t)-p_{\mathrm{desired}}
|^2dt
}
]

와 최종 translation drift

[
e_{p,f}
=======

|
p(T)-p(0)
|
]

를 사용합니다.

## D-8. 20 Hz feedback만으로 주장할 수 없는 것

20 Hz RS-232 feedback만으로는 다음을 주장할 수 없습니다.

* tire traction force
* lateral tire force
* instantaneous high-frequency slip
* wheel normal force
* arm-induced ground impulse
* true tire friction coefficient
* high-bandwidth steering lag
* time-tagged base–arm synchronization
* direct dynamic stability margin

20 Hz feedback으로 가능한 것은 저주파 wheel-kinematics proxy와 PCU response identification입니다. 4WIS/4WID 연구가 일반적으로 가정하는 네 steering·drive motor 직접 입력과 ROMO-B의 scalar interface는 완전히 다릅니다. 기존 4WIS/4WID 제어 연구는 네 독립 wheel actuator를 직접 조정하는 구조를 전제로 합니다. DOI 10.1049/joe.2014.0241. ([Directory of Open Access Journals][7])

---

# E. 결정적 hardware experiment

## E-1. Payload 조건

정확한 mass·offset·height 수치는 승인된 안전 범위가 아직 unknown이므로 임의로 만들면 안 됩니다.

최소 네 조건을 구성하십시오.

| 조건 | 변경 변수                            | 목적                    |
| -- | -------------------------------- | --------------------- |
| P0 | nominal front-heavy installation | 기준                    |
| P1 | lower mass, 동일 (x,z)             | mass effect           |
| P2 | 유사 mass, smaller forward offset  | (x_G) effect          |
| P3 | 유사 mass·offset, lower height     | (z_G) held-out effect |

P0–P2를 identification에 사용하고 P3를 완전 held-out으로 남기는 구성이 가장 낫습니다.

가능하면 무차원 변수로 정리하십시오.

[
\lambda_m=\frac{m_p}{M},
\qquad
\lambda_x=\frac{x_p}{L},
\qquad
\lambda_z=\frac{z_p}{W_k}.
]

## E-2. Surface 조건

surface variation은 다음 조건에서만 넣습니다.

* static/dynamic safety review 완료
* surface friction을 별도 tribometer 또는 instrumented pull test로 특성화
* payload restraint 검증
* 최소 corner load reserve 확보

가능하면

* 기준 산업용 바닥
* 더 낮은 마찰의 균일한 검증 surface

두 조건만 사용하십시오.

바닥 마찰을 정량화하지 않고 “high/low friction floor”라고 부르는 것은 허용되지 않습니다.

## E-3. Command profiles

외부에서 보낼 수 있는 것은 scalar packet뿐입니다. 따라서 모든 arm 조건에서 packet sequence를 byte-equivalent하게 유지해야 합니다.

2WIS primary에는 두 개의 저속 S-curve를 사용하십시오.

* profile V1: 승인된 최저 반복 가능 속도
* profile V2: V1보다 큰 속도 또는 acceleration이지만 vendor·mount 승인 범위 내

초기 후보로 0.1–0.2 m/s 수준을 사용할 수 있지만 이는 제조사 안전한계가 아니라 pilot 후보일 뿐입니다.

Pivot에서는 raw scalar speed가 실제 yaw rate와 어떤 관계인지 먼저 식별한 뒤, 두 개의 measured yaw-rate band로 보고하십시오. scalar command를 곧바로 rad/s라고 간주하면 안 됩니다.

## E-4. Baselines

### 2WIS primary

| 조건 | arm start/end/path                         | 변경 요소                   |
| -- | ------------------------------------------ | ----------------------- |
| H  | picking pose에서 hold                        | arm effect 없음           |
| Q  | low/rear compensated posture로 이동 후 hold    | quasi-static load shift |
| N  | 동일 picking-to-stow path, 고정 nominal timing | naive concurrent        |
| A  | N과 동일 path, authority-gated timing         | 제안법                     |

primary comparison은

[
A\quad \text{vs.}\quad N
]

입니다.

(A) 대 H를 primary로 두면 concurrency와 posture 변화가 모두 섞입니다.

### Pivot negative control

* neutral posture
* rear/low posture
* left/right load-biased posture
* slow common-mode arm trajectory
* fast common-mode trajectory

모두 같은 pivot base packet을 사용합니다.

Pivot에서 평가할 것은 “geometry residual 제거”가 아니라

* effective center drift
* load-weighted residual distribution
* yaw-rate attenuation
* support margin
* trajectory endpoint error

입니다.

## E-5. Ablations

1. posture-only
   팔을 옮긴 뒤 정지시켜 quasi-static 효과만 평가합니다.

2. slow versus fast
   같은 geometric path를 다른 duration으로 실행하여 reaction effect를 분리합니다.

3. phase shift
   path와 duration은 같고 base command 대비 start offset만 변경합니다.

4. common versus differential
   aggregate CoM 이동과 yaw reaction을 분리합니다.

5. gate off
   uncertainty lower bound 없이 point-estimate만 사용하는 scheduler와 비교합니다.

6. timing-shuffled
   선택된 timing의 효과가 session drift 또는 우연인지 확인합니다.

## E-6. Session 구조

최소 구조는 다음입니다.

### Identification

* 독립 3 session
* P0–P2
* 2WIS와 Pivot
* arm condition당 최소 4–5 randomized matched repetitions

### Held-out confirmation

* 독립 2 session
* held-out P3
* 2WIS primary
* A/N/Q 조건
* profile당 최소 8 matched blocks

최종 표본수는 pilot paired variance로 계산하십시오.

[
n
\ge
\left(
\frac{(1.96+0.84)\sigma_d}
{\delta^\star}
\right)^2.
]

여기서 practical effect threshold는

[
\delta^\star
============

\max
\left(
0.10J_{\mathrm{baseline}},
3\sigma_{\mathrm{repeat}}
\right)
]

정도로 사전등록할 수 있습니다. 10%는 안전한계가 아니라 practical-effect floor입니다.

## E-7. 첫 주 falsification test

| 일자    | 작업                                                       | 실패 시 의미                               |
| ----- | -------------------------------------------------------- | ------------------------------------- |
| Day 1 | steering-axis 위치, wheel plane, track, wheel radius 광학 측정 | manual geometry를 그대로 사용할 수 없음         |
| Day 2 | 2WIS/4WIS/Pivot 실제 angle, latency, hysteresis 측정         | steering residual 분리가 불가능             |
| Day 3 | arm 고정, 무저속 Pivot 반복. ICR gap과 (J_{\mathrm{geom}}) 측정    | pivot hypothesis 미검증                  |
| Day 4 | arm 고정, 2WIS 일정 곡률·두 acceleration profile                | rear-slip 가설의 base evidence 확보        |
| Day 5 | arm posture별 static corner load와 CoM 측정                  | 팔 load-shift authority가 noise 이하이면 중단 |
| Day 6 | 동일 packet에서 Q/N/A low-speed matched trial                | arm effect 부호·크기 확인                   |
| Day 7 | session-blocked model과 go/no-go memo                     | claim 동결                              |

## E-8. Go/No-Go 기준

### Geometry gate

[
|\theta^\star-\theta_{\mathrm{pivot}}|

>

3\sigma_\theta
]

그리고

[
\sqrt{J_{\mathrm{geom}}}

>

3\sigma_{e_\perp}.
]

이를 만족하지 못하면 “structural pivot residual” claim을 중단합니다.

### 2WIS diagnosis gate

command+steering-only model (M_0)에 wheel residual과 payload/load 변수를 추가한 (M_1)이 held-out trajectory error를

[
\max(10%,3\sigma_{\mathrm{CV}})
]

이상 줄여야 합니다.

그렇지 않으면 slip decomposition은 과잉 모델링입니다.

### Arm-authority gate

[
\Delta J_{\mathrm{excess}}
==========================

## J_{\mathrm{excess}}^{N}

J_{\mathrm{excess}}^{A}
]

에 대해

* discovery 3/3 session에서 같은 부호
* held-out one-sided 95% lower bound (>0)
* improvement (>\max(10%,3\sigma_{\mathrm{repeat}}))
* safety endpoint 악화 없음

을 모두 만족해야 합니다.

### Prediction gate

* useful/negligible/harmful class 정확도 ≥80%
* false-safe 0회
* held-out payload에서 effect sign 유지

### 즉시 중단 조건

* manufacturer-approved arm/mount dynamic limit 미확보
* wheel unloading 또는 contact-loss 징후
* E-stop, PCU timeout, CAN fault
* mount loosening·resonance
* payload displacement
* steering angle ground truth 불확실
* external tracking과 RS-232/CAN sync 실패

현재 manufacturer dynamic limit은 unknown입니다. 임의의 torque·temperature percentage를 안전기준으로 만들면 안 됩니다.

---

# F. 팔이 Pivot geometry mismatch를 고칠 수 있는가

## 결론

고칠 수 없습니다.

팔은 다음 값을 바꾸지 못합니다.

[
L,\quad W_k,\quad \delta_{FL},\delta_{FR},\delta_{RL},\delta_{RR}.
]

따라서 wheel normal line의 비동시성을 없앨 수 없습니다.

팔이 바꿀 수 있는 것은 다음입니다.

* corner normal-load distribution
* total CoM 위치와 높이
* yaw inertia
* 순간적인 chassis yaw reaction
* 실제 yaw rate
* effective ICR의 least-squares compromise
* 어느 wheel이 더 많은 scrub을 담당하는지
* support margin
* trajectory endpoint drift

팔로 특정 wheel을 unload하면 그 wheel의 scrub force와 마찰 손실이 줄 가능성은 있습니다. 그러나 다음 trade-off가 생깁니다.

* wheel unloading이 지나치면 stability와 drive authority가 악화됩니다.
* 다른 wheel에 residual과 load가 집중됩니다.
* arm reaction의 return payback이 생깁니다.
* 전체 geometry floor는 남습니다.

## Supervisor가 실제로 할 수 있는 것

| 조치                                  |    가능 여부 | 설명                                        |
| ----------------------------------- | -------: | ----------------------------------------- |
| scalar speed derating               |       가능 | Pivot yaw rate를 낮춰 residual을 선형적으로 줄임     |
| acceleration ramp 완화                |       가능 | 20 Hz SPEED profile을 부드럽게 구성              |
| mode selection                      |       가능 | 2WIS/4WIS/Pivot 중 선택                      |
| arm posture·timing                  |       가능 | load distribution과 yaw inertia 변경         |
| Pivot 내부 curvature command          |   문서상 불가 | Pivot steering은 fixed cross configuration |
| per-wheel speed allocation          |       불가 | 외부 command가 없음                            |
| per-wheel steer correction          |       불가 | 외부 command가 없음                            |
| wheel torque redistribution         |       불가 | 외부 command가 없음                            |
| 2WIS/4WIS curvature precompensation | 제한적으로 가능 | scalar STEER mapping이 반복 가능할 때만           |
| Pivot 대신 multi-arc maneuver         |  가능하며 권장 | 2WIS/4WIS arc와 translation을 조합            |

Pivot이 정확한 중심 회전을 제공하지 못한다면 가장 정직한 해결은 팔을 더 세게 흔드는 것이 아니라 다음 중 하나입니다.

1. 낮은 yaw-rate Pivot
2. 짧은 Pivot + translation correction
3. 4WIS multi-arc
4. 2WIS three-point maneuver
5. desired final pose를 기준으로 command duration·sequence 보정

즉, arm scheduler보다 mode/maneuver supervisor가 더 큰 trajectory 효과를 낼 가능성이 높습니다. 이를 baseline으로 포함하지 않으면 Reviewer는 “팔이 아니라 기동 방식을 바꾸면 되는 문제”라고 반박합니다.

---

# G. 2026-09-15 IEEE Access 원고

## 제목

KALO-H: Separating Structural Steering Scrub from Bimanual Load-Transfer Authority in a Closed-Interface 4WID–4WIS Mobile Manipulator

가속·traction·new 4WIS control이라는 단어는 제목에서 제외해야 합니다.

## 한 문장 claim

> Using synchronized external motion, wheel-speed and steering feedback, payload mass properties, and arm states under identical scalar base packets, we separate the irreducible rolling residual imposed by steering geometry from the condition-dependent excess residual, and demonstrate that bimanual posture and timing are gated only on the latter.

한국어로는 다음입니다.

> 동일한 scalar base packet에서 외부 운동, 휠 속도·조향 feedback, payload 질량특성 및 arm state를 동기화하여 조향 기하가 부과하는 제거 불가능한 rolling residual과 조건 의존 excess residual을 분리하고, 양팔 posture와 timing은 후자에 대해서만 적용합니다.

## Primary endpoint

2WIS held-out 조건에서

[
\Delta J_{\mathrm{primary}}
===========================

## J_{\mathrm{excess}}^{N}

J_{\mathrm{excess}}^{A}
]

를 primary로 둡니다.

여기서

[
J_{\mathrm{excess}}
===================

\frac1T
\int_0^T
\left[
\sum_i e_{\perp,i}^2
--------------------

J_{\mathrm{geom}}(\dot\psi,\delta)
\right]dt.
]

양수일수록 authority-gated timing이 naive timing보다 controllable residual을 줄였다는 뜻입니다.

Pivot에서는 (J_{\mathrm{geom}})이 positive이고 arm posture에 의해 제거되지 않는다는 것이 mechanism gate입니다. Pivot trajectory drift와 support margin은 secondary입니다.

## Contribution 세 개

1. A geometry-grounded decomposition of wheel rolling residual into a steering-limit floor, steering-realization error, and condition-dependent excess residual for a 20-Hz scalar-command 4WID–4WIS platform.

2. A synchronized hardware measurement and uncertainty pipeline combining external body twist, individual wheel speed and steering feedback, reconstructed total CoM, payload mass properties, arm states, and optional dynamic corner loads without assuming wheel-level command access.

3. A randomized, session-held-out evaluation showing whether identical-path bimanual posture and timing can reduce the excess residual in 2WIS operation, while Pivot serves as a structural negative control demonstrating what the arms cannot correct.

## 반드시 들어가야 할 비교군

* arm held
* best fixed quasi-static posture
* naive concurrent
* authority-gated
* scalar speed derating
* multi-arc or non-Pivot maneuver baseline
* offline oracle timing

speed derating과 multi-arc가 빠지면 “왜 팔을 써야 하는가”에 답하지 못합니다.

## RA-L로 올리기 위한 추가 요구

* dynamic corner-load 또는 full force-plate evidence
* 최소 3–4 payload mass/offset/height configurations
* 2WIS와 Pivot 외에 held-out 4WIS
* gray-box reduced-order model
* cross-payload·cross-mode prediction
* calibrated prediction intervals
* best static posture, speed derating, multi-arc, oracle baseline
* 5개 이상 독립 session
* full-cycle arm return/payback
* 영상이 아닌 본문 수치로 load와 residual mechanism 증명

영상에서 “덜 미끄러져 보인다”는 것은 slip, traction, ICR 또는 stability 증거가 아닙니다.

## T-RO 수준 요구

* bounded steering geometry에서 irreducible residual의 일반 정리
* hidden wheel controller 아래의 authority identifiability 조건
* 20 Hz feedback에서 residual을 식별할 수 있는 observability 분석
* model uncertainty와 timing uncertainty를 포함한 robust 또는 chance guarantee
* second commercial platform
* 다른 steering architecture에서 cross-platform validation
* mass ratio, geometry ratio, steering-limit ratio에 대한 dimensionless scaling
* platform을 바꿔도 useful/negligible/harmful class를 예측

이 요구는 2026-09-15까지 현실적으로 달성할 수 없습니다.

---

# H. Hostile Reviewer 2 의견과 최종 판정

> The manuscript correctly observes that ROMO-B’s fixed ±30° cross-steering configuration is incompatible with a zero-slip centered pivot for the documented steering-axis geometry. However, this is a platform-specific steering limitation, not a new mobile-manipulation principle. The proposed arm motions cannot make the four wheel-normal lines concurrent and therefore cannot eliminate the structural scrub component. At most, the arms can redistribute wheel loads, alter yaw inertia, change the least-squares body twist, or reduce a load-dependent excess residual. These effects are already closely surrounded by prior work on mobile-manipulator tire slip, arm-based wheel-load redistribution, dynamic weight shifting, tip-over avoidance, and 4WIS ICR control. The paper would remain unpublishable if it claims traction improvement, pivot correction, or a new 4WIS controller without direct wheel-force evidence and wheel-level authority. A defensible IEEE Access paper requires a preregistered decomposition of structural and controllable residuals, synchronized external ground truth, identical base packets, direct or validated load measurements, held-out payload/session prediction, and Pivot used as a negative control rather than a success case. Without those elements, the work is merely calibration of one vendor platform.

최종 범주 판정은 다음과 같습니다.

현재 아이디어 그대로: platform-specific calibration exercise

수정된 geometry/residual decomposition + held-out authority study: publishable scientific question for IEEE Access, 조건부

RA-L 이상: 현재는 NO

가장 냉정한 한 문장 결론은 다음입니다.

> Pivot의 9.72° 구조 오차는 논문을 자동으로 강하게 만드는 발견이 아니라, 팔이 고칠 수 없는 항과 고칠 가능성이 있는 항을 분리하도록 강제하는 falsification device이며, 그 분리를 실제 force·motion·held-out 데이터로 증명하지 못하면 KALO-H는 여전히 특정 ROMO-B의 보정 실험에 불과합니다.

[1]: https://www.cambridge.org/core/journals/robotica/article/abs/modeling-and-control-of-a-mobile-manipulator/BF97FA07141F1DC6FF16C93CFDF74ADE "Modeling and control of a mobile manipulator | Robotica | Cambridge Core"
[2]: https://colab.ws/articles/10.1109%2Frobio.2011.6181521?utm_source=chatgpt.com "Dynamics modeling of a mobile manipulator for wheel slip avoidance | CoLab"
[3]: https://ouci.dntb.gov.ua/en/works/9jeeOeVl/?utm_source=chatgpt.com "Slip Analysis for a Wheeled Mobile Manipulator"
[4]: https://ieeexplore.ieee.org/document/10629175?utm_source=chatgpt.com "Instantaneous Center of Rotation Tracking Control of Four-Wheel Independent Steering Vehicles Under Large-Curvature Turning Conditions"
[5]: https://www.researchgate.net/publication/303889789_Dynamic_Weight-Shifting_for_Improved_Maneuverability_and_Rollover_Prevention_in_High-Speed_Mobile_Manipulators?utm_source=chatgpt.com "Dynamic Weight-Shifting for Improved Maneuverability and Rollover Prevention in High-Speed Mobile Manipulators | Request PDF"
[6]: https://doaj.org/article/0fd424f5e7ae4829963f7659695e51c2?utm_source=chatgpt.com "Online Dynamic Tip-Over Avoidance for a Wheeled Mobile Manipulator With an Improved Tip-Over Moment Stability Criterion – DOAJ"
[7]: https://doaj.org/article/22a3debd753540c38220707d67ee3359?utm_source=chatgpt.com "Kinematics, dynamics and control design of 4WIS4WID mobile robots – DOAJ"
