# ROMO-B 플랫폼 매뉴얼 검증 통합본

- 원본: `Scan_20260717_154609.pdf`
- 원본 분량: 38쪽
- 목적: Ubuntu 22.04, ROS 2 Humble, USB to RS232, Livox Mid-360, Autoware 연동을 위한 매뉴얼 핵심 내용의 텍스트화
- 작성 기준: 스캔 페이지를 직접 판독하고, 기존 추출본의 오류와 누락을 재검증하여 반영
- 문서 상태: 검증 개정본
- 주의: 이 문서는 원본 매뉴얼을 대체하는 법적 또는 제조사 공식 문서가 아니다. 실제 구동 전에는 실물과 제조사 자료를 함께 확인해야 한다.

---

# 1. 검증 결과 요약

## 1.1 그대로 사용해도 되는 핵심 내용

다음 내용은 원본 매뉴얼과 일치한다.

- 플랫폼은 4개 바퀴 독립 구동, 4개 바퀴 독립 조향 구조다.
- 조향 모드는 2WIS, 4WIS, Pivot 세 가지다.
- 상위제어기 HLV와 플랫폼 제어기 PCU는 RS232로 통신한다.
- 매뉴얼에 공개된 외부 상위제어기 인터페이스는 CAN이 아니라 RS232다.
- 통신 속도는 115200 bps다.
- Parity는 None, Stop bit는 1이다.
- 제어 주기는 50 ms, 즉 20 Hz다.
- HLV에서 PCU로 보내는 패킷은 13바이트다.
- PCU에서 HLV로 보내는 패킷은 25바이트다.
- HLV에서 PCU 방향의 2바이트 수치는 Big Endian이다.
- PCU에서 HLV 방향의 2바이트 수치는 Little Endian이다.
- 속도 명령은 `m/s × 100`이다.
- 속도 피드백은 `raw × 0.01 m/s`다.
- 조향 명령은 문서상 `1 raw = 1 deg`다.
- 조향 피드백은 `raw × 0.1 deg`다.
- ALIVE는 0부터 255 범위에서 매 주기 1씩 증가한다.
- 상위제어기 통신이 약 0.3초 동안 정상 갱신되지 않으면 Auto Fail 조건이 된다.

## 1.2 기존 추출본에서 수정한 사항

다음 사항을 수정하거나 불확실성으로 명시했다.

1. 축거는 대부분의 페이지에서 323 mm지만 한 페이지에서 324 mm로 표기되어 있다.
2. 4WIS 일부 인쇄 수식에는 `R = L / tan(delta)`로 보이는 오류가 있다.
3. 4WIS 예시값과 기하 구조에 맞는 관계는 `R = (L/2) / tan(delta)`다.
4. 우측 컨트롤 패널 설명에 Serial Port를 포함했다.
5. 원격 조종기의 Fail/Safe 기능을 추가했다.
6. Auto와 HLV E-Stop 조건은 원문에 `0에서 1로 변경`이라고 적혀 있어 상승 에지 여부를 실물 검증 항목으로 남겼다.
7. 32쪽 전원 절차의 어색한 `On` 문구를 원문 불일치 항목으로 분리했다.
8. SPEED와 STEER의 음수 표현을 signed int16 two's complement로 보는 것은 구현상 매우 유력한 해석이지만 원문에 자료형 명칭은 없다.
9. ALIVE의 255 다음 0 순환은 일반적인 구현으로 추정되지만 원문에 명시되지 않았다.
10. `DIS-CON`은 통신 미연결 또는 단절을 뜻하며 전기적 단락을 뜻하지 않는다.
11. RS232 Data bits, DB9 핀 배열, straight/null-modem 여부는 매뉴얼에 없다.
12. Auto Fail에서 단순 속도 0인지 Full Braking인지 페이지 간 설명이 일치하지 않는다.

---

# 2. 플랫폼 개요

ROMO-B는 사용 목적에 맞춰 센서, 로봇팔, 디지털 사이니지, 적재 모듈 등을 상판에 장착할 수 있는 다목적 모바일 로봇 플랫폼이다.

매뉴얼에서 사용하는 핵심 구동 형식은 다음과 같다.

```text
4WID-4WIS

4WID: Four-Wheel Independent Drive
4WIS: Four-Wheel Independent Steering
```

즉, 네 바퀴가 각각 독립적으로 구동되고 각각 독립적으로 조향된다.

플랫폼은 다음 조향 및 이동 형태를 지원한다.

- 2WIS: 전륜 조향
- 4WIS: 전륜과 후륜을 함께 사용하는 4륜 조향
- Pivot: 제자리 회전에 가까운 회전
- 개요에서는 Crab 계열 이동 가능성도 언급되지만, 공개 RS232 프로토콜의 조향 모드 값은 2WIS, 4WIS, Pivot 세 가지뿐이다.

---

# 3. 전체 하드웨어 및 제어 구조

## 3.1 상위 구조

매뉴얼의 제어 구조도를 텍스트로 재구성하면 다음과 같다.

```text
 LiDAR ─────────┐
 GPS ───────────┤
 Camera ────────┼────> High-Level Controller, HLV
 Ultrasonic ────┘                 │
                                  │ RS232, ch4
                                  ▼
                        Platform Control Unit, PCU
                                  │
          ┌───────────────────────┼────────────────────────┐
          │                       │                        │
          │ RS232 ch1             │ RS232 ch2              │ RS232 ch3
          ▼                       ▼                        ▼
 BLDC Motor Driver #1      BLDC Motor Driver #2          LCD Panel
          │                       │
          ├─ In-wheel Motor #1    ├─ In-wheel Motor #2
          └─ In-wheel Motor #4    └─ In-wheel Motor #3

 PCU ── PWM ──> Steering Servo FL
 PCU ── PWM ──> Steering Servo FR
 PCU ── PWM ──> Steering Servo RL
 PCU ── PWM ──> Steering Servo RR

 Remote transmitter ))) RF ((( Remote receiver
                                  │
                                  └── PWM channels ──> PCU

 Body E-Stop switch ── DIO ──> PCU
```

모터 번호와 실제 FL, FR, RL, RR 대응은 배선 또는 제조사 테스트 프로그램으로 다시 확인하는 것이 안전하다.

## 3.2 HLV

HLV는 High-Level Controller의 약자다.

사용하려는 구성에서는 Ubuntu 22.04 ROS 2 컴퓨터가 HLV 역할을 한다.

HLV의 역할:

- Livox Mid-360 데이터 수신
- 위치 추정
- 지도 처리
- 경로 계획
- 장애물 인식
- Autoware 제어 명령 생성
- 차량 중심 속도 명령 생성
- 등가 조향각 명령 생성
- Auto/Manual 명령 생성
- E-Stop 명령 생성
- 조향 모드 명령 생성
- ALIVE 갱신
- RS232 명령 패킷 송신
- 플랫폼 상태 피드백 수신

## 3.3 PCU

PCU는 Platform Control Unit의 약자다.

매뉴얼상 PCU의 메인 제어기는 National Instruments myRIO-1900이다.

PCU의 역할:

- HLV와 RS232 통신
- 원격 조종기 입력 처리
- 본체 E-Stop 처리
- Auto/Manual 상태 관리
- 조향 모드 관리
- Ackermann 기반 바퀴별 목표값 계산
- 4개 인휠 BLDC 모터 제어
- 4개 조향 서보 제어
- LCD 상태 표시
- 각 바퀴 속도 피드백 송신
- 각 바퀴 조향각 피드백 송신
- PCU ALIVE 송신

## 3.4 ROS 2 기반 적용 구조

```text
Livox Mid-360
      │ Ethernet
      ▼
Ubuntu 22.04
ROS 2 Humble
Autoware
      │
      ▼
ROMO-B ROS 2 Vehicle Interface
      │ USB
      ▼
USB to RS232 변환기
      │ RS232
      ▼
ROMO-B PCU
      │
      ├── BLDC Motor Driver
      ├── Steering Servo
      ├── LCD
      ├── Remote receiver
      └── E-Stop
```

---

# 4. 외형 및 기계 구조

## 4.1 주요 외형 치수

| 항목 | 매뉴얼 표기 |
|---|---:|
| 길이 | 756 mm |
| 전체 너비 | 597 mm |
| 높이 | 342 mm |
| 지상고 | 62 mm |
| 축거 | 대부분 323 mm |
| 축거 예외 표기 | 한 페이지 324 mm |
| 외형도 또는 사양표의 윤거 | 529 mm |
| 조향 계산에 사용하는 회전 중심 간 거리 | 390 mm |
| 플랫폼 무게 | 55 kg |
| 허용 Payload | 100 kg |

## 4.2 축거 323 mm와 324 mm

원본 매뉴얼의 반복 표기는 다음과 같다.

- 사양표: 323 mm
- 외형 치수도: 323 mm
- 조향 제어 알고리즘: `L = 0.323 m`
- 제조사 테스트 프로그램: `L = 0.323 m`
- 일부 조향 설명: 0.324 m

따라서 초기 제어 파라미터는 다음이 합리적이다.

```text
L = 0.323 m
```

다만 최종 URDF, Autoware 차량 파라미터, 오도메트리 계산에는 실물의 전후 조향축 중심 간 거리를 측정해 확정해야 한다.

## 4.3 529 mm와 390 mm

매뉴얼은 두 종류의 좌우 거리를 사용한다.

```text
529 mm: 외형도 또는 일반 사양표에서 윤거로 표기된 값
390 mm: 좌우 조향 너클 회전 중심 사이 거리, 조향 계산에 사용하는 값
```

Ackermann 계산의 초기 파라미터는 다음과 같다.

```text
L = 0.323 m
W = 0.390 m
```

URDF의 실제 바퀴 중심 위치는 실측값을 사용해야 한다.

## 4.4 상부 구조

상판은 알루미늄이며 다수의 장착 홀이 있다.

장착 대상으로 언급되거나 도식에 표시된 구성:

- LiDAR
- GPS
- 카메라
- 초음파 센서
- 로봇팔
- 디지털 사이니지
- 사용자 정의 센서 모듈
- 사용자 정의 상부 장치

상판 또는 상부에는 플랫폼 통신 포트와 배터리 전원 출력 관련 포트가 표시되어 있다.

## 4.5 전면 구조

전면에는 센서 장착을 위한 구조가 있다.

- 전방 장애물 인식용 LiDAR 장착 공간
- 초음파 센서 장착 구조
- 전면 커버 내부 또는 마운트 부

Livox Mid-360은 360도 수평 시야를 사용하므로, 매뉴얼의 전면 센서 공간보다 상판 중앙 또는 차폐가 적은 상부 위치가 더 적합할 수 있다. 이는 매뉴얼 명시가 아니라 시스템 설계 권장사항이다.

## 4.6 하부 구조

- 4개 인휠 구동 모터
- 4개 독립 조향 모듈
- 조향 링크 및 Tie rod
- Dual A-arm 현가 구조
- Shock absorber 없음

Shock absorber가 없으므로 노면 진동이 LiDAR와 IMU에 직접 전달될 가능성이 있다.

---

# 5. 기본 구성품

매뉴얼의 기본 구성품:

1. ROMO-B 플랫폼
2. ROMO-B 배터리 충전기
3. AC 전원선
4. 수동 제어용 원격 조종기

USB to RS232 변환기가 기본 구성품에 포함되는지는 문서만으로 확실하지 않다.

---

# 6. 플랫폼 주요 사양

## 6.1 플랫폼

| 항목 | 사양 |
|---|---|
| 구동 방식 | 4WID-4WIS |
| 최대 속도 | 1.5 m/s |
| 환산 최대 속도 | 5.4 km/h |
| 연속 운전 시간 | 최대 약 3시간 |
| 플랫폼 무게 | 55 kg |
| 허용 적재량 | 100 kg |
| 메인 프레임 | 알루미늄 |
| 상판 | 알루미늄 |
| 외장 커버 | PLA |
| 현가장치 | Dual A-arm |
| Shock absorber | 없음 |

연속 운전 시간은 운용 조건, 적재량, 노면, 속도에 따라 달라질 수 있다.

## 6.2 인휠 BLDC 모터

바퀴당 사양으로 읽히는 항목:

| 항목 | 사양 |
|---|---:|
| 형식 | In-wheel type BLDC motor |
| 정격 출력 범위 | 100 W에서 350 W |
| 극수 | 30 poles |
| 정격 전압 | 24 V에서 48 V |
| 정격 전류 | 8 A |
| 정격 토크 | 약 8 N m |
| 최대 토크 | 약 16 N m |
| 바퀴 직경 | 약 206 mm |
| 엔코더 분해능 | 4096 PPR |
| 모터 무게 | 약 4 kg |
| 정격 회전속도 | 200 rpm |
| 최대 회전속도 | 300 rpm |
| 200 rpm 기준 선속도 표기 | 약 2 m/s |

플랫폼 전체의 소프트웨어 제어 범위는 최대 1.5 m/s다.

## 6.3 조향 서보 모터

바퀴당 사양:

| 항목 | 사양 |
|---|---:|
| 입력 전압 | 24 V |
| 무부하 동작 속도 | 약 0.37 s / 60 deg |
| 정격 토크 | 약 140 kg cm |
| 최대 토크 | 약 400 kg cm |
| 무부하 전류 | 약 0.1 A |
| Stall 전류 | 약 14 A |
| 입력 신호 | PWM |
| PWM 범위 | 약 800 us에서 2200 us |
| PWM 주파수 | 약 50 Hz에서 400 Hz |
| 모터 무게 | 약 765 g |
| 개별 조향 가능 범위 | -30 deg에서 +30 deg |

조향 부호:

```text
음수: 좌조향
양수: 우조향
```

---

# 7. 배터리 및 충전

## 7.1 배터리

| 항목 | 사양 |
|---|---:|
| 종류 | LiFePO4 |
| 에너지 | 614 Wh |
| 표기 용량 | 24 V / 24 Ah |
| 명목 전압 | 25.6 V |
| 최대 전압 | 27.2 V |
| 최소 전압 | 23 V |
| 배터리 무게 | 5.3 kg |
| 작동 온도 | -20 degC에서 45 degC |

매뉴얼상 운용 기준:

- 완충 전압: 약 27.2 V
- 충전 필요 기준: 23 V 이하
- 23 V 이하가 표시되면 플랫폼 전원을 끄고 충전

## 7.2 충전기

| 항목 | 사양 |
|---|---:|
| 배터리 종류 | LiFePO4용 |
| 충전 전압 | 29.2 V |
| 최대 충전 전류 | 14 A |
| 완전 방전 후 완충 시간 | 약 90분 |
| 플랫폼 측 커넥터 | XT60로 표시 |

충전기 LED 상태:

| 상태 | LED1 | LED2 |
|---|---|---|
| 충전 중 | 빨간색 | 빨간색 |
| 완충 | 빨간색 | 초록색 |

## 7.3 충전 절차

1. 플랫폼 메인 전원을 끈다.
2. 충전 포트 커버를 연다.
3. 극성을 확인한다.
4. 충전 커넥터를 연결한다.
5. 충전기 AC 전원을 연결한다.
6. LED 상태를 확인한다.
7. 완충 후 충전기 AC 전원을 먼저 해제한다.
8. 플랫폼 충전 커넥터를 분리한다.
9. 충전 포트 커버를 닫는다.

---

# 8. 우측 컨트롤 패널

매뉴얼에서 확인되는 주요 항목:

- Power ON/OFF Switch
- E-Stop Switch
- LCD Panel
- Voltmeter
- Charging Port
- Serial Port

## 8.1 Power ON/OFF

플랫폼 메인 전원을 켜고 끈다.

중요한 전원 순서:

```text
시작:
원격 조종기 ON
→ 플랫폼 메인 전원 ON

종료:
플랫폼 메인 전원 OFF
→ 원격 조종기 OFF
```

## 8.2 E-Stop

본체의 E-Stop 버튼은 누르면 고정되는 Push-lock 형태다.

해제할 때는 버튼을 회전하여 복귀시키는 구조로 설명되어 있다.

주의:

- E-Stop은 플랫폼 전원이 켜져 있을 때 전기적 제동을 수행한다.
- 플랫폼 전원이 꺼지면 모터가 Free 상태가 될 수 있다.
- 전원 OFF를 주차 브레이크로 간주하면 안 된다.

## 8.3 LCD

16 x 2 LCD로 플랫폼 상태를 표시한다.

표시 내용:

- HLV 통신 상태
- Auto/Manual 상태
- E-Stop 상태
- 조향 모드
- Auto Fail 상태

## 8.4 Voltmeter

배터리 전압을 표시한다.

- 완충 상태 예시: 약 27.2 V
- 충전 필요 기준: 23 V 이하

## 8.5 Serial Port

상위제어기와 플랫폼 간 RS232 연결에 사용하는 포트가 표시된다.

다만 다음 사항은 매뉴얼에 없다.

- DB9 핀 번호
- TX 핀
- RX 핀
- GND 핀
- straight cable 여부
- null-modem cable 여부
- 상판 통신 포트와 패널 Serial Port의 차이

## 8.6 Charging Port

충전기 연결용 포트다.

충전 중에는 전압이 높은 상태일 수 있으므로 포트에 금속 이물질이 들어가지 않도록 주의해야 한다.

---

# 9. 원격 조종기

원격 조종기는 Manual 주행뿐 아니라 Auto 진입과 E-Stop 조건에도 사용된다.

## 9.1 주요 스위치 및 스틱

- Power ON/OFF
- E-Stop
- Auto/Manual
- Steering mode
- Throttle
- Steering
- Trim 1
- Trim 2

## 9.2 조향 모드

- 2WIS
- 4WIS
- Pivot

## 9.3 Manual 주행

Manual 상태에서는 원격 조종기의 스로틀과 조향 스틱으로 주행한다.

매뉴얼 권장사항:

- 출발할 때 스로틀을 급격히 올리지 않는다.
- 초기 출발은 약 25% 이하에서 천천히 수행한다.
- 스틱 중립 주변에는 Dead band가 있다.

## 9.4 Fail/Safe 기능

원격 조종기에는 Fail/Safe 기능이 있다고 설명되어 있다.

- 플랫폼 전원이 켜진 상태에서 원격 조종기 전원이 꺼지면 입력값이 중립 상태로 변경되는 구조다.
- Fail/Safe 기능이 항상 정상 작동한다고 가정하지 말고 플랫폼 사용 전 조종기 전원과 상태를 확인해야 한다.
- 원격 조종기 배터리 상태도 확인해야 한다.

원격 조종기는 자율주행 중에도 안전 인터록의 일부로 취급해야 한다.

---

# 10. 플랫폼 동작 모드

ROMO-B는 다음 상태를 가진다.

```text
Booting
Manual
Auto
Auto Fail
E-Stop
```

## 10.1 Booting Mode

초기 상태:

| 항목 | 값 |
|---|---|
| Speed | 0 m/s |
| Steer | 0 deg |
| Steer mode | 2WIS |
| E-Stop | Off |

부팅 표시:

- 전원을 켠 뒤 약 20초 동안 LCD에서 커서만 표시될 수 있다.
- 이후 약 2초 동안 부팅 완료 메시지가 표시된다.
- 그 다음 현재 통신 및 주행 상태가 표시된다.

## 10.2 Manual Mode

Manual 조건에서는 원격 조종기가 플랫폼을 제어한다.

| Steering mode | 동작 |
|---|---|
| 2WIS | 전륜 조향 |
| 4WIS | 4륜 조향 |
| Pivot | 제자리 회전 계열 |

Manual 모드에서 HLV 명령이 플랫폼을 직접 움직이는 것으로 가정하면 안 된다.

## 10.3 Auto Mode

매뉴얼의 Auto 진입 조건:

1. 원격 조종기의 Auto/Manual 스위치가 Auto 위치
2. HLV와 PCU의 통신이 정상
3. HLV에서 PCU로 보내는 패킷의 AorM이 0에서 1로 변경

세 번째 조건은 다른 페이지에서 AorM을 단순 상태값으로 설명한다.

따라서 실제 구현은 다음 절차가 안전하다.

```text
1. AorM = 0, speed = 0, steer = 0으로 통신 시작
2. PCU ALIVE와 정상 피드백 확인
3. 원격 조종기를 Auto 위치로 설정
4. AorM을 0에서 1로 전환
5. 이후 AorM = 1을 지속적으로 송신
```

상승 에지가 반드시 필요한지는 확실하지 않음이다.

## 10.4 Auto Fail Mode

Auto Fail 조건으로 설명된 항목:

- 원격 조종기가 Auto 위치가 아님
- HLV와 PCU 통신 실패
- HLV 패킷 수신 실패
- 패킷 형식 이상
- ALIVE 갱신 정지
- AorM 조건 불충족

Auto Fail 표의 상태:

| 항목 | 상태 |
|---|---|
| Speed | 0 m/s |
| Steer | 0 deg |
| Steer mode | 원격 조종기 설정 |
| E-Stop | Off로 표기된 페이지가 있음 |

문서 불일치:

- Auto Fail 표에는 E-Stop Off가 기재되어 있다.
- 다른 페이지의 E-Stop 진입 조건에는 Auto Fail이 포함되어 있다.

따라서 통신 단절 시 동작이 다음 중 무엇인지는 실물 검증이 필요하다.

- 속도 명령만 0으로 바뀜
- 조향도 0으로 바뀜
- 전기적 Full Braking 실행
- E-Stop 상태로 전환

## 10.5 E-Stop Mode

E-Stop 진입 조건으로 적힌 항목:

1. 원격 조종기 E-Stop On
2. 본체 패널 E-Stop 버튼 누름
3. Auto Fail 실행
4. HLV 패킷의 ESTOP이 0에서 1로 변경

E-Stop 상태:

| 항목 | 상태 |
|---|---|
| Speed | 0 m/s |
| Steer | 0 deg |
| Steering mode | 원격 조종기 설정 |
| E-Stop | On |

HLV ESTOP도 상승 에지가 필요한지, 값이 1인 상태 유지로 충분한지는 확실하지 않음이다.

---

# 11. LCD 표시 해석

## 11.1 통신

| 표시 | 의미 |
|---|---|
| HLV CON | HLV와 PCU 통신 정상 |
| DIS-CON | HLV 통신 미연결 또는 단절 |

`DIS-CON`은 전기적 단락을 뜻하지 않는다.

## 11.2 운전 상태

| 표시 | 의미 |
|---|---|
| MANUAL | 수동 운전 |
| AUTO | 자율 또는 상위제어기 운전 |
| AUTO FAIL | Auto 조건 실패 |

## 11.3 E-Stop 원인

| 표시 | 의미 |
|---|---|
| Re_E-ST | 원격 조종기 E-Stop |
| Bu_E-ST | 본체 버튼 E-Stop |
| Hi_E-ST | HLV E-Stop |
| E-STOP | 복수 E-Stop 조건 또는 일반 E-Stop 표시 |

## 11.4 조향 모드

| 표시 | 의미 |
|---|---|
| 2WIS | 전륜 조향 |
| 4WIS | 4륜 조향 |
| PIV | Pivot |

---

# 12. 조향 범위

## 12.1 개별 바퀴 조향

```text
-30 deg에서 +30 deg
```

## 12.2 HLV 조향 명령 범위

| 모드 | HLV_STEER 명령 범위 |
|---|---:|
| 2WIS | -22 deg에서 +22 deg |
| 4WIS | -18 deg에서 +18 deg |
| Pivot | STEER 값보다 모드와 SPEED 부호가 핵심 |

## 12.3 18 deg 등가 조향각 예시

매뉴얼 예시에서 2WIS와 4WIS의 바퀴별 조향각이 제시된다.

### 2WIS

| 바퀴 | 예시 조향각 크기 |
|---|---:|
| FL | 약 15.2 deg |
| FR | 약 22.0 deg |
| RL | 0 deg |
| RR | 0 deg |

### 4WIS

| 바퀴 | 예시 조향각 크기 |
|---|---:|
| 외측 전륜 | 약 13.1 deg |
| 내측 전륜 | 약 28.1 deg |
| 외측 후륜 | 약 13.1 deg |
| 내측 후륜 | 약 28.1 deg |

4WIS에서 후륜은 전륜과 반대 방향으로 조향한다.

---

# 13. 조향 제어 기하

## 13.1 기호

```text
L = 축거, 초기값 0.323 m
W = 조향 너클 회전 중심 간 거리, 0.390 m
R = 차량 중심 기준 선회 반경
delta = 차량 중심 등가 조향각
v = 차량 중심 선속도
```

조향각 부호는 매뉴얼 기준으로 음수 좌회전, 양수 우회전이다.

---

# 14. 2WIS 기구학

2WIS에서는 전륜만 조향하고 후륜은 0 deg로 유지한다.

차량 중심 선회 반경:

```text
R = L / tan(delta)
```

우회전을 기준으로 내측이 오른쪽이라고 하면:

```text
delta_FR = atan(L / (R - W/2))
delta_FL = atan(L / (R + W/2))
delta_RR = 0
delta_RL = 0
```

바퀴별 선회 반경:

```text
R_FR = sqrt(L^2 + (R - W/2)^2)
R_FL = sqrt(L^2 + (R + W/2)^2)
R_RR = R - W/2
R_RL = R + W/2
```

바퀴별 선속도:

```text
v_FR = v × R_FR / R
v_FL = v × R_FL / R
v_RR = v × R_RR / R
v_RL = v × R_RL / R
```

좌회전에서는 좌우 관계와 부호가 대칭으로 바뀐다.

---

# 15. 4WIS 기구학

4WIS에서는 전륜과 후륜이 반대 방향으로 조향하여 회전 반경을 줄인다.

## 15.1 매뉴얼 인쇄 오류 가능성

일부 식에는 다음 관계가 보인다.

```text
R = L / tan(delta)
```

그러나 4WIS의 기하 구조와 매뉴얼의 18 deg 예시값을 동시에 만족하는 식은 다음이다.

```text
R = (L/2) / tan(delta)
```

검산:

```text
L = 0.323 m
W = 0.390 m
delta = 18 deg

R = (0.323/2) / tan(18 deg)
  ≈ 0.497 m
```

이때 계산되는 바퀴 조향각은 약 28.1 deg와 13.1 deg로 매뉴얼 예시와 일치한다.

따라서 구현에서는 다음 식을 사용해야 한다.

## 15.2 우회전 기준 조향각

```text
delta_FR = +atan((L/2) / (R - W/2))
delta_FL = +atan((L/2) / (R + W/2))

delta_RR = -atan((L/2) / (R - W/2))
delta_RL = -atan((L/2) / (R + W/2))
```

실제 부호 규칙은 PCU 피드백으로 확인해야 한다.

## 15.3 바퀴별 선회 반경

```text
R_FR = R_RR = sqrt((L/2)^2 + (R - W/2)^2)
R_FL = R_RL = sqrt((L/2)^2 + (R + W/2)^2)
```

## 15.4 바퀴별 선속도

```text
v_FR = v × R_FR / R
v_RR = v × R_RR / R
v_FL = v × R_FL / R
v_RL = v × R_RL / R
```

---

# 16. Pivot 모드

Pivot 모드에서는 이론적인 45 deg 대신 조향 모터 한계 때문에 약 30 deg를 사용한다.

## 16.1 바퀴 조향각

| 바퀴 | 조향각 |
|---|---:|
| FL | +30 deg |
| FR | -30 deg |
| RL | -30 deg |
| RR | +30 deg |

도식:

```text
FL  /                 \  FR

          회전 중심

RL  \                 /  RR
```

## 16.2 SPEED 부호와 회전 방향

매뉴얼 테스트 프로그램 설명:

```text
Pivot에서 SPEED 양수: CW
Pivot에서 SPEED 음수: CCW
```

## 16.3 바퀴별 속도 범위

### CW

| 바퀴 | 속도 범위 |
|---|---:|
| FL | 0에서 +1.5 m/s |
| FR | -1.5에서 0 m/s |
| RL | 0에서 +1.5 m/s |
| RR | -1.5에서 0 m/s |

### CCW

| 바퀴 | 속도 범위 |
|---|---:|
| FL | -1.5에서 0 m/s |
| FR | 0에서 +1.5 m/s |
| RL | -1.5에서 0 m/s |
| RR | 0에서 +1.5 m/s |

Autoware의 일반 Ackermann 제어 모델은 Pivot을 직접 가정하지 않으므로 초기 자율주행에서는 2WIS를 사용하는 것이 안전하다.

---

# 17. RS232 통신 사양

매뉴얼 38쪽의 명시값:

| 항목 | 값 |
|---|---|
| Interface | RS232 |
| Baud rate | 115200 bps |
| Parity | None |
| Stop bit | 1 |
| Data bits | 문서에 없음 |
| Cycle time | 50 ms |
| 송신 빈도 | 20 Hz |
| PCU to HLV byte order | Little Endian |
| HLV to PCU byte order | Big Endian |
| HLV to PCU packet length | 13 bytes |
| PCU to HLV packet length | 25 bytes |

일반적인 `8N1`일 가능성이 높지만 Data bits는 확실하지 않음이다.

---

# 18. HLV에서 PCU로 보내는 명령 패킷

총 13바이트다.

| Offset | 길이 | 필드 | 값 또는 범위 |
|---:|---:|---|---|
| 0 | 1 | S | `0x53` |
| 1 | 1 | T | `0x54` |
| 2 | 1 | X | `0x58` |
| 3 | 1 | AorM | `0x00` Manual, `0x01` Auto |
| 4 | 1 | ESTOP | `0x00` Off, `0x01` On |
| 5 | 1 | STEER_MODE | `0x00` 2WIS, `0x01` 4WIS, `0x02` Pivot |
| 6 | 2 | SPEED | -150에서 +150 |
| 8 | 2 | STEER | -300에서 +300 |
| 10 | 1 | ALIVE | 0에서 255 |
| 11 | 1 | ETX 0 | `0x0D` |
| 12 | 1 | ETX 1 | `0x0A` |

## 18.1 프레임 경계

헤더:

```text
0x53 0x54 0x58
ASCII: S T X
```

일반적인 단일 STX 제어문자 `0x02`가 아니다.

테일:

```text
0x0D 0x0A
CR LF
```

## 18.2 AorM

```text
0x00: Manual
0x01: Auto
```

## 18.3 ESTOP

```text
0x00: E-Stop Off
0x01: E-Stop On
```

## 18.4 STEER_MODE

```text
0x00: 2WIS
0x01: 4WIS
0x02: Pivot
0x03에서 0xFF: neutral 처리로 설명
```

## 18.5 SPEED

매뉴얼 정의:

```text
Command raw = target speed [m/s] × 100
```

예시:

```text
1.00 m/s  → 100
0.50 m/s  → 50
-0.20 m/s → -20
```

범위:

```text
-150에서 +150
= -1.50 m/s에서 +1.50 m/s
```

음수는 후진이다.

## 18.6 STEER

매뉴얼 정의:

```text
Command value 10 deg → result value 10 deg
```

따라서 문서상 명령 스케일은 다음과 같다.

```text
1 raw = 1 deg
```

모드별 실제 사용 범위:

```text
2WIS: -22 deg에서 +22 deg
4WIS: -18 deg에서 +18 deg
```

패킷 필드 자체의 표기 범위는 -300에서 +300이다.

## 18.7 ALIVE

원문:

```text
Increasing each one step, 0에서 255
```

확정 내용:

- 50 ms마다 갱신
- 정상 운용 중 매 주기 1 증가
- 0에서 255 범위

확실하지 않음:

- 255 다음에 0으로 wrap되는지
- PCU가 반복값을 몇 회까지 허용하는지
- 첫 송신값이 반드시 0이어야 하는지

---

# 19. PCU에서 HLV로 보내는 피드백 패킷

총 25바이트다.

| Offset | 길이 | 필드 |
|---:|---:|---|
| 0 | 1 | `0x53` |
| 1 | 1 | `0x54` |
| 2 | 1 | `0x58` |
| 3 | 1 | AorM |
| 4 | 1 | ESTOP |
| 5 | 1 | STEER_MODE |
| 6 | 2 | FL_SPEED |
| 8 | 2 | FR_SPEED |
| 10 | 2 | RL_SPEED |
| 12 | 2 | RR_SPEED |
| 14 | 2 | FL_STEER |
| 16 | 2 | FR_STEER |
| 18 | 2 | RL_STEER |
| 20 | 2 | RR_STEER |
| 22 | 1 | ALIVE |
| 23 | 1 | `0x0D` |
| 24 | 1 | `0x0A` |

2바이트 수치는 Little Endian이다.

## 19.1 바퀴 속도 피드백

```text
speed [m/s] = raw × 0.01
```

예시:

```text
raw 100  → 1.00 m/s
raw -20  → -0.20 m/s
```

매뉴얼 표기 오차:

```text
±5%
```

각 바퀴 범위:

```text
-1.50 m/s에서 +1.50 m/s
```

## 19.2 바퀴 조향각 피드백

```text
steering [deg] = raw × 0.1
```

예시:

```text
raw 100  → 10.0 deg
raw -50  → -5.0 deg
```

매뉴얼 표기 오차:

```text
±1 deg
```

각 바퀴 범위:

```text
-30.0 deg에서 +30.0 deg
```

## 19.3 상태 피드백

- 현재 Auto/Manual 상태
- 현재 E-Stop 상태
- 현재 조향 모드
- PCU ALIVE
- 4개 바퀴 속도
- 4개 바퀴 조향각

---

# 20. 자료형 및 음수 표현

SPEED와 STEER는 각각 2바이트이고 음수 범위를 가진다.

구현상 가장 자연스러운 해석:

```text
signed 16-bit integer
two's complement
```

그러나 원본 매뉴얼에는 `int16_t` 또는 `two's complement`라는 표현이 없다.

따라서 다음 시험이 필요하다.

- +1 raw
- -1 raw
- +10 raw
- -10 raw
- 후진 속도
- 좌조향
- PCU 피드백 바이트 확인

---

# 21. 명령 패킷 예시

다음 예시는 문서 정의를 그대로 적용한 구현 예시다.

조건:

```text
Auto
E-Stop Off
2WIS
0.50 m/s
우조향 10 deg
ALIVE = 1
```

예상 바이트:

```text
53 54 58 01 00 00 00 32 00 0A 01 0D 0A
```

필드 해석:

```text
53 54 58 : Header
01       : Auto
00       : E-Stop Off
00       : 2WIS
00 32    : SPEED 50, Big Endian
00 0A    : STEER 10, Big Endian
01       : ALIVE
0D 0A    : Tail
```

후진과 좌조향의 음수 바이트는 실물 검증 후 확정한다.

---

# 22. 제조사 HLV 테스트 프로그램

매뉴얼에는 Windows와 LabVIEW 기반의 ROMO-B HLV Test Program이 설명되어 있다.

예시 설치 경로:

```text
C:\Program Files (x86)\ROMO-B_V1_4WIS
```

실행 파일 예시:

```text
ROMO-B_HLV_TEST.exe
```

## 22.1 Main 탭

상위제어기 명령 영역:

- HLV_AorM
- HLV_ESTOP
- HLV_STEERMODE
- HLV_STEER
- HLV_SPEED
- HLV_ALIVE
- HLV_Serial Packet

플랫폼 피드백 영역:

- PCU_AorM
- PCU_ESTOP
- PCU_STEERMODE
- PCU_ALIVE
- FL/FR/RL/RR Speed Feed
- FL/FR/RL/RR Steer Feed
- PCU_Serial Packet

## 22.2 Platform Control Algorithm 탭

입력:

- Steering mode
- Vehicle speed
- Equivalent steering angle
- L
- W

출력:

- 각 바퀴 목표 조향각
- 각 바퀴 목표 선속도

## 22.3 정상 연결 확인

1. USB to RS232의 COM 포트를 선택한다.
2. 프로그램 실행 버튼을 누른다.
3. PCU_ALIVE가 계속 증가하는지 확인한다.
4. 각 명령 스위치와 다이얼을 조작한다.
5. 플랫폼 동작과 피드백을 확인한다.

PCU_ALIVE가 증가하지 않을 때 확인:

- COM 포트 번호
- 플랫폼 전원
- USB to RS232 연결
- 플랫폼 Serial Port
- 케이블 종류
- 케이블 접점

---

# 23. 안전 지침

## 23.1 플랫폼 고정

상위제어기 시험 전에는 55 kg 플랫폼을 견딜 수 있는 견고한 받침을 사용하여 네 바퀴를 모두 지면에서 띄운다.

- 불안정한 박스 사용 금지
- 플랫폼이 기울거나 떨어지지 않게 고정
- 조향 시 바퀴가 받침에 닿지 않게 공간 확보
- E-Stop 담당자 배치

## 23.2 경사로

- 별도 주차 브레이크가 없는 것으로 취급한다.
- 전원이 꺼지면 바퀴가 Free 상태가 될 수 있다.
- 경사면에서 고임목 또는 물리적 고정 장치를 사용한다.
- 내리막에서 급가속 또는 급정지하지 않는다.

## 23.3 물과 환경

- 방수 플랫폼으로 간주하면 안 된다.
- 비와 눈에서 사용하지 않는다.
- 젖은 노면에서 사용하지 않는다.
- 플랫폼 내부로 물이 들어가지 않게 한다.

## 23.4 운용 장소

- 차량 통행이 많은 도로에서 사용하지 않는다.
- 사람이 플랫폼 위에 탑승하지 않는다.
- 주행 반경에 사람과 장애물을 두지 않는다.
- 초기 시험은 저속으로 수행한다.

## 23.5 전원 및 배터리

- 원격 조종기를 먼저 켠다.
- 플랫폼 전원을 그다음 켠다.
- 종료 시 플랫폼 전원을 먼저 끈다.
- 원격 조종기를 마지막에 끈다.
- 배터리를 23 V 이하로 과방전하지 않는다.
- 충전은 플랫폼 전원을 끈 상태에서 수행한다.

## 23.6 32쪽 문구 불일치

32쪽에는 Auto Mode를 Off로 만든 뒤 메인 전원을 `On`으로 한다는 문구가 보인다.

문맥상 다음 중 하나일 수 있다.

- 플랫폼을 켜기 전 Auto Mode를 Off로 설정하라는 뜻
- 인쇄 오류
- 특정 초기화 절차

확실하지 않음이다.

전원 종료 절차는 다른 페이지에서 반복적으로 확인되는 다음 순서를 따른다.

```text
플랫폼 전원 OFF
→ 원격 조종기 OFF
```

---

# 24. ROS 2 및 Autoware 연동 시 해석

이 절은 매뉴얼 원문이 아니라 매뉴얼을 기반으로 한 구현 권장사항이다.

## 24.1 초기 조향 모드

```text
STEER_MODE = 0x00
2WIS 사용
```

이유:

- Autoware의 기본 차량 제어는 Ackermann 또는 bicycle model과 가장 잘 맞는다.
- 4WIS와 Pivot은 기본 Autoware 차량 모델과 직접 일치하지 않는다.
- 2WIS가 최소 구현 경로다.

## 24.2 권장 초기 제한

```yaml
control_rate_hz: 20.0
max_velocity_mps: 0.3
max_steering_deg: 5.0
command_timeout_ms: 200
steering_mode: 2WIS
```

실물 검증 뒤 단계적으로 범위를 늘린다.

## 24.3 ROS 2 드라이버 분리

권장 패키지 구조:

```text
romo_b_serial_driver
  ├── serial framing
  ├── packet encoder
  ├── packet decoder
  ├── ALIVE management
  ├── timeout monitor
  └── diagnostics

romo_b_vehicle_interface
  ├── Autoware control input
  ├── speed conversion
  ├── steering conversion
  ├── vehicle status reports
  └── odometry estimation
```

## 24.4 Autoware 입력과 ROMO-B 명령

```text
Autoware target longitudinal velocity
→ SPEED raw = m/s × 100

Autoware tire angle
→ degree 변환
→ 2WIS HLV_STEER 범위로 제한

Autoware engage
→ AorM

Emergency command
→ ESTOP
```

Autoware의 tire angle과 ROMO-B의 등가 중심 조향각이 완전히 같은 정의인지는 실물 경로 시험으로 검증해야 한다.

## 24.5 피드백 사용

피드백으로 생성할 수 있는 ROS 2 정보:

- 차량 속도
- 조향 상태
- Auto/Manual 상태
- E-Stop 상태
- 조향 모드
- 바퀴별 속도
- 바퀴별 조향각
- 진단 정보
- wheel odometry

바퀴별 속도 단순 평균만으로 오도메트리를 계산하면 회전 중 오차가 발생할 수 있으므로 2WIS 기구학을 반영한다.

---

# 25. Livox Mid-360 적용 구조

## 25.1 센서 연결

```text
Livox Mid-360
      │ Ethernet
      ▼
Ubuntu 22.04
      │
      ├── livox_ros_driver2
      ├── PointCloud2
      ├── Localization
      ├── Perception
      └── Autoware
```

## 25.2 장착 권장

- 상판 중앙 또는 차폐가 적은 위치
- 플랫폼 외장보다 높은 위치
- 케이블이 조향 또는 적재물과 간섭하지 않는 위치
- 진동이 적고 강성이 높은 브래킷
- `base_link`와 LiDAR 프레임 간 외부 파라미터 측정
- 바닥 반사와 자체 차체 반사를 고려한 높이 선정

## 25.3 추가 센서 권장

Mid-360 한 대만으로 저속 자율주행을 구성할 수 있으나 다음을 권장한다.

- IMU
- 바퀴 속도 피드백
- 바퀴 조향각 피드백
- 실외 운용 시 RTK-GNSS
- 안정적인 시간 동기화

---

# 26. 실물 검증이 필요한 항목

## 26.1 RS232 전기 및 배선

- Data bits
- DB9 핀 배열
- TX/RX/GND
- straight cable
- null-modem cable
- RS232 전압 레벨
- 상판 포트와 패널 포트 차이

## 26.2 프로토콜

- signed 음수 표현
- ALIVE 255 이후 동작
- 수신 프레임 손실 후 재동기화 방식
- CRC 또는 checksum 부재가 실제로 맞는지
- 송신 주기 허용 오차
- 20 Hz보다 빠른 송신 처리
- 20 Hz보다 느린 송신 timeout
- PCU ALIVE 의미

## 26.3 상태 전환

- AorM 상승 에지 요구 여부
- ESTOP 상승 에지 요구 여부
- Auto 진입 순간 기존 명령 초기화 여부
- Auto Fail 시 제동 방식
- 통신 단절 시 조향 상태
- 원격 조종기 Fail/Safe 실제 동작

## 26.4 조향 및 속도

- STEER 1 raw가 실제로 1 deg인지
- 좌우 조향 부호
- 4WIS 후륜 조향 피드백 부호
- 속도 피드백이 실제 센서값인지 제어기 내부 명령값인지
- 조향 피드백이 실제 센서값인지 목표값인지
- 2WIS 중심 조향각과 Autoware tire angle의 대응
- 실제 최소 회전 반경

---

# 27. 권장 시험 순서

## 27.1 준비

1. 로봇을 견고한 받침으로 들어 올린다.
2. 네 바퀴가 지면 및 받침과 접촉하지 않는지 확인한다.
3. 원격 조종기를 켠다.
4. E-Stop 동작을 확인한다.
5. 플랫폼 전원을 켠다.
6. LCD 부팅을 확인한다.

## 27.2 수신 시험

1. USB to RS232를 연결한다.
2. 포트 권한을 확인한다.
3. 송신하지 않고 수신 데이터부터 기록한다.
4. `53 54 58` 헤더를 찾는다.
5. 25바이트 길이를 확인한다.
6. `0D 0A` 테일을 확인한다.
7. PCU ALIVE 증가를 확인한다.
8. 각 2바이트 필드의 Little Endian 해석을 확인한다.

## 27.3 0 명령 시험

1. SPEED = 0
2. STEER = 0
3. AorM = 0
4. ESTOP = 0
5. STEER_MODE = 0
6. ALIVE 증가
7. 20 Hz 송신

## 27.4 Auto 전환

1. 정상 피드백 확인
2. 원격 조종기 Auto 위치
3. AorM 0에서 1로 전환
4. PCU_AorM 확인
5. LCD AUTO 확인
6. SPEED 0 유지
7. STEER 0 유지

## 27.5 속도 시험

1. +0.05 m/s
2. 0 m/s
3. +0.10 m/s
4. 0 m/s
5. -0.05 m/s
6. 0 m/s

## 27.6 조향 시험

1. 0 deg
2. +1 deg
3. 0 deg
4. -1 deg
5. 0 deg
6. +5 deg
7. 0 deg
8. -5 deg
9. 0 deg

각 단계에서 FL, FR, RL, RR 조향 피드백을 기록한다.

## 27.7 안전 시험

- HLV ESTOP 0에서 1
- 본체 E-Stop
- 원격 조종기 E-Stop
- ALIVE 정지
- USB 케이블 분리
- ROS 2 노드 종료
- 송신 주기 지연
- 잘못된 헤더
- 잘못된 STEER_MODE

각 시험 후 실제 바퀴 토크, 제동 상태, LCD 상태를 기록한다.

---

# 28. ROS 2 구현 체크리스트

## 28.1 시리얼

- [ ] `/dev/ttyUSB*` 장치 확인
- [ ] udev 고정 이름 생성
- [ ] dialout 그룹 권한
- [ ] 115200 baud
- [ ] Parity None
- [ ] Stop bit 1
- [ ] Data bits 실물 확인
- [ ] 케이블 straight/null-modem 확인

## 28.2 송신

- [ ] 13바이트 고정 길이
- [ ] Header `53 54 58`
- [ ] Big Endian SPEED
- [ ] Big Endian STEER
- [ ] ALIVE 증가
- [ ] Tail `0D 0A`
- [ ] 20 Hz
- [ ] timeout 시 안전 명령

## 28.3 수신

- [ ] 25바이트 고정 길이
- [ ] Header 재동기화
- [ ] Tail 확인
- [ ] Little Endian 속도
- [ ] Little Endian 조향
- [ ] PCU ALIVE 감시
- [ ] Auto/Manual 상태
- [ ] E-Stop 상태
- [ ] 조향 모드 상태

## 28.4 안전

- [ ] 물리 E-Stop 담당자
- [ ] 바퀴 공중 시험
- [ ] 최대 속도 제한
- [ ] 최대 조향 제한
- [ ] 명령 stale timeout
- [ ] 통신 끊김 감지
- [ ] Auto Fail 진단
- [ ] Manual 전환 처리
- [ ] 비정상 패킷 폐기

---

# 29. 최종 적용 판단

ROMO-B는 USB to RS232로 Ubuntu 22.04 ROS 2 및 Autoware와 연결할 수 있다.

권장 초기 구성:

```text
Ubuntu 22.04
ROS 2 Humble
Autoware
Livox Mid-360 1대
USB to RS232
2WIS
20 Hz 차량 명령
0.3 m/s 이하 초기 시험
±5 deg 이하 초기 조향 시험
```

첫 구현 목표는 자율주행 전체가 아니라 다음 순서로 잡는 것이 안전하다.

```text
RS232 수신 확인
→ 13바이트 0 명령 송신
→ Auto 전환
→ 저속 직진
→ 미소 조향
→ 정지 및 E-Stop
→ ROS 2 Vehicle Interface
→ Wheel odometry
→ Mid-360 Localization
→ Autoware 경로 추종
```

---

# 30. 구현 시 최우선 확인 사항

1. USB to RS232 케이블 배선 방식
2. Data bits
3. 음수 SPEED와 STEER 표현
4. AorM 상태값 또는 상승 에지 조건
5. HLV ESTOP 상태값 또는 상승 에지 조건
6. Auto Fail 시 실제 제동
7. PCU 피드백이 실측값인지 내부 명령값인지
8. Autoware tire angle과 ROMO-B 등가 조향각의 관계
9. 4WIS 수식은 `R = (L/2) / tan(delta)` 사용
10. 초기 자율주행은 2WIS 사용

---

# 31. 문서 내 확실하지 않은 사항 모음

다음은 원본 매뉴얼만으로 확정할 수 없다.

- RS232 Data bits
- RS232 핀맵
- straight/null-modem 여부
- ALIVE wrap 방식
- signed two's complement 명시
- Auto 전환 상승 에지 필수 여부
- HLV E-Stop 상승 에지 필수 여부
- Auto Fail의 실제 제동 방식
- 피드백의 센서값 또는 명령값 구분
- 상판 통신 포트와 패널 Serial Port의 관계
- 32쪽 전원 절차의 `On` 문구 의도
- Crab 주행의 외부 프로토콜 지원 방식

이 항목들은 실물 시험 또는 제조사 문의로 확정해야 한다.
