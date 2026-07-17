# 설치 판단 기록

작성 기준: 2026-07-16, Ubuntu 22.04 / ROS 2 Humble / OpenArm-v1.

## 원본 자료에서 확인한 핵심

- OpenArm-v1 최종 성공 통신은 `Classic CAN 1 Mbps`, `can0=오른팔`, `can1=왼팔`, `OpenArm(..., False)`이다.
- ExoArm 입력은 `/exo/joint_command`의 16개 position이며 순서는 왼팔 7, 오른팔 7, 왼쪽 trigger, 오른쪽 trigger이다.
- 현재 동작 확인된 브리지는 `/home/mh/openarm_teleop/exo_openarm_bridge/exo_to_openarm.py`이다.
- 동봉 DP3 예제는 단팔, OpenArm 리더, `can0/can2`, CAN-FD를 전제로 하므로 현재 장비에 직접 사용할 수 없다.
- 제공 `convert_to_ptv.py`는 다음 프레임의 follower 상태를 action으로 만든다. 새 변환기도 같은 1-step shift를 양팔에 적용한다.

## 보존한 기존 경로

다음 경로는 조사만 했고 이 프로젝트에서 수정하지 않는다.

- `/home/mh/exo_openarm_ws`
- `/home/mh/foxtech_exoarm`
- `/home/mh/openarm_can`
- `/home/mh/openarm_teleop`
- `/home/mh/openarm_ws`

## 현재 하드웨어 검사 결과

- PCAN-USB FD 어댑터 2개가 `can0`, `can1`로 인식됨.
- 두 인터페이스 모두 Classic CAN 1 Mbps, MTU 16, ERROR-ACTIVE.
- RealSense D405/D435iCB와 ExoArm USB 장치는 검사 시점에 연결되지 않았음.
- GPU는 NVIDIA GeForce RTX 2050 4 GiB. DP3 기본 배치 128은 사용할 수 없어 배치 4로 설정함.

## 데이터 의미

- raw `state/velocity/torque`: 오른팔 8개(can0) 뒤 왼팔 8개(can1), 단위 degree/degree/s/Nm.
- raw `exo_action_raw`: ExoArm ROS 메시지 원본 16개. 좌표계가 follower와 다르므로 직접 학습하지 않는다.
- converted `state`: 현재 follower 양팔 16차원.
- converted `action` 기본값: 다음 프레임 follower 양팔 qpos 16차원.
- `point_cloud`: 가슴 D435iCB의 정렬된 RGB-D에서 만든 1024x6 XYZRGB.
- `img`: 가슴 RGB, `img_hand`: 손 D405 RGB. 외부 보정 없이 두 카메라 포인트클라우드는 합치지 않는다.

## 안전 경계

- 수집기는 SocketCAN `recv()`만 사용하며 모터 enable/send 코드를 제공하지 않는다.
- 기존 ExoArm 브리지와 함께 실행할 수 있으나, 브리지를 먼저 정상 구동해야 모터 피드백이 들어온다.
- 학습 결과의 실제 모터 배포는 포함하지 않는다. 데이터/정책 검증과 관절별 제한 확인 전 자동 구동은 안전하지 않다.

