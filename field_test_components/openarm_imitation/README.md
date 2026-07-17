# OpenArm-v1 Bimanual + ExoArm-7 모방학습 작업공간

기존 OpenArm/ExoArm 설정을 변경하지 않는 별도 작업공간이다. 가슴 D435iCB를 DP3의 주 포인트클라우드로 사용하고, 손 D405 RGB도 같은 타임스텝에 저장한다. 수집기는 CAN을 수신만 하며 기존 텔레옵 브리지가 로봇을 제어한다.

## 1. 최초 환경 확인

```bash
source /opt/ros/humble/setup.bash
source /home/mh/openarm_imitation/.venv-collect/bin/activate
python /home/mh/openarm_imitation/scripts/preflight.py
```

카메라 두 대를 USB 3.x 포트에 연결한 뒤 시리얼을 기록한다.

```bash
source /opt/ros/humble/setup.bash
source /home/mh/openarm_imitation/.venv-collect/bin/activate
python /home/mh/openarm_imitation/scripts/discover_cameras.py --write
```

두 카메라를 같은 USB 2.0 허브에 연결하지 않는다. 출력의 `usb=` 값이 모두 3.x인지 확인한다.

## 2. 기존 텔레옵 실행

터미널 1:

```bash
cd /home/mh/foxtech_exoarm/qnbot-exo-std-demo
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch qnbot_teleoperator websocket_teleoperator.launch.py
```

터미널 2:

```bash
cd /home/mh/foxtech_exoarm/software-tool-v1.2.3/linux
./"Qnbot HMI Control-1.2.3.AppImage"
```

HMI forwarding은 `ws://127.0.0.1:19091`이다.

터미널 3에서 기존 검증된 브리지를 실행한다. 로봇과 ExoArm을 안전한 중립 자세에 두고 E-stop을 손에 둔다.

```bash
source /opt/ros/humble/setup.bash
python3 /home/mh/openarm_teleop/exo_openarm_bridge/exo_to_openarm.py
```

## 3. 수집

터미널 4:

```bash
/home/mh/openarm_imitation/scripts/run_collection.sh pick_and_place 50
```

- `SPACE`: 에피소드 시작 / 저장
- `d`: 현재 에피소드 폐기
- `q`: 종료

수집 중 모니터의 센서 age가 200 ms를 넘는 프레임은 저장하지 않는다.

## 4. 확인과 DP3 변환

```bash
source /home/mh/openarm_imitation/.venv-collect/bin/activate
RAW=/home/mh/openarm_imitation/datasets/pick_and_place/dataset.zarr
PTV=/home/mh/openarm_imitation/datasets/pick_and_place_ptv/dataset.zarr

python /home/mh/openarm_imitation/scripts/validate_dataset.py "$RAW" --strict
python /home/mh/openarm_imitation/scripts/preview_dataset.py "$RAW"
python /home/mh/openarm_imitation/scripts/convert_dp3.py --src "$RAW" --dst "$PTV"
python /home/mh/openarm_imitation/scripts/validate_dataset.py "$PTV" --strict
python /home/mh/openarm_imitation/scripts/make_dp3_task.py --dataset "$PTV" --name openarm_pick_and_place
```

기본 action은 안전한 position-only 16차원이다. 제공 자료의 실험적 48차원(qpos+velocity+torque)이 꼭 필요한 경우에만 변환 시 `--action-mode qpos_velocity_torque`를 사용한다.

## 5. 학습

RTX 2050 4 GiB에 맞춰 batch 4로 시작한다.

```bash
/home/mh/openarm_imitation/scripts/train_dp3.sh openarm_pick_and_place 1000
```

DP3 공식 기본 구성은 더 큰 VRAM을 전제로 한다. CUDA OOM이 나면 batch를 2로 낮추거나 학습 서버를 사용한다.

## 비상 정지

```bash
/home/mh/openarm_imitation/scripts/emergency_disable.sh
```

상세 판단과 데이터 차원은 [docs/SETUP_AUDIT.md](docs/SETUP_AUDIT.md)에 정리되어 있다.

