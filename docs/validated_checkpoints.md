# OpenArm 검증 체크포인트 등록부

이 문서는 TensorBoard 수치가 아니라 사용자가 GUI에서 직접 확인해 좋은 기준으로 지정한 정책만
기록한다. 현재 확정된 good baseline은 **3개**다. 새 버전은 학습 중이거나 sampled 지표만 좋으면
포함하지 않으며, 정확한 run과 checkpoint를 GUI로 확인한 뒤 추가한다.

## 확정 목록

| 버전 | 상태 | 원본 체크포인트 | 소스 기준 | 확인된 장점 | 남은 보완점 |
| --- | --- | --- | --- | --- | --- |
| v18 | `USER_APPROVED_REFERENCE` | `2026-07-10_16-30-21_hybrid_place_v18/model_825.pt` | standalone commit 없음; 변경은 `4455992`에 누적 반영 | 사용자가 다시 지정한 강한 파지·들기·초기 운반 기준 | 완전한 최종 place 검증 기록은 없음 |
| v28 | `USER_APPROVED` | `2026-07-12_03-16-50/model_4999.pt` | `b4bedea` | 실제 grasp, lift, transport, place 성공 | 파지 확률, 운반 가속, 최종 높이의 자연스러움 보완 필요 |
| v29 | `USER_APPROVED_WITH_CAVEATS` | `2026-07-12_15-37-08/model_2100.pt` | `afe12f3` | 운반, 감속, 하강, release가 정상인 최적화 기준 | 파지 확률을 더 높이고 최종 목표를 조금 낮출 필요 |

세 모델은 모두 사용자 GUI 승인 기준이다. 아직 `4 seeds x 128 env`와 GUI 10회로 구성된 정식
acceptance 전체를 통과한 것으로 기록하지 않으며, `formal_acceptance`는 별도 검증 전까지 pending이다.

저장소 보존 경로:

```text
artifacts/checkpoints/openarm_pick_place/v18/checkpoint.pt
artifacts/checkpoints/openarm_pick_place/v28/checkpoint.pt
artifacts/checkpoints/openarm_pick_place/v29/checkpoint.pt
```

SHA-256:

```text
v18 fa0fce2fb50b6871ad04620037f8db8e24493cdbb728e1023e45dadaa85e6d8e
v28 0b165eaf5e8e087b856fd53111981a0433e4ee248f6bf108363b3ca4609dbf84
v29 fec454a3aec1770408a7c8d553deba3d20fd7d890531affe17f6082f20121ea3
```

## 평가 대기

| 버전 | Run | 상태 |
| --- | --- | --- |
| v30 | `2026-07-12_22-06-37/model_1775.pt` | run 최고점 보존 및 GUI 검토 중; 이후 정책이 0%로 붕괴해 good baseline 3개에는 포함하지 않음 |

## 등록 규칙

1. sampled TensorBoard 성공률만으로 확정하지 않는다.
2. actor-mean 첫 완전 episode와 1-env GUI 실제 동작을 확인한다.
3. 사용자가 좋은 버전으로 명시한 정확한 `run/model_N.pt`만 등록한다.
4. 파지·운반·놓기 중 어느 부분이 검증됐는지와 남은 보완점을 함께 적는다.
5. 확정 체크포인트는 `agent.yaml`, `env.yaml`, SHA-256과 함께 저장소에 보존한다.
