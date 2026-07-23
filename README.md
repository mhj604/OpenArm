# OpenArm IEEE Access 연구 패키지

이 디렉터리는 전방 편심 하중을 가진 ROMO-B/OpenArm 모바일 매니퓰레이터의
2WIS(Ackermann) 궤적 오차와 Pivot 조향 한계를 엄격하게 분석한 결과를 보존합니다.

핵심 결론은 다음 한 문장입니다.

> ROMO-B의 고정 ±30° Pivot은 매뉴얼 기하에서 요구되는 약 ±39.72°와 맞지 않아
> 제거 불가능한 구조적 rolling residual을 만들며, 양팔은 이 기하 오차 자체가 아니라
> payload, 하중이동, 마찰 및 관성 때문에 추가되는 condition-dependent residual만 바꿀 수 있습니다.

## 문서 안내

- [ROMO-B 및 Livox Mid-360 통합 스펙시트](docs/ROMOB_LIVOX_SPEC_SHEET_KO.md)
- [ROMO-B 및 Livox Mid-360 통합 운용 매뉴얼](docs/ROMOB_LIVOX_INTEGRATION_MANUAL_KO.md)
- [엄격한 연구 분석](docs/RESEARCH_AUDIT_KO.md)
- [하드웨어 실험 프로토콜](docs/EXPERIMENT_PROTOCOL_KO.md)
- [저널·주장 경계](docs/VENUE_AND_CLAIM_BOUNDARIES_KO.md)
- [출처·검증 한계](docs/PROVENANCE_AND_LIMITATIONS.md)
- [문헌과 공식 링크](references/SOURCES.md)
- [BibTeX](references/references.bib)
- [원본 자료 목록](source_files/README.md)
- [ROMO-B/OpenArm Isaac Sim 5.1 모델](simulation/README.md)
- [Isaac Sim 구현 명세와 정확도 경계](docs/ISAAC_SIM_ROMOB_OPENARM_KO.md)
- [ROMO-B 핵심 근거 페이지](evidence/manual_pages/README.md)
- [Oracle/Web ChatGPT 자문 기록](evidence/oracle/README.md)

## 패키지 범위

포함된 항목:

- 사용자가 제공한 ROMO-B, Livox, OpenArm 하드웨어 문서와 OpenArm USD 파일
- ROMO-B 4WID-4WIS 베이스와 전방 OpenArm을 결합한 검증 가능한 Isaac Sim 5.1 USDA
- 매뉴얼 명시값·근삿값·파생값·미확정값을 분리한 기계 판독형 수치 명세와 테스트
- 368.6 MB ROMO-B 원본의 38쪽 전체 압축본과 원본 SHA-256
- ROMO-B 매뉴얼 3, 26, 31, 38쪽 고해상도 근거 이미지
- 수식, 확정 사실/가설 분리, 선행연구 충돌, 실험·통계 설계, Go/No-Go 기준
- DOI 기반 동료심사 문헌 목록과 공식 저널 범위 링크
- 외부 모델 자문의 원문 및 모델 검증 한계

포함하지 않은 항목:

- 학교 구독으로 열람한 논문의 저작권 원문 PDF
- ROMO-B 368.6 MB 원본 바이트 그대로의 파일(GitHub 100 MB 제한 초과)

구독 논문은 DOI, 출판사 링크와 검증한 주장만 기록했습니다. ROMO-B는 전체 38쪽을
보존한 읽기 가능한 압축본을 제공하며 원본 파일의 해시와 크기를 별도로 남겼습니다.

## 재생성 및 검증

압축본 재생성:

```bash
python OpenArm_IEEE-ACCESS/scripts/build_compressed_romo_manual.py \
  /path/to/ROMO-B.pdf \
  OpenArm_IEEE-ACCESS/source_files/hardware/ROMO-B_manual_complete_compressed.pdf
```

패키지 검증:

```bash
python OpenArm_IEEE-ACCESS/scripts/validate_research_bundle.py
```

`references/SHA256SUMS`는 저장된 자료와 분석 증거의 무결성을 확인합니다.

Isaac Sim 모델의 생성·검증 방법은 [`simulation/README.md`](simulation/README.md)를
참고하십시오. 현재 17 cm 장착 조건은 차체 기하 중심에서 전방 `+0.170 m`로 해석해
적용했으며, 기준점이 전면 끝이라면 사용할 `+0.208 m`도 설정에 별도로 보존했습니다.
