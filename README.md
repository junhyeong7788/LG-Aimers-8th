# LG Aimers 8th Portfolio (EXAONE-4.0-1.2B Compression)

## 1) 프로젝트 요약

- 대회: LG Aimers 8th (Phase2, 코드 제출형)
- 목표: EXAONE-4.0-1.2B 경량화에서 정확도/속도 동시 최적화
- 평가식: `Score = max(0.5 * PerfNorm + 0.5 * SpeedNorm, 0)`
- 핵심 결론: 서버 기준 최고점은 **FP8 Static Mixed512** 계열에서 달성

## 2) 최종 성과

- **최종 순위: 약 630팀 중 82등 (상위 ~13%)**

| 모델                              |        서버 점수 |      시간 | 날짜       | 해석                              |
| --------------------------------- | ---------------: | --------: | ---------- | --------------------------------- |
| **exp3_fp8_static_mixed512**      | **0.6219616518** |  약 9분대 | 2026-02-22 | 전체 최고점                       |
| exp3_fp8_static_mixed512 (재제출) |     0.6194291468 |  9분 46초 | 2026-02-25 | 최고점 대비 -0.00253              |
| exp2_fp8_static_manta256          |     0.6105535201 |         - | 2026-02-22 | 안정적 상위                       |
| FP8 Dynamic base                  |             0.61 |         - | 2026-02    | 초기 강한 기준선                  |
| exp_r1_fp8_static_m224_p288_t512  |     0.5050556795 |  13분 2초 | 2026-02-23 | 로컬 상위였지만 서버 하락         |
| u1_u2_klmix_static                |      0.477602577 | 12분 19초 | 2026-02-25 | 학습 기반 접근의 서버 일반화 실패 |

## 3) 접근 방식 요약

1. FP8 Dynamic baseline 수립

- 빠른 추론, 안정적 기준점 확보

2. FP8 Static 캘리브레이션 분포 최적화

- `MANTA + pile` 혼합 비율 탐색
- **256:256 mixed512** 조합이 가장 강건

3. 3rd dataset(ARC) 소량 혼합 탐색

- 512 총량 내 분포 미세 조정
- 일부 로컬 개선은 있었으나 mixed512 확실한 우위는 미확인

4. Seed-ensemble calibration

- 다중 seed 캘리브레이션 병합(`k=5`, `cap=1024`)
- 로컬 상승(예: overall 0.4585) 확인, 서버 확정 우위는 미검증

5. Stability pipeline (U1/U2/KL mix)

- 로컬 지표 상승 가능성 확인
- 서버 점수 급락으로 hidden 분포 일반화 리스크 확인

## 4) 최종 교훈

- 로컬 상위 모델이 서버 상위 모델을 보장하지 않음
- 학습 기반 weight 변경은 분산(리스크)이 큼
- 본 대회에서는 **양자화-only + 캘리브레이션 분포 최적화**가 가장 재현성 높았음

## 5) 문서/코드 바로가기

- 정량 요약: `RESULTS_SUMMARY.md`
- 핵심 코드 인덱스: `CORE_CODE_INDEX.md`
- 원본 상세 리포트: `code/EXPERIMENT_REPORT.md`

## 6) GitHub 업로드 패키지 구성

- 문서
  - `README.md`
  - `RESULTS_SUMMARY.md`
  - `CORE_CODE_INDEX.md`
- 코드/재현
  - `code/01_fp8_static_mixed512_baseline.ipynb`
  - `code/02_fp8_seed_ensemble_calibration.ipynb`
  - `code/03_fp8_stability_u1_u2_pipeline.ipynb`
  - `code/run_eval.py`
  - `code/requirements.txt`
  - `code/REPRODUCE.md`
  - `code/RESULTS_TABLE.csv`
  - `code/DATA_LICENSES.md`
  - `code/figures/`
