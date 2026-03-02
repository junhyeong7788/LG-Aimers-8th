# Results Summary

## 1) 정량 성과 요약

### 서버 성과 (최종)

| 순위 성격 | 모델 | 점수 | 비고 |
|-----------|------|-----:|------|
| 최고점 | exp3_fp8_static_mixed512 | **0.6219616518** | 2026-02-22 |
| 차상위권 | exp2_fp8_static_manta256 | 0.6105535201 | static 계열 |
| 기준선 | FP8 Dynamic base | 0.61 | 초기 baseline |
| 최종 재제출 | exp3_fp8_static_mixed512 | 0.6194291468 | 2026-02-25 |
| 학습형 비교 | u1_u2_klmix_static | 0.477602577 | 2026-02-25 |

### 로컬 대표 성과 (비교 파일 기준)

| 모델 | overall_median | time_median(s) | ScoreProxy | 출처 |
|------|---------------:|---------------:|-----------:|------|
| exp_se_exp_b1_fp8_static_m224_p224_a064_t512_k5_cap1024 | 0.4585 | 293.3 | 0.5590 | `eval/comparison_20260224_063151.json` |
| exp_u1_u2_klmix_static | 0.4563 | 292.7 | 0.5522 | `eval/comparison_20260224_123041.json` |
| exp_a0_fp8_static_m256_p256_a000_t512 | 0.4530 | 290.4 | 0.5577 | `eval/comparison_20260223_181251.json` |
| exp3_fp8_static_mixed512 | 0.4523 | 292.1 | - | `EXPERIMENT_REPORT.md` |

## 2) 핵심 비교 포인트

1. 로컬 최고가 서버 최고를 보장하지 않음
- `exp_r1`은 로컬 상위였지만 서버 0.5050으로 급락

2. 학습형 접근(U1/U2/KL)은 상향 가능성과 하락 리스크가 동시에 큼
- 로컬에서는 `ScoreProxy` 상위
- 서버에서는 `u1_u2_klmix_static`가 0.4776으로 하락

3. 서버 최적 전략은 static mixed 계열 유지
- mixed512가 최고점 및 재현성 측면에서 가장 안정적

## 3) 최종 요약

- Best Score: **0.6219616518**
- Best Family: **FP8 Static (MANTA + pile mixed calibration)**
- Practical Strategy: **서버 검증 앵커(mixed512) 중심 + 제한적 변형 실험**
