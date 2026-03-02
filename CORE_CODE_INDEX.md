# Core Code Index

## 1) 핵심 실험 노트북

| 구분 | 파일 | 역할 | 대표 산출물 |
|------|------|------|------------|
| New Approaches | `07_new_approaches/code/Experiments_NewApproaches.ipynb` | FP8 dynamic/static, mixed512 기준선 확립 | `07_new_approaches/code/exp3_fp8_static_mixed512/submit_exp3_fp8_static_mixed512.zip` |
| Calibration Search | `08_calibration_search/code/Experiments_CalibrationSearch.ipynb` | MANTA:pile 비율 탐색 (512 고정) | `08_calibration_search/code/exp*_fp8_static_*/submit_*.zip` |
| Refined Search | `09_refined_calibration_search/code/Experiments_RefinedCalibration.ipynb` | 비율/총량/구성(no_pilecc, stratified) 정밀 탐색 | `09_refined_calibration_search/code/exp_r*/submit_*.zip` |
| Third Dataset Search | `10_third_dataset_search/code/Experiments_ThirdDatasetSearch.ipynb` | ARC 소량 혼합 3-way 분포 탐색 | `10_third_dataset_search/code/exp_a*/submit_*.zip` |
| Seed Ensemble | `11_seed_ensemble_calibration_search/code/Experiments_SeedEnsembleCalibration.ipynb` | 다중 seed 캘리브레이션 병합 | `11_seed_ensemble_calibration_search/code/exp_se_exp_b1_fp8_static_m224_p224_a064_t512_k5_cap1024/submit_exp_se_exp_b1_fp8_static_m224_p224_a064_t512_k5_cap1024.zip` |
| Stability Pipeline | `12_fp8_stability_pipeline/code/Experiments_FP8_StabilityPipeline.ipynb` | U1/U2/U3 학습 + FP8 static 통합 파이프라인 | `12_fp8_stability_pipeline/code/exp_u1_u2_klmix_static/submit_exp_u1_u2_klmix_static.zip` |

## 2) 평가/분석 코드

| 파일 | 역할 |
|------|------|
| `eval/run_eval.py` | 다중 모델 반복 평가, 중앙값/시간/효율/ScoreProxy 산출 |
| `eval/comparison_*.json` | 비교 결과(모델별 task 성능/시간/ScoreProxy) 저장 |
| `EXPERIMENT_REPORT.md` | 전체 실험 기록, 서버 제출 결과, 교훈 정리 |

## 3) 제출에 실제 사용한 대표 아티팩트

1. mixed512 최고점
- `07_new_approaches/code/exp3_fp8_static_mixed512/submit_exp3_fp8_static_mixed512.zip`

2. seed-ensemble 대표
- `11_seed_ensemble_calibration_search/code/exp_se_exp_b1_fp8_static_m224_p224_a064_t512_k5_cap1024/submit_exp_se_exp_b1_fp8_static_m224_p224_a064_t512_k5_cap1024.zip`

3. 학습형 대표
- `12_fp8_stability_pipeline/code/exp_u1_u2_klmix_static/submit_exp_u1_u2_klmix_static.zip`

## 4) 재현 최소 커맨드

```bash
python ${PROJECT_ROOT}/eval/run_eval.py \
  --model_path ${PROJECT_ROOT}/open/base_model \
  ${PROJECT_ROOT}/07_new_approaches/code/exp3_fp8_static_mixed512/model \
  ${PROJECT_ROOT}/11_seed_ensemble_calibration_search/code/exp_se_exp_b1_fp8_static_m224_p224_a064_t512_k5_cap1024/model \
  ${PROJECT_ROOT}/12_fp8_stability_pipeline/code/exp_u1_u2_klmix_static/model \
  --n_runs 3 \
  --baseline_model_idx 0 \
  --tasks gsm8k,mmlu,arc_challenge \
  --limit 512 \
  --output_dir ${PROJECT_ROOT}/eval_results
```

## 5) 포트폴리오 보조 파일

- `code/REPRODUCE.md`: 실행 순서와 대표 CLI
- `code/RESULTS_TABLE.csv`: 서버/로컬 핵심 지표 테이블
- `code/DATA_LICENSES.md`: 사용 데이터셋 라이선스 요약
- `code/figures/01_server_score_comparison.svg`
- `code/figures/02_local_overall_comparison.svg`
- `code/figures/03_local_vs_server_scatter.svg`
- `code/run_eval.py`, `code/requirements.txt`: 코드 리뷰/재현용 최소 파일
