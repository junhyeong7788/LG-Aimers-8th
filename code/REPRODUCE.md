# REPRODUCE

## 목적
본 폴더의 핵심 실험(기준선, 시드 앙상블, 안정화 파이프라인)을 최소 단계로 재현한다.

## 1) 환경
- Python 3.11
- 핵심 라이브러리: `transformers`, `datasets`, `llmcompressor`, `compressed-tensors`, `torch`
- 참고: `requirements.txt` (로컬 개발 환경 기준)

## 2) 핵심 노트북 실행 순서
1. `01_fp8_static_mixed512_baseline.ipynb`
2. `02_fp8_seed_ensemble_calibration.ipynb`
3. `03_fp8_stability_u1_u2_pipeline.ipynb`

노트북 내부 경로는 `${PROJECT_ROOT}` 기준으로 작성되어 있다.

## 3) 로컬 평가 명령 (대표)
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

## 4) 결과 확인
- 비교 요약: `${PROJECT_ROOT}/eval/comparison_*.json`
- 세부 결과: `${PROJECT_ROOT}/eval/eval_*.json`

## 5) 서버 제출 유의사항
- 제출 파일은 `submit.zip`이며 최상위 구조는 `model/`만 허용
- 서버 점수는 로컬 점수와 분포가 달라 역전이 발생할 수 있음
