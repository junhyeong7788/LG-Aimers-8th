# EXAONE-4.0-1.2B 경량화 실험 보고서

## 1. 대회 개요

- 주최: LG AI 연구원 / 데이콘
- 목표: EXAONE-4.0-1.2B 모델 경량화 (성능 유지 + 추론 속도 개선)
- 점수 공식: `Score = max(0.5 × PerfNorm + 0.5 × SpeedNorm, 0)`
  - PerfNorm = 경량화 모델 정확도 / 기본 모델 정확도
  - SpeedNorm = 1 - (경량화 TPOT / 기본 TPOT)
  - 성능과 속도가 50:50 동등 기여
- 평가 벤치마크: 비공개 (서버 hidden benchmark)
- 로컬 평가 벤치마크: GSM8K, MMLU, ARC Challenge (각 512 샘플, 중앙값 3회)

### 평가 서버 환경

| 항목 | 사양 |
|------|------|
| GPU | L4 (22.4 GiB VRAM) |
| CPU | 6 vCPU |
| RAM | 28 GB |
| 추론 엔진 | vLLM 0.14.1 (수정 불가) |
| 직렬화 포맷 | compressed-tensors 0.13.0 |
| 시간 제한 | 20분 |
| 파일 제한 | ZIP 10GB, 압축해제 32GB |
| 1일 제출 | 최대 3회 |

### 기본 모델 아키텍처

| 항목 | 값 |
|------|-----|
| 모델 | EXAONE-4.0-1.2B (Exaone4ForCausalLM) |
| 레이어 | 30 (Post-LN 아키텍처) |
| Hidden size | 2048 |
| Attention heads | 32 (KV heads: 8) |
| Intermediate size | 4096 |
| Vocab size | 102,400 |
| tie_word_embeddings | true |

---

## 2. 기술적 제약사항

### 2.1 vLLM lm_head 양자화 불가

vLLM의 `Exaone4ForCausalLM`에서 `lm_head`는 `ParallelLMHead` = `VocabParallelEmbedding`(Embedding 레이어)으로 구현되어 있다. Embedding 레이어는 `weight_scale` 파라미터 슬롯이 없으므로, FP8/INT8 양자화된 lm_head를 로드하면 `ValueError: There is no module or parameter named 'lm_head.weight_scale'` 오류가 발생한다.

- 해결: 모든 양자화에서 `ignore=["lm_head"]` 필수
- `tie_word_embeddings=False`로 untie해도 lm_head는 여전히 Embedding 타입이므로 양자화 불가
- 관련 파일: `vllm/model_executor/models/exaone4.py` L468-L516

### 2.2 EXAONE-4.0 Post-LN 아키텍처

EXAONE-4.0은 Pre-LN이 아닌 Post-LN 구조를 사용한다:

```
residual = hidden_states
hidden_states = self_attn(hidden_states)
hidden_states = post_attention_layernorm(hidden_states)  # POST-LN
hidden_states = residual + hidden_states
```

이로 인해 SmoothQuant(Pre-LN에서 Norm→Linear 직접 연결을 전제)가 근본적으로 비호환이다. `RuntimeError: Error resolving mappings for given architecture` 발생.

### 2.3 KV Cache FP8 제어 불가

vLLM 0.14.1에서 `kv_cache_scheme`을 model config.json에 추가해도 자동으로 FP8 KV cache가 활성화되지 않는다. `get_kv_cache_quant_algo_string()`은 `quant_method.startswith("modelopt")`인 경우만 처리하며, compressed-tensors 포맷은 해당하지 않는다. 평가 서버의 vLLM 엔진 설정(`--kv-cache-dtype`)을 제어할 수 없으므로 이 방향은 불가능하다.

---

## 3. 전체 실험 결과

### 3.1 서버 제출 결과

| 모델 | 서버 점수 | 비고 |
|------|----------|------|
| **FP8 Static Mixed512 (exp3_fp8_static_mixed512)** | **0.6219616518** | **신규 최고점 (2026-02-22 갱신)** |
| FP8 Static Refined (exp_r1_fp8_static_m224_p288_t512) | 0.5050556795 (13분 2초) | 로컬 score_proxy 1위였으나 서버 급락 (2026-02-23) |
| FP8 Static MANTA256 (exp2_fp8_static_manta256) | 0.6105535201 | FP8 Dynamic과 동급 이상 |
| FP8 Dynamic base | 0.61 | 기존 최고점, 현재 3위 |
| FP8 sel_layer0_qkv | 0.50 | 로컬에서는 base보다 높았으나 서버 하락 |
| KD LoRA (exp3_kd_lora) | 0.3764 | concept drift로 대폭 하락 |

### 3.2 로컬 평가 전체 결과 (역대 모든 실험)

아래 표는 모든 실험의 로컬 평가 결과를 정확도 기준 내림차순으로 정리한 것이다.

#### FP8 계열

| 실험 | 방법 | gsm8k | mmlu | arc | 전체 | vs Base | TPOT(s) | 평가일 |
|------|------|-------|------|-----|------|---------|---------|--------|
| exp3_kd_lora | KD LoRA r=32 + FP8 Dynamic | 0.5723 | 0.5113 | 0.3613 | 0.4816 | +11.4% | 284.0 | 02/21 |
| exp2_presft_fp8 | LoRA SFT + FP8 Dynamic | 0.5723 | 0.4672 | 0.3613 | 0.4669 | +8.0% | 288.0 | 02/20 |
| **exp3_fp8_static_mixed512** | **FP8 Static (MANTA256 + pile256)** | **0.6465** | **0.3726** | **0.3379** | **0.4523** | **+4.7%** | **292.1** | **02/22** |
| exp2_fp8_static_manta256 | FP8 Static (MANTA 256) | 0.6543 | 0.3638 | 0.3320 | 0.4500 | +4.1% | 296.1 | 02/22 |
| exp1_fp8_static | FP8 Static (MANTA 캘리브레이션) | 0.6465 | 0.3752 | 0.3320 | 0.4512 | +4.4% | 293.4 | 02/22 |
| exp4_fp8_static_manta512 | FP8 Static (MANTA 512) | 0.6230 | 0.3725 | 0.3223 | 0.4393 | +1.6% | 290.2 | 02/22 |
| exp1_fp8_sel_layer0_qkv | FP8 Dynamic + Layer0 QKV 보호 | 0.6387 | 0.3540 | 0.3320 | 0.4416 | +2.2% | 281.1 | 02/19 |
| exp1_fp8_selective_qkv | FP8 Dynamic + L0,L29 QKV 보호 | 0.6328 | 0.3529 | 0.3320 | 0.4392 | +1.6% | 300.7 | 02/19 |
| **FP8 Dynamic base** | **FP8 Dynamic (data-free)** | **0.6289** | **0.3474** | **0.3203** | **0.4322** | **기준** | **279.3** | **02/19** |
| exp1_fp8_sel_layer29_qkv | FP8 Dynamic + L29 QKV 보호 | 0.6250 | 0.3468 | 0.3223 | 0.4314 | -0.2% | 286.5 | 02/19 |
| exp1_fp8_sel_layer29_oproj | FP8 Dynamic + L29 OProj 보호 | 0.6211 | 0.3482 | 0.3223 | 0.4305 | -0.4% | 287.1 | 02/19 |
| exp2_kd_balanced | KD Full FT alpha=0.5 + FP8 | 0.4395 | 0.4272 | 0.3047 | 0.3904 | -9.7% | 284.6 | 02/21 |
| exp1_kd_conservative | KD Full FT alpha=0.3 + FP8 | 0.4199 | 0.4364 | 0.3125 | 0.3896 | -9.9% | 286.6 | 02/21 |
| exp3_kd_fp8 (이전 KD) | KD 수동루프 + FP8 Dynamic | 0.0000 | 0.2280 | 0.2344 | 0.1541 | -64.3% | 283.8 | 02/20 |

#### W4A16 계열

| 실험 | 방법 | gsm8k | mmlu | arc | 전체 | vs Base | TPOT(s) | 평가일 |
|------|------|-------|------|-----|------|---------|---------|--------|
| exp3_autoround_gs64 | AutoRound MANTA gs=64 | 0.5957 | 0.3815 | 0.3457 | 0.4410 | +2.0% | 319.9 | 02/20 |
| exp1_autoround_manta | AutoRound MANTA gs=128 | 0.6113 | 0.3417 | 0.3242 | 0.4257 | -1.5% | 319.0 | 02/20 |
| exp4_autoround_iter400 | AutoRound iter=400 | 0.5762 | 0.3338 | 0.3320 | 0.4140 | -4.2% | 321.3 | 02/20 |
| exp2_autoround_pile10k | AutoRound pile-10k | 0.5957 | 0.3050 | 0.3398 | 0.4135 | -4.3% | 318.9 | 02/20 |
| exp5_gptq_baseline | GPTQ W4A16 baseline | 0.5938 | 0.3007 | 0.3340 | 0.4095 | -5.3% | 322.8 | 02/20 |

#### Layer Pruning 계열

| 실험 | 방법 | gsm8k | mmlu | arc | 전체 | vs Base | TPOT(s) | 평가일 |
|------|------|-------|------|-----|------|---------|---------|--------|
| exp2_prune1L | 1 레이어 제거 + FP8 Dynamic | 0.3496 | 0.3670 | 0.3145 | 0.3437 | -20.5% | 281.0 | 02/22 |
| exp3_prune2L | 2 레이어 제거 + FP8 Dynamic | 0.1504 | 0.3209 | 0.3262 | 0.2658 | -38.5% | 278.3 | 02/22 |

---

## 4. 실험별 상세 분석

### 4.1 FP8 Dynamic base (기준선, 서버 0.61)

- 방법: llmcompressor `QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])`
- 특징: data-free, 캘리브레이션 불필요
- 추론 시 activation scale을 매 forward마다 동적 계산
- 모든 Linear 레이어를 균일하게 FP8 양자화
- 서버 점수 0.61로 장기간 최고점이었으며, 로컬 정확도(0.4322)와 서버 점수 사이 큰 괴리(+41%) 존재
- 이 괴리는 서버의 hidden 벤치마크가 로컬 벤치마크(gsm8k, mmlu, arc)와 다른 분포이기 때문

### 4.2 FP8 Selective Quantization (서버 0.50)

- 방법: 특정 레이어(L0, L29)의 QKV projection을 FP8 양자화에서 제외(FP16 유지)
- sel_layer0_qkv가 로컬에서 0.4416(base 대비 +2.2%)으로 가장 높았으나, 서버에서 0.50으로 하락
- 교훈: 비균일 양자화는 로컬에서 좋아도 서버에서 역효과 가능
- 서버의 hidden 벤치마크에서는 균일한 양자화가 더 안정적

### 4.3 FP8 Static

- 방법: `scheme="FP8"` + 캘리브레이션 데이터로 activation scale 사전 계산
- 핵심 결과(신규):
  - `exp3_fp8_static_mixed512` 서버 **0.6219616518** (신규 최고점)
  - `exp2_fp8_static_manta256` 서버 **0.6105535201** (FP8 Dynamic과 동급 이상)
- 로컬에서는 Dynamic 대비 TPOT가 느리지만(약 +3~5%), 서버 hidden 벤치마크에서 PerfNorm 이득이 이를 상쇄하고 초과함
- 결론: 이 대회의 hidden 분포에서는 **FP8 Static + 적절한 calibration 분포 설계**가 유효함이 실증됨

### 4.4 W4A16 AutoRound / GPTQ

- 방법: 4bit weight + 16bit activation, Marlin 커널로 추론
- gs64(group_size=64) 변형이 로컬 0.4410으로 가장 높음
- TPOT 319~323s로 FP8(279s) 대비 약 14% 느림
- 점수 공식에서 TPOT 페널티가 크기 때문에 FP8 대비 불리
- TPOT가 base보다 느리면 SpeedNorm이 음수 → 점수 하락

### 4.5 SmoothQuant (실패)

- EXAONE-4.0이 Post-LN 아키텍처이므로 SmoothQuant의 전제 조건(Pre-LN, Norm→Linear 직접 연결) 불충족
- `RuntimeError: Error resolving mappings for given architecture` 발생
- EXAONE-4.0은 SmoothQuant의 MAPPINGS_REGISTRY에 미등록
- 근본적 비호환으로 실험 자체 불가

### 4.6 Knowledge Distillation (서버 0.3764)

#### 이전 KD (수동 학습 루프) — 완전 실패

- alpha=0.7, temperature=2.0, 500 샘플, LoRA r=16
- gradient clipping/LR scheduler 없이 수동 루프 → gradient explosion
- gsm8k=0.0, 전체 0.1541

#### 개선 KD (HF Trainer 기반)

`compute_loss` 오버라이드로 KD loss 구현. 두 가지 치명적 버그를 수정:

1. `fp16=True` → `bf16=True`: 모델이 FP16 로드 상태에서 `fp16=True`는 AMP GradScaler가 FP32 master weight를 기대하여 `ValueError: Attempting to unscale FP16 gradients` 발생
2. KD loss 스케일 불일치: `F.kl_div(reduction="batchmean")`이 3D 텐서 (B, S, V)에서 batch_size(B)로만 나눔 → KD loss가 시퀀스 길이에 비례하여 수백 규모로 폭발. `reduction="none"` 후 non-padding 토큰 수로 직접 평균하여 해결

| KD 실험 | alpha | temp | 방식 | gsm8k | 전체 | vs Base |
|---------|-------|------|------|-------|------|---------|
| exp3_kd_lora | 0.3 | 1.0 | LoRA r=32 | 0.5723 | 0.4816 | +11.4% |
| exp2_kd_balanced | 0.5 | 1.5 | Full FT | 0.4395 | 0.3904 | -9.7% |
| exp1_kd_conservative | 0.3 | 1.0 | Full FT | 0.4199 | 0.3896 | -9.9% |

- Full Fine-Tuning은 lr=1e-6로도 원본 분포를 과도하게 변형 → catastrophic forgetting
- LoRA r=32가 원본 weight를 동결하여 forgetting 최소화
- exp3_kd_lora가 로컬 역대 최고(0.4816)였으나, 서버 제출 시 0.3764로 대폭 하락
- 학습 데이터(MANTA+KMMLU+GSM8K)와 서버 hidden 벤치마크 분포 불일치 → concept drift

### 4.7 Pre-SFT + FP8 Dynamic

- 방법: LoRA SFT (MANTA+KMMLU+GSM8K 1000샘플) → merge → FP8 Dynamic 양자화
- 로컬 0.4669 (base 대비 +8.0%)
- mmlu/arc 개선되었으나 gsm8k 0.5723 (base 0.6289 대비 -9%)
- 특정 태스크에 과적합하면서 다른 태스크 성능 저하 (concept drift 징후)
- 서버 미제출

### 4.8 Layer Pruning + FP8 Dynamic (실패)

- 방법: cosine similarity 기반 레이어 중요도 분석 → 가장 redundant한 레이어 제거 → FP8 Dynamic 양자화
- EXAONE-4.0의 `config.json`에 `layer_types` 배열이 존재하여, `num_hidden_layers` 변경 시 함께 수정 필요 (미수정 시 `ValueError: num_hidden_layers must be equal to the number of layer types`)

| 변형 | 제거 레이어 수 | 정확도 하락 | TPOT 변화 | 판정 |
|------|-------------|-----------|----------|------|
| 29L | 1개 | -20.5% | -0.62% (느림) | 실패 |
| 28L | 2개 | -38.5% | +0.34% | 실패 |

실패 원인:
- 1.2B 모델(30L)은 이미 최소 규모이므로 레이어 1개 제거도 치명적
- gsm8k 0.6289 → 0.3496 (1L 제거), 0.1504 (2L 제거)로 수학 추론 능력 붕괴
- TPOT 개선이 거의 0% — vLLM 추론 시간 중 Linear 연산 비중이 작고, KV cache 관리/sampling 오버헤드가 지배적
- 정확도 하락으로 인해 잘못된 긴 응답 생성 → 오히려 총 평가 시간 증가 가능

### 4.9 New Approaches (2026-02-22): FP8 Static 캘리브레이션 분포 실험

`07_new_approaches`에서 아래 4개를 동일 파이프라인으로 비교했다.

| 모델 | 로컬 전체 | 로컬 시간(s) | 서버 점수 | 해석 |
|------|----------|-------------|----------|------|
| exp0_fp8_dynamic_base | 0.4322 | 283.4 | 0.61 | 빠르고 안정적 기준선 |
| exp2_fp8_static_manta256 | 0.4500 | 296.1 | 0.6105535201 | 정확도 이득이 속도 페널티를 거의 상쇄 |
| **exp3_fp8_static_mixed512** | **0.4523** | **292.1** | **0.6219616518** | **현재 최적: 정확도 이득 + 속도 손실 최소화** |
| exp4_fp8_static_manta512 | 0.4393 | 290.2 | 미제출 | 분포 다양성 부족 가능성 |

핵심 관찰:
- `MANTA only`보다 `MANTA + 일반 텍스트(pile)` 혼합이 더 강건하게 동작했다.
- static의 성패는 “FP8 vs Dynamic” 자체보다 **캘리브레이션 데이터 분포 매칭**의 영향이 더 크다.
- `exp3_fp8_static_mixed512`와 `exp3_fp8_static_m256_p256`는 아티팩트 기준 사실상 동일 모델(동일 비율 실험명 차이)로 재확인됨.

### 4.10 Calibration Search (2026-02-22): 비율 탐색 1차 (`08_calibration_search`)

`MANTA : pile` 비율을 고정 총량 512에서 바꿔가며 실험했다.

| 모델 | 비율 (MANTA:pile) | 로컬 전체 | 시간(s) | ScoreProxy |
|------|-------------------|----------|--------|-----------|
| exp3_fp8_static_m256_p256 | 256:256 | 0.4523 | 292.3 | 0.5482 |
| exp4_fp8_static_m320_p192 | 320:192 | 0.4415 | 291.7 | 0.5365 |
| exp1_fp8_static_m448_p064 | 448:64 | 0.4398 | 292.9 | 0.5326 |
| exp2_fp8_static_m384_p128 | 384:128 | 0.4364 | 290.9 | 0.5318 |
| exp5_fp8_static_m512_p000 | 512:0 | 0.4391 | 293.9 | 0.5302 |

핵심 관찰:
- 256:256 균형 혼합이 가장 높았고, 한쪽 분포로 치우칠수록 로컬 성능이 하락했다.
- `MANTA only`(512:0)는 속도 이득 없이 정확도만 하락하는 경향을 보였다.

### 4.11 Refined Search (2026-02-23): 고정밀 분포/총량 탐색 2차 (`09_refined_calibration_search`)

2차 실험에서는 비율, 총량, 데이터 구성(no_pilecc, stratified)을 동시에 탐색했다.

| 모델 | 로컬 전체 | 시간(s) | ScoreProxy | 서버 제출 |
|------|----------|--------|-----------|----------|
| exp_r1_fp8_static_m224_p288_t512 | 0.4476 | 292.0 | 0.5538 | **0.5050556795 (13분 2초)** |
| exp_s2_fp8_static_m320_p320_t640 | 0.4477 | 297.3 | 0.5454 | 미제출 |
| exp_r2_fp8_static_m192_p320_t512 | 0.4456 | 297.5 | 0.5426 | 미제출 |
| exp_q1_fp8_static_m256_p256_t512_no_pilecc | 0.4447 | 297.1 | 0.5423 | 미제출 |
| exp_q2_fp8_static_m256_p256_t512_stratified | 0.4418 | 295.6 | 0.5413 | 미제출 |

실패 요약 (`exp_r1` 서버 급락):
- 로컬 ScoreProxy 1위였지만 서버 점수는 mixed512(0.6219) 대비 크게 낮았다.
- `m224:p288`로 pile 비중을 높인 분포가 hidden benchmark와 어긋났을 가능성이 높다.
- 총 소요시간 13분 2초로 기존 mixed512(약 9분 41초) 대비 느려져 SpeedNorm도 손해를 봤을 가능성이 크다.
- 결론적으로 local proxy는 후보 필터링 지표로는 유효하지만, 최종 순위 보장은 하지 못함이 재확인되었다.

---

## 5. 핵심 교훈

### 5.1 로컬 vs 서버 점수 괴리

| 모델 | 로컬 전체 | 서버 점수 | 괴리 |
|------|----------|----------|------|
| exp3_fp8_static_mixed512 | 0.4523 | 0.6219616518 | +37.5% |
| exp2_fp8_static_manta256 | 0.4500 | 0.6105535201 | +35.7% |
| exp_r1_fp8_static_m224_p288_t512 | 0.4476 | 0.5050556795 | +12.8% |
| FP8 Dynamic base | 0.4322 | 0.61 | +41% |
| FP8 sel_layer0_qkv | 0.4416 | 0.50 | +13% |
| exp3_kd_lora | 0.4816 | 0.3764 | -22% |

로컬 평가(gsm8k, mmlu, arc_challenge)는 서버의 hidden 벤치마크와 분포가 다르다. 로컬에서의 개선이 서버 점수 개선을 보장하지 않으며, 오히려 역전될 수 있다.

### 5.2 Proxy 최적화의 함정 (신규)

- `exp_r1`은 로컬 proxy 기준 1위였지만 서버에서 급락했다.
- 원인: 로컬 측정 시간(전체 elapsed)이 서버 점수식의 TPOT와 정확히 같지 않고, hidden 분포도 다르기 때문이다.
- 따라서 proxy 랭킹 단독 선택은 위험하며, 서버 검증 이력이 있는 anchor 모델(mixed512)을 항상 함께 운영해야 한다.

### 5.3 “학습 기반 변경”은 하락, “정적 양자화 분포 최적화”는 상승

| 접근 | 모델 변형 정도 | 서버 결과 |
|------|-------------|----------|
| FP8 Static Mixed512 | 없음 (양자화만) | **0.6219616518 (최고)** |
| FP8 Static MANTA256 | 없음 (양자화만) | 0.6105535201 |
| FP8 Dynamic (data-free) | 없음 (양자화만) | 0.61 |
| FP8 Selective | 일부 레이어 차별화 | 0.50 |
| KD + FP8 | weight 학습으로 변경 | 0.3764 (대폭 하락) |

정리하면, “모델 weight를 학습으로 크게 바꾸는 방식”은 hidden set에서 불안정했고,  
반대로 “원본 분포를 유지한 양자화 + 캘리브레이션 분포 최적화”는 점수 개선에 유효했다.

### 5.4 점수 공식의 두 축

```
Score = max(0.5 × PerfNorm + 0.5 × SpeedNorm, 0)
PerfNorm  = Perf_model / Perf_base
SpeedNorm = 1 - TPOT_model / TPOT_base
```

- 성능(PerfNorm)과 속도(SpeedNorm)가 **50:50 동등 기여**
- PerfNorm: FP8 Static(Mixed512)이 hidden set에서 Dynamic 대비 추가 이득 달성
- SpeedNorm: FP8 Static은 Dynamic 대비 느리지만, FP16 baseline 대비는 여전히 개선 구간 유지 가능
- W4A16은 TPOT가 14% 느림 → SpeedNorm이 음수 → 점수 하락
- Layer Pruning은 TPOT 개선 0% + 정확도 하락 → 양쪽 모두 불리

### 5.5 1.2B 모델의 한계

- 레이어 수(30)가 적어 Layer Pruning 불가
- 파라미터가 적어 양자화 정밀도 손실에 민감
- FP8(8bit)가 W4(4bit)보다 유리 — 4bit는 정보 손실이 큰 반면 속도 이득이 적음

---

## 6. 실험 환경 및 파일 구조

### 6.1 프로젝트 디렉토리

```
${PROJECT_ROOT}/
  open/base_model/          # 기본 모델 (EXAONE-4.0-1.2B)
  01_pruning/code/          # Pruning + GPTQ 실험
  02_autoround/code/        # AutoRound W4A8/W4A16 실험
  03_FP8/code/              # FP8 Selective 양자화 실험
  04_W4A16/code/            # W4A16 AutoRound 비교 실험
  05_FP8_accuracy/code/     # FP8 정확도 개선 실험 (SFT, KD)
  06_final/code/            # Layer Pruning 실험
  07_new_approaches/code/   # FP8 Static 분포 실험 1차
  08_calibration_search/code/ # FP8 Static 비율 탐색
  09_refined_calibration_search/code/ # FP8 Static 정밀 탐색
  10_third_dataset_search/code/ # 3rd dataset 소량 혼합 탐색
  11_seed_ensemble_calibration_search/code/ # 시드 앙상블 캘리브레이션
  12_fp8_stability_pipeline/code/ # U1/U2/U3 안정화 파이프라인
  eval/                     # 평가 스크립트 및 결과
  submission/               # 제출 이력
```

### 6.2 평가 결과 파일

| 파일 | 날짜 | 내용 |
|------|------|------|
| eval_20260219_081915.json | 02/19 | FP8 Dynamic 초기 실험 (2개 모델) |
| eval_20260219_141406.json | 02/19 | FP8 Selective 비교 (5개 모델, 5회) |
| eval_20260220_072150.json | 02/20 | W4A16 AutoRound/GPTQ 비교 (5개 모델) |
| eval_20260220_120052.json | 02/20 | FP8 정확도 개선 (4개 모델: Dynamic, Static, Pre-SFT, KD) |
| eval_20260221_184858.json | 02/21 | KD 개선 실험 (5개 모델: base, Static, KD x3) |
| eval_20260222_062252.json | 02/22 | Layer Pruning 실험 (4개 모델: base, Static, Prune1L, Prune2L) |
| eval_20260222_115332.json | 02/22 | New Approaches (base + FP8 Static 3종 + INT8 시도) |
| eval_20260222_173304.json | 02/22 | Calibration Search 1차 (MANTA:pile 비율 sweep) |
| eval_20260223_115521.json | 02/23 | Refined Search 2차 (비율/총량/구성 동시 탐색) |
| eval_20260223_181251.json | 02/23 | Third Dataset Search (MANTA/pile/ARC 3-way 탐색) |
| eval_20260224_063151.json | 02/24 | Seed Ensemble Calibration 결과 |
| eval_20260224_123041.json | 02/24 | FP8 Stability Pipeline (U1/U2) 비교 |

### 6.3 사용 도구

| 도구 | 버전 | 용도 |
|------|------|------|
| llmcompressor | 0.9.0.1 | FP8/GPTQ 양자화 |
| compressed-tensors | 0.13.0 | 양자화 모델 직렬화 |
| auto-round | - | W4A16 AutoRound 양자화 |
| vLLM | 0.14.1 | 추론 엔진 (평가) |
| lm_eval | - | 벤치마크 평가 |
| transformers | 4.54+ | 모델 로딩/저장 |
| peft | - | LoRA 학습 |

---

## 7. 결론

본 절의 결론은 2026-02-23 시점 기준 요약이며, 2026-02-25 최종 반영 내용은 `9절`을 따른다. 2026-02-23 기준으로는 현재 최적 모델이 **FP8 Static Mixed512 (서버 0.6219616518)** 이고, 정밀 탐색 모델 `exp_r1_fp8_static_m224_p288_t512`는 로컬 proxy 1위였지만 서버 0.5050556795로 하락했다.

근본 원인은 세 가지이다:

1. 서버 벤치마크가 비공개이므로 최적화 방향을 정할 수 없다
2. 1.2B 규모 모델은 양자화/프루닝 여유가 극히 적다
3. 점수 공식이 50:50이므로, hidden set에서의 PerfNorm/SpeedNorm 균형을 맞추지 못하면 로컬 proxy 상위 모델도 서버에서 쉽게 역전된다

따라서 현 시점의 최적 전략은 **학습 기반 대규모 weight 변경(KD/SFT/Pruning)보다, FP8 Static의 캘리브레이션 분포를 정교하게 설계하되, mixed512를 앵커로 둔 보수적 제출 운영**이다.

---

## 8. 향후 개선 로드맵 (상세)

### 8.1 목표

- 단기 목표: 0.6219를 안정적으로 재현하고, 0.63+ 구간 진입 시도
- 중기 목표: 제출 3회/일 제약 하에서 “고점 + 안전빵” 포트폴리오 운영

### 8.2 우선순위 실험 축

1. **캘리브레이션 분포 비율 탐색 (최우선)**
   - 현재 best: MANTA256 + pile256
   - 다음 실험: MANTA320+pile128+datasetC64, MANTA256+pile192+datasetC64, MANTA224+pile224+datasetC64
   - 원칙: 3번째 데이터셋은 처음부터 크게 넣지 말고(10~15%), mixed512 앵커 대비 미세 변화만 확인
   - 목적: hidden set과의 분포 미스매치를 줄이면서도 기존 고점을 훼손하지 않는 안전한 확장

2. **캘리브레이션 샘플 수 탐색**
   - 256, 384, 512, 768 비교
   - 과대 샘플은 특정 분포 과적합 가능성이 있어 무조건 증가가 정답은 아님

3. **시드 앙상블 기반 모델 생성 (신규)**
   - 동일 비율(예: 256:256)에서 seed만 바꾼 모델 3~5개를 생성하고 로컬 분산을 측정
   - 권장: seed = 13, 42, 52, 77, 101
   - 목적: 단일 seed 편향을 줄이고, 서버 제출 1회 상황에서 tail risk를 완화

4. **캘리브레이션 시퀀스 길이 분포 제어**
   - 짧은 질의/중간 길이/긴 추론 샘플 비율을 고정해 scale 편향 완화
   - 동일 512샘플이라도 길이 분포 변경만으로 결과가 달라질 수 있음

5. **제출 포트폴리오 전략 고정**
   - Slot A: mixed512 계열 앵커(서버 검증 완료)
   - Slot B: 3-dataset 소량 혼합 변형
   - Slot C: FP8 Dynamic base(안전빵)

### 8.3 비추천/보류 축

- KD, SFT, Full FT: hidden concept drift 리스크가 이미 서버에서 확인됨
- Layer Pruning: 정확도 손실 대비 속도 이득이 미미
- W4A16/GPTQ: TPOT 열세로 점수식에서 구조적으로 불리
- INT8 W8A8: 현재 개발 GPU(SM120)에서 커널 비호환 에러 확인, 재현/검증 리스크 큼

### 8.4 실행 계획 (다음 3일)

1. Day 1: mixed512 동일 비율 seed sweep(3~5개) + 로컬 분산 분석
2. Day 2: 3-dataset 소량 혼합(10~15%) 실험 2~3종 + mixed 대비 A/B 평가
3. Day 3: 서버 제출은 mixed512 앵커 + seed-best 1개 + 3-dataset 1개로 포트폴리오 확정

### 8.5 성공 기준

- 서버 최고점 갱신(0.6219616518 초과)
- 동일 계열 모델의 제출 간 변동폭 축소(재현성 개선)
- “고점 모델 1개 + 차선 모델 1개 + 안전빵 모델 1개” 운영 체계 정착

---

## 9. 최신 업데이트 (2026-02-24 ~ 2026-02-25)

아래 내용은 `10_third_dataset_search`, `11_seed_ensemble_calibration_search`, `12_fp8_stability_pipeline` 및 2026-02-25 최종 제출 결과를 반영한 추가 업데이트이다.

### 9.1 Third Dataset Search (2026-02-23, `10_third_dataset_search`)

평가 파일: `comparison_20260223_181251.json`

| 모델 | gsm8k | mmlu | arc | 전체 | 시간(s) | ScoreProxy | 비고 |
|------|------:|-----:|----:|-----:|--------:|-----------:|------|
| exp_a0_fp8_static_m256_p256_a000_t512 | 0.6484 | 0.3726 | 0.3379 | 0.4530 | 290.4 | 0.5577 | mixed512 앵커 |
| exp_a1_fp8_static_m256_p224_a032_t512 | 0.6484 | 0.3696 | 0.3242 | 0.4474 | 288.8 | 0.5539 | ARC 32 치환 |
| exp_b1_fp8_static_m224_p224_a064_t512 | 0.6328 | 0.3687 | 0.3301 | 0.4439 | 291.4 | 0.5454 | ARC 64 + 분포 조정 |

핵심 관찰:
- 기존 앵커(`a0`, 256:256:0)가 여전히 상위권을 유지했다.
- 3rd dataset(ARC) 소량 혼합은 일부 모델에서 성능 분산 완화 신호가 있었으나, 앵커를 명확히 초과하지는 못했다.
- 이 결과를 바탕으로 Stage-2 시드 앙상블 후보를 `a0/a1/b1` 중심으로 선정했다.

### 9.2 Seed Ensemble Calibration (2026-02-24, `11_seed_ensemble_calibration_search`)

평가 파일: `comparison_20260224_063151.json`

| 모델 | gsm8k | mmlu | arc | 전체 | 시간(s) | ScoreProxy |
|------|------:|-----:|----:|-----:|--------:|-----------:|
| exp_se_exp_b1_fp8_static_m224_p224_a064_t512_k5_cap1024 | 0.6621 | 0.3795 | 0.3340 | 0.4585 | 293.3 | 0.5590 |
| exp_a0_fp8_static_m256_p256_a000_t512 | 0.6465 | 0.3726 | 0.3379 | 0.4523 | 293.7 | 0.5511 |
| exp_a1_fp8_static_m256_p224_a032_t512 | 0.6484 | 0.3696 | 0.3242 | 0.4474 | 293.6 | 0.5455 |
| exp_b1_fp8_static_m224_p224_a064_t512 | 0.6328 | 0.3687 | 0.3301 | 0.4439 | 292.3 | 0.5434 |

핵심 관찰:
- 시드 앙상블 캘리브레이션(`k=5`, `cap=1024`)은 로컬 기준에서 `a0` 대비 +0.0062 전체 향상을 보였다.
- 개선은 주로 GSM8K/MMLU에서 발생했고 ARC는 소폭 하락했다.
- 다만 과거 `exp_r1` 사례처럼 로컬 상위가 서버에서 역전된 전례가 있어, 서버 일반화 리스크는 별도로 관리해야 한다.

### 9.3 FP8 Stability Pipeline (2026-02-24, `12_fp8_stability_pipeline`)

평가 파일: `comparison_20260224_123041.json`

| 모델 | gsm8k | mmlu | arc | 전체 | 시간(s) | ScoreProxy |
|------|------:|-----:|----:|-----:|--------:|-----------:|
| exp_se_exp_b1_fp8_static_m224_p224_a064_t512_k5_cap1024 | 0.6602 | 0.3795 | 0.3340 | 0.4579 | 294.6 | 0.5508 |
| exp_u1_u2_klmix_static | 0.6270 | 0.4140 | 0.3281 | 0.4563 | 292.7 | 0.5522 |
| exp_a0_fp8_static_m256_p256_a000_t512 | 0.6484 | 0.3726 | 0.3379 | 0.4530 | 296.8 | 0.5414 |
| exp_u1_bon6_static_anchor | 0.4922 | 0.4203 | 0.3242 | 0.4122 | 295.9 | 0.4951 |

핵심 관찰:
- `exp_u1_u2_klmix_static`는 로컬에서 SpeedNorm 포함 점수(ScoreProxy) 기준 상위였으나,
- `U1 only` 앵커(`exp_u1_bon6_static_anchor`)가 크게 하락하며 학습 기반 접근의 분산이 큼을 확인했다.
- 즉, 학습 기반 개선은 “상향 여지”는 있으나 hidden 분포에서 실패 확률도 동시에 큰 고변동 전략이다.

### 9.4 최종 서버 제출 결과 (2026-02-25)

아래 두 제출은 대회 종료 직전 최종 검증으로 수행되었다.

| 제출 파일 | 제출 시각 | 서버 점수 | 총 소요시간 |
|----------|-----------|----------:|------------:|
| exp3_fp8_static_mixed512.zip | 2026-02-25 02:35:19 | 0.6194291468 | 9분 46초 |
| u1_u2_klmix_static.zip | 2026-02-25 02:28:08 | 0.477602577 | 12분 19초 |

결과 해석:
- `mixed512` 재제출 점수(0.6194291468)는 기존 최고(0.6219616518) 대비 `-0.00253` 하락했지만, 동일 계열 내 변동 범위로 해석 가능하다.
- `u1_u2_klmix_static`는 로컬 대비 서버에서 크게 낮게 형성되어(hidden mismatch) 최종 제출 전략으로는 비우호적이었다.
- 최종적으로 서버 최고점은 여전히 `exp3_fp8_static_mixed512`의 **0.6219616518**로 유지된다.

### 9.5 업데이트된 최종 결론

- “양자화-only + 캘리브레이션 분포 최적화” 계열(FP8 static mixed)이 서버에서 가장 안정적이었다.
- 학습 기반(U1/U2/KL mix)은 로컬 잠재력은 있으나 서버 일반화 리스크가 매우 컸다.
- 제출 제약(일 3회, 팀 분할 제출) 상황에서는 고변동 모델 단독 배팅보다, `mixed512` 같은 서버 검증 앵커를 중심으로 운영하는 전략이 합리적이다.

---

## 10. 워크스페이스 점검 결과 (2026-02-25)

전체 디렉토리(`${PROJECT_ROOT}`) 기준 점검 결과:

1. 노트북 무결성
   - `07~12` 실험 노트북 6개(JSON/코드 셀 파싱) 정상.
2. 제출 ZIP 구조
   - `submit_*.zip` 36개 점검 결과 모두 `model/` 단일 최상위 구조 및 핵심 파일(`config.json`, `*.safetensors`) 확인.
3. 핵심 모델 아티팩트
   - `07 mixed512`, `10 a0`, `11 seed-ensemble(b1)`, `12 u1_u2_klmix`의 `model/config.json` 및 가중치 파일 존재 확인.
4. 평가 로그/결과
   - `eval/comparison_*.json` 최신 파일이 `20260224_123041`까지 정상 누적되어 있으며, 10~12 단계 로컬 분석의 근거 파일이 보존됨.

참고:
- `01~02` 일부 초기 노트북은 셀 내 쉘 구문/표현식으로 인해 정적 AST 파싱에서 경고가 발생할 수 있으나, 이는 본 최신 파이프라인(07~12) 업데이트 범위와 직접 충돌하지 않는다.
