"""
lm_eval 반복 평가 & 중앙값 기반 모델 비교 스크립트

사용법:
  # 단일 모델 3회 평가
  python run_eval.py --model_path /path/to/model

  # 여러 모델 비교 (각 3회)
  python run_eval.py --model_path /path/to/model_A /path/to/model_B

  # 평가 횟수 변경
  python run_eval.py --model_path /path/to/model --n_runs 5

  # 기존 결과만 분석 (평가 실행 없이)
  python run_eval.py --model_path /path/to/model --analyze_only

  # task / limit 변경
  python run_eval.py --model_path /path/to/model --tasks gsm8k,mmlu --limit 256
"""

import argparse
import gc
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median, mean, stdev

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DEFAULT_TASKS = "gsm8k,mmlu,arc_challenge"
DEFAULT_LIMIT = 512
DEFAULT_N_RUNS = 3
DEFAULT_GPU_UTIL = 0.85
DEFAULT_MAX_MODEL_LEN = 2048

# 각 task에서 추출할 주요 메트릭
TASK_METRICS = {
    "gsm8k": "exact_match,strict-match",
    "mmlu": "acc,none",
    "arc_challenge": "acc_norm,none",
}


def cleanup_gpu():
    """GPU 메모리 정리: 잔여 vllm/lm_eval 프로세스 종료 + 캐시 해제"""
    # 1) 잔여 vllm 관련 프로세스 강제 종료
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            for pid_str in result.stdout.strip().split("\n"):
                pid = int(pid_str.strip())
                if pid != os.getpid():
                    try:
                        os.kill(pid, signal.SIGKILL)
                        print(f"    잔여 GPU 프로세스 종료: PID {pid}")
                    except ProcessLookupError:
                        pass
    except Exception:
        pass

    # 2) Python 레벨 정리
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    # 3) GPU 메모리 해제 대기 (최대 15초)
    for _ in range(15):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            mem_used = int(result.stdout.strip().split("\n")[0])
            if mem_used < 1000:  # 1GB 미만이면 정리 완료
                break
        except Exception:
            break
        time.sleep(1)

    # 현재 GPU 메모리 출력
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        used, total = result.stdout.strip().split(", ")
        print(f"    GPU 메모리: {used}MB / {total}MB")
    except Exception:
        pass


def run_lm_eval(model_path: str, output_dir: str, tasks: str, limit: int,
                gpu_util: float, max_model_len: int, run_idx: int) -> str:
    """lm_eval 1회 실행, 결과 디렉토리 경로 반환"""
    run_output = os.path.join(output_dir, f"run_{run_idx}")
    os.makedirs(run_output, exist_ok=True)

    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", (
            f"pretrained={model_path},"
            f"gpu_memory_utilization={gpu_util},"
            f"enable_thinking=False,"
            f"max_model_len={max_model_len}"
        ),
        "--tasks", tasks,
        "--limit", str(limit),
        "--output_path", run_output,
        "--apply_chat_template",
        "--batch_size", "auto",
    ]

    print(f"\n{'='*60}")
    print(f"  Run {run_idx + 1} 시작 - {datetime.now().strftime('%H:%M:%S')}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  [ERROR] Run {run_idx + 1} 실패 (return code: {result.returncode})")
        return None

    print(f"  Run {run_idx + 1} 완료 - 소요시간: {elapsed:.1f}초")

    # run별 메타 정보 저장 (속도 분석용)
    with open(os.path.join(run_output, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_idx": run_idx + 1,
                "elapsed_sec": elapsed,
                "tasks": tasks,
                "limit": limit,
                "gpu_util": gpu_util,
                "max_model_len": max_model_len,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # GPU 메모리 정리
    print(f"  GPU 메모리 정리 중...")
    cleanup_gpu()

    return run_output


def find_result_json(run_output_dir: str) -> str | None:
    """lm_eval 결과 JSON 파일 경로 찾기 (가장 최신 파일 반환)"""
    candidates = []
    for root, dirs, files in os.walk(run_output_dir):
        for f in files:
            if f.startswith("results_") and f.endswith(".json"):
                candidates.append(os.path.join(root, f))
    if not candidates:
        return None
    # 파일명에 타임스탬프가 포함되므로 정렬 시 가장 최신이 마지막
    return sorted(candidates)[-1]


def parse_results(json_path: str, tasks: list[str]) -> dict:
    """결과 JSON에서 주요 메트릭 추출"""
    with open(json_path, "r") as f:
        data = json.load(f)

    results = {}
    for task in tasks:
        if task in data.get("results", {}):
            metric_key = TASK_METRICS.get(task)
            if metric_key and metric_key in data["results"][task]:
                results[task] = data["results"][task][metric_key]
            else:
                # fallback: 첫 번째 숫자 메트릭 사용
                for k, v in data["results"][task].items():
                    if isinstance(v, (int, float)) and "stderr" not in k and k != "alias":
                        results[task] = v
                        break
    return results


def compute_stats(scores: list[float]) -> dict:
    """점수 리스트에 대한 통계 산출"""
    stats = {
        "median": median(scores),
        "mean": mean(scores),
        "min": min(scores),
        "max": max(scores),
        "n": len(scores),
    }
    if len(scores) >= 2:
        stats["stdev"] = stdev(scores)
    else:
        stats["stdev"] = 0.0
    return stats


def compute_competition_proxy_score(
    perf_model: float, perf_base: float, time_model: float, time_base: float
) -> tuple[float, float, float] | None:
    """
    대회 공식의 proxy 점수 계산.

    주의: 여기서 time_*은 TPOT가 아닌 run elapsed median이므로
    Score는 공식 점수의 근사치(proxy)이다.
    """
    if perf_base <= 0 or time_base <= 0:
        return None
    perf_norm = perf_model / perf_base
    speed_norm = 1.0 - (time_model / time_base)
    score_proxy = max(0.5 * perf_norm + 0.5 * speed_norm, 0.0)
    return perf_norm, speed_norm, score_proxy


def validate_model_dir(model_path: str) -> tuple[bool, str | None]:
    """모델 경로 기본 유효성 검사"""
    if not os.path.isdir(model_path):
        return False, f"모델 경로가 디렉토리가 아님: {model_path}"

    required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(model_path, f))]
    if missing:
        return False, f"필수 파일 누락: {missing}"

    has_safetensors = any(
        name.endswith(".safetensors") for name in os.listdir(model_path)
    )
    if not has_safetensors:
        return False, "safetensors 파일이 없습니다."

    return True, None


def load_run_elapsed_sec(run_output_dir: str) -> float | None:
    """run_meta.json에서 실행 시간(초) 로드"""
    meta_path = os.path.join(run_output_dir, "run_meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        elapsed = meta.get("elapsed_sec")
        return float(elapsed) if elapsed is not None else None
    except Exception:
        return None


def make_model_name(model_path: str) -> str:
    """모델 경로에서 고유한 이름 생성 (상위 폴더 포함)"""
    parts = Path(model_path).parts
    # 'model'만으로 끝나면 상위 폴더명도 포함
    meaningful = [p for p in parts if p not in ("/", "model")]
    if len(meaningful) >= 2:
        return f"{meaningful[-2]}__{meaningful[-1]}"
    elif meaningful:
        return meaningful[-1]
    return Path(model_path).name


def evaluate_model(model_path: str, base_output_dir: str, tasks: str,
                   limit: int, n_runs: int, gpu_util: float,
                   max_model_len: int, analyze_only: bool) -> dict:
    """단일 모델에 대해 N회 평가 후 통계 반환"""
    model_name = make_model_name(model_path)
    output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    task_list = [t.strip() for t in tasks.split(",")]
    is_valid, error_msg = validate_model_dir(model_path)
    if not is_valid:
        return {
            "model_name": model_name,
            "model_path": model_path,
            "valid": False,
            "error": error_msg,
            "stats": {task: None for task in task_list},
            "overall_median": None,
            "time_stats": None,
            "quality_per_sec": None,
        }

    # ── 평가 실행 ──
    if not analyze_only:
        # 이전 run 결과 제거 (stale 결과 혼입 방지)
        for run_dir in Path(output_dir).glob("run_*"):
            if run_dir.is_dir():
                shutil.rmtree(run_dir, ignore_errors=True)

        for i in range(n_runs):
            run_lm_eval(model_path, output_dir, tasks, limit,
                        gpu_util, max_model_len, i)

    # ── 결과 수집 ──
    all_runs = {}  # {task: [score1, score2, ...]}
    for task in task_list:
        all_runs[task] = []

    run_dirs = sorted(Path(output_dir).glob("run_*"))
    if not run_dirs:
        print(f"  [WARNING] {model_name}: 결과 디렉토리를 찾을 수 없습니다.")
        return None

    elapsed_runs = []
    for run_dir in run_dirs:
        json_path = find_result_json(str(run_dir))
        if json_path is None:
            print(f"  [WARNING] {run_dir}에서 결과 JSON을 찾을 수 없습니다.")
            continue

        scores = parse_results(json_path, task_list)
        for task in task_list:
            if task in scores:
                all_runs[task].append(scores[task])

        elapsed = load_run_elapsed_sec(str(run_dir))
        if elapsed is not None:
            elapsed_runs.append(elapsed)

    # ── 통계 산출 ──
    stats = {}
    for task in task_list:
        if all_runs[task]:
            stats[task] = compute_stats(all_runs[task])
            stats[task]["scores"] = all_runs[task]
        else:
            stats[task] = None

    # 전체 평균(median) 점수 = 각 task median의 평균
    task_medians = [s["median"] for s in stats.values() if s is not None]
    overall_median = mean(task_medians) if task_medians else None
    time_stats = compute_stats(elapsed_runs) if elapsed_runs else None
    quality_per_sec = (
        overall_median / time_stats["median"]
        if (overall_median is not None and time_stats and time_stats["median"] > 0)
        else None
    )

    return {
        "model_name": model_name,
        "model_path": model_path,
        "valid": True,
        "error": None,
        "stats": stats,
        "overall_median": overall_median,
        "time_stats": time_stats,
        "quality_per_sec": quality_per_sec,
    }


def print_model_report(result: dict):
    """단일 모델 평가 결과 출력"""
    print(f"\n{'='*70}")
    print(f"  모델: {result['model_name']}")
    print(f"  경로: {result['model_path']}")
    print(f"{'='*70}")
    if not result.get("valid", True):
        print(f"  [INVALID] {result.get('error', 'unknown error')}")
        print(f"{'='*70}")
        return

    for task, stats in result["stats"].items():
        if stats is None:
            print(f"\n  [{task}] 결과 없음")
            continue

        metric = TASK_METRICS.get(task, "?")
        print(f"\n  [{task}] (metric: {metric})")
        print(f"    실행 횟수: {stats['n']}회")
        print(f"    각 점수:   {['%.4f' % s for s in stats['scores']]}")
        print(f"    ─────────────────────────")
        print(f"    중앙값:    {stats['median']:.4f}  ← 기준 점수")
        print(f"    평균:      {stats['mean']:.4f}")
        print(f"    최소:      {stats['min']:.4f}")
        print(f"    최대:      {stats['max']:.4f}")
        print(f"    표준편차:  {stats['stdev']:.4f}")

    print(f"\n  {'─'*40}")
    if result.get("overall_median") is not None:
        print(f"  전체 중앙값 평균: {result['overall_median']:.4f}")
    else:
        print(f"  전체 중앙값 평균: N/A")
    if result.get("time_stats"):
        t = result["time_stats"]
        print(f"  실행시간 중앙값: {t['median']:.1f}초")
        print(f"  실행시간 평균:   {t['mean']:.1f}초")
        if result.get("quality_per_sec") is not None:
            print(f"  효율(정확도/초): {result['quality_per_sec']:.6f}")
    else:
        print(f"  실행시간 통계:   없음(run_meta.json 미발견)")
    print(f"{'='*70}")


def print_comparison(results: list[dict], baseline_model_idx: int):
    """여러 모델 비교 결과 출력"""
    valid_results = [r for r in results if r.get("valid", True) and r.get("overall_median") is not None]
    if len(valid_results) < 2:
        print("\n  [INFO] 비교 가능한(유효한) 모델이 2개 미만입니다.")
        return

    baseline = None
    if 0 <= baseline_model_idx < len(results):
        candidate = results[baseline_model_idx]
        if candidate.get("valid", True) and candidate.get("overall_median") is not None:
            baseline = candidate

    print(f"\n\n{'#'*70}")
    print(f"  모델 비교 (중앙값 + 실행시간 기준)")
    print(f"{'#'*70}")

    # 테이블 헤더
    tasks = list(valid_results[0]["stats"].keys())
    header = f"  {'모델':<30}"
    for task in tasks:
        header += f"  {task:<15}"
    header += f"  {'전체':>8}  {'시간(s)':>10}  {'효율':>10}"
    if baseline is not None:
        header += f"  {'PerfNorm':>9}  {'SpeedNorm':>10}  {'ScoreProxy':>11}"
    print(header)
    print(f"  {'─'*len(header)}")

    # 각 모델 행
    for r in sorted(valid_results, key=lambda x: x["overall_median"], reverse=True):
        row = f"  {r['model_name']:<30}"
        for task in tasks:
            s = r["stats"].get(task)
            val = f"{s['median']:.4f}" if s else "N/A"
            row += f"  {val:<15}"
        t = r.get("time_stats")
        tval = f"{t['median']:.1f}" if t else "N/A"
        qps = r.get("quality_per_sec")
        qps_val = f"{qps:.6f}" if qps is not None else "N/A"
        row += f"  {r['overall_median']:.4f}  {tval:>10}  {qps_val:>10}"
        if baseline is not None:
            base_time = (baseline.get("time_stats") or {}).get("median")
            this_time = (r.get("time_stats") or {}).get("median")
            proxy = None
            if base_time is not None and this_time is not None:
                proxy = compute_competition_proxy_score(
                    perf_model=r["overall_median"],
                    perf_base=baseline["overall_median"],
                    time_model=this_time,
                    time_base=base_time,
                )
            if proxy is None:
                row += f"  {'N/A':>9}  {'N/A':>10}  {'N/A':>11}"
            else:
                perf_norm, speed_norm, score_proxy = proxy
                row += f"  {perf_norm:>9.4f}  {speed_norm:>+10.4f}  {score_proxy:>11.4f}"
        print(row)

    # 추천
    best = max(
        valid_results,
        key=lambda x: (
            x["quality_per_sec"] if x.get("quality_per_sec") is not None else -1.0
        ),
    )
    print(f"\n  >>> 추천 모델 (효율 기준): {best['model_name']}")
    print(f"      전체 중앙값 평균: {best['overall_median']:.4f}")
    if best.get("time_stats"):
        print(f"      실행시간 중앙값: {best['time_stats']['median']:.1f}초")
    if best.get("quality_per_sec") is not None:
        print(f"      효율(정확도/초): {best['quality_per_sec']:.6f}")
    print(f"      경로: {best['model_path']}")

    # 기준 모델 대비 상대 지표
    if baseline is not None:
        base_acc = baseline["overall_median"]
        baseline_time_stats = baseline.get("time_stats") or {}
        base_time = baseline_time_stats.get("median")
        print(f"\n  기준 모델: {baseline['model_name']} (index={baseline_model_idx})")
        for r in valid_results:
            rel_acc = (r["overall_median"] / base_acc) if base_acc > 0 else None
            rel_time_reduction = None
            r_time_stats = r.get("time_stats") or {}
            r_time = r_time_stats.get("median")
            if base_time and r_time:
                rel_time_reduction = (base_time - r_time) / base_time

            rel_acc_str = f"{rel_acc:.4f}" if rel_acc is not None else "N/A"
            rel_time_str = (
                f"{rel_time_reduction:+.2%}" if rel_time_reduction is not None else "N/A"
            )
            score_proxy_str = "N/A"
            if base_time and r_time:
                proxy = compute_competition_proxy_score(
                    perf_model=r["overall_median"],
                    perf_base=base_acc,
                    time_model=r_time,
                    time_base=base_time,
                )
                if proxy is not None:
                    score_proxy_str = f"{proxy[2]:.4f}"
            print(
                f"    - {r['model_name']}: 정확도비 {rel_acc_str}, 시간감소율 {rel_time_str}, 점수프록시 {score_proxy_str}"
            )
    else:
        print("\n  [INFO] 기준 모델이 유효하지 않아 상대 지표를 생략합니다.")

    # 상세 JSON 저장
    summary_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    summary = []
    for r in results:
        perf_norm = None
        speed_norm = None
        score_proxy = None
        if (
            baseline is not None
            and r.get("overall_median") is not None
            and (r.get("time_stats") or {}).get("median") is not None
            and (baseline.get("time_stats") or {}).get("median") is not None
        ):
            proxy = compute_competition_proxy_score(
                perf_model=r["overall_median"],
                perf_base=baseline["overall_median"],
                time_model=(r.get("time_stats") or {})["median"],
                time_base=(baseline.get("time_stats") or {})["median"],
            )
            if proxy is not None:
                perf_norm, speed_norm, score_proxy = proxy

        entry = {
            "model_name": r["model_name"],
            "model_path": r["model_path"],
            "overall_median": r["overall_median"],
            "valid": r.get("valid", True),
            "error": r.get("error"),
            "time_stats": r.get("time_stats"),
            "quality_per_sec": r.get("quality_per_sec"),
            "perf_norm_proxy": perf_norm,
            "speed_norm_proxy": speed_norm,
            "score_proxy": score_proxy,
        }
        for task, stats in r["stats"].items():
            if stats:
                entry[f"{task}_median"] = stats["median"]
                entry[f"{task}_scores"] = stats["scores"]
        summary.append(entry)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="lm_eval 반복 평가 & 중앙값 기반 모델 비교"
    )
    parser.add_argument(
        "--model_path", nargs="+", required=True,
        help="평가할 모델 경로 (여러 개 가능)"
    )
    parser.add_argument(
        "--n_runs", type=int, default=DEFAULT_N_RUNS,
        help=f"모델당 평가 반복 횟수 (default: {DEFAULT_N_RUNS})"
    )
    parser.add_argument(
        "--tasks", type=str, default=DEFAULT_TASKS,
        help=f"평가 task (default: {DEFAULT_TASKS})"
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"task당 평가 샘플 수 (default: {DEFAULT_LIMIT})"
    )
    parser.add_argument(
        "--gpu_util", type=float, default=DEFAULT_GPU_UTIL,
        help=f"GPU memory utilization (default: {DEFAULT_GPU_UTIL})"
    )
    parser.add_argument(
        "--max_model_len", type=int, default=DEFAULT_MAX_MODEL_LEN,
        help=f"최대 모델 시퀀스 길이 (default: {DEFAULT_MAX_MODEL_LEN})"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="결과 저장 디렉토리 (default: ./eval_results)"
    )
    parser.add_argument(
        "--analyze_only", action="store_true",
        help="평가 실행 없이 기존 결과만 분석"
    )
    parser.add_argument(
        "--baseline_model_idx", type=int, default=0,
        help="비교 시 기준 모델의 입력 순서 index (default: 0)"
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")

    print(f"\n{'#'*60}")
    print(f"  lm_eval 반복 평가 스크립트")
    print(f"{'#'*60}")
    print(f"  모델 수:     {len(args.model_path)}개")
    print(f"  반복 횟수:   {args.n_runs}회")
    print(f"  Tasks:       {args.tasks}")
    print(f"  Limit:       {args.limit}")
    print(f"  GPU util:    {args.gpu_util}")
    print(f"  Max len:     {args.max_model_len}")
    print(f"  기준 모델:   index {args.baseline_model_idx}")
    print(f"  결과 저장:   {args.output_dir}")
    if args.analyze_only:
        print(f"  모드:        기존 결과 분석만 (--analyze_only)")
    print(f"{'#'*60}")

    results = []
    for idx, model_path in enumerate(args.model_path):
        print(f"\n  [{idx + 1}/{len(args.model_path)}] 모델 평가 중: {model_path}")
        result = evaluate_model(
            model_path=model_path,
            base_output_dir=args.output_dir,
            tasks=args.tasks,
            limit=args.limit,
            n_runs=args.n_runs,
            gpu_util=args.gpu_util,
            max_model_len=args.max_model_len,
            analyze_only=args.analyze_only,
        )
        if result:
            results.append(result)

    # ── 모든 평가 완료 후 결과 한번에 출력 ──
    if not results:
        print("\n  [ERROR] 평가 결과가 없습니다.")
        return

    print(f"\n\n{'#'*70}")
    print(f"  전체 평가 결과 ({len(results)}개 모델)")
    print(f"{'#'*70}")

    for result in results:
        print_model_report(result)

    if len(results) >= 2:
        print_comparison(results, args.baseline_model_idx)

    # JSON 저장
    summary_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    summary = []
    for r in results:
        entry = {
            "model_name": r["model_name"],
            "model_path": r["model_path"],
            "overall_median": r["overall_median"],
            "valid": r.get("valid", True),
            "error": r.get("error"),
            "time_stats": r.get("time_stats"),
            "quality_per_sec": r.get("quality_per_sec"),
        }
        for task, stats in r["stats"].items():
            if stats:
                entry[f"{task}_median"] = stats["median"]
                entry[f"{task}_scores"] = stats["scores"]
        summary.append(entry)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {summary_path}")


if __name__ == "__main__":
    main()
