#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-/data/models/Tensor-0.1-Flash-35B-A3B}"
DEPLOY_DIR="${MODEL_PATH}/deployment"
RESULT_ROOT="${MODEL_PATH}/results/qwen36-vllm-npu3"

curl -fsS http://127.0.0.1:9108/health >/dev/null || { echo "服务未就绪" >&2; exit 1; }
conda env list | awk '{print $1}' | grep -Fxq vllm_bench || { echo "找不到 vllm_bench" >&2; exit 1; }
mkdir -p "${RESULT_ROOT}/bench" "${RESULT_ROOT}/logs"

conda run --no-capture-output -n vllm_bench \
  python "${DEPLOY_DIR}/bench_concurrency.py" \
  --base-url http://127.0.0.1:9108 \
  --model Tensor-0.1-Flash-35B-A3B \
  --concurrency 5 10 20 30 \
  --requests "${BENCH_REQUESTS:-60}" \
  --max-tokens "${BENCH_OUTPUT_TOKENS:-256}" \
  --timeout "${BENCH_TIMEOUT:-1800}" \
  --output-dir "${RESULT_ROOT}/bench" \
  2>&1 | tee "${RESULT_ROOT}/logs/benchmark-$(date +%Y%m%d-%H%M%S).log"
