#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/qwen36.conf"

curl --fail --silent "http://127.0.0.1:${SERVICE_PORT}/health" >/dev/null || \
  die "服务健康检查失败；请先执行 03_check_service.sh"
command -v conda >/dev/null 2>&1 || die "宿主机找不到 conda"
conda env list | awk '{print $1}' | grep -Fxq vllm_bench || die "找不到 Conda 环境 vllm_bench"

mkdir -p "${RESULT_ROOT}/bench"
conda run --no-capture-output -n vllm_bench \
  python "${DEPLOY_DIR}/bench_concurrency.py" \
  --base-url "http://127.0.0.1:${SERVICE_PORT}" \
  --model "${SERVED_MODEL_NAME}" \
  --concurrency 5 10 20 30 \
  --requests "${BENCH_REQUESTS:-60}" \
  --max-tokens "${BENCH_OUTPUT_TOKENS:-256}" \
  --timeout "${BENCH_TIMEOUT:-1800}" \
  --output-dir "${RESULT_ROOT}/bench" \
  2>&1 | tee "${RESULT_ROOT}/logs/benchmark-$(date +%Y%m%d-%H%M%S).log"
