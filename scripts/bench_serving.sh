#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8020}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4-Flash}"
TOKENIZER_PATH="${TOKENIZER_PATH:?TOKENIZER_PATH is required}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-128}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-200}"
BENCH_CONCURRENCY_LIST="${BENCH_CONCURRENCY_LIST:-5 10 80}"
RESULT_ROOT="${RESULT_ROOT:-results}"

if ! command -v vllm >/dev/null 2>&1; then
  printf '负载机缺少 vllm CLI；只需安装客户端，不要在服务容器内临时升级框架。\n' >&2
  exit 1
fi

run_dir="$RESULT_ROOT/serving/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"

for concurrency in $BENCH_CONCURRENCY_LIST; do
  log_file="$run_dir/c${concurrency}.log"
  printf '[RUN] concurrency=%s input=%s output=%s prompts=%s\n' \
    "$concurrency" "$BENCH_INPUT_LEN" "$BENCH_OUTPUT_LEN" "$BENCH_NUM_PROMPTS"

  vllm bench serve \
    --backend openai-chat \
    --base-url "$API_BASE_URL" \
    --endpoint /v1/chat/completions \
    --model "$SERVED_MODEL_NAME" \
    --tokenizer "$TOKENIZER_PATH" \
    --dataset-name random \
    --random-input-len "$BENCH_INPUT_LEN" \
    --random-output-len "$BENCH_OUTPUT_LEN" \
    --num-prompts "$BENCH_NUM_PROMPTS" \
    --request-rate inf \
    --max-concurrency "$concurrency" \
    --temperature 0 \
    --top-p 1 \
    --ignore-eos 2>&1 | tee "$log_file"
done

printf '[DONE] 结果保存在 %s\n' "$run_dir"
