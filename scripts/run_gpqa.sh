#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPQA_DIR="${GPQA_DIR:-$repo_root/benchmarks/gpqa}"
if [[ "$GPQA_DIR" != /* ]]; then
  GPQA_DIR="$repo_root/$GPQA_DIR"
fi

EVAL_API_URL="${EVAL_API_URL:-http://127.0.0.1:8020/v1}"
EVAL_API_KEY="${EVAL_API_KEY:-EMPTY}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4-Flash}"
GPQA_ROUNDS="${GPQA_ROUNDS:-3}"
GPQA_MAX_TOKENS="${GPQA_MAX_TOKENS:-32768}"
GPQA_TIMEOUT_SECONDS="${GPQA_TIMEOUT_SECONDS:-60000}"
RESULT_ROOT="${RESULT_ROOT:-$repo_root/results}"

if ! command -v evalscope >/dev/null 2>&1; then
  printf '缺少 evalscope CLI；请从内网 PyPI 安装并固定版本。\n' >&2
  exit 1
fi
if [[ ! -f "$GPQA_DIR/gpqa_diamond.csv" ]]; then
  printf '缺少 %s/gpqa_diamond.csv；执行 git submodule update --init benchmarks/gpqa。\n' "$GPQA_DIR" >&2
  exit 1
fi

dataset_args="{\"gpqa_diamond\":{\"dataset_id\":\"$GPQA_DIR\"}}"
generation_config="{\"max_tokens\":$GPQA_MAX_TOKENS,\"temperature\":1.0,\"top_p\":1.0,\"stream\":true}"
run_root="$RESULT_ROOT/gpqa/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_root"

printf '[INFO] GPQA-Diamond: 3 rounds, thinking disabled at the serving configuration, API concurrency=1.\n'
for round in $(seq 1 "$GPQA_ROUNDS"); do
  round_dir="$run_root/round-$round"
  printf '[RUN] GPQA-Diamond round %s/%s\n' "$round" "$GPQA_ROUNDS"
  evalscope eval \
    --model "$SERVED_MODEL_NAME" \
    --model-id "$SERVED_MODEL_NAME" \
    --eval-type openai_api \
    --api-url "$EVAL_API_URL" \
    --api-key "$EVAL_API_KEY" \
    --datasets gpqa_diamond \
    --dataset-args "$dataset_args" \
    --eval-batch-size 1 \
    --generation-config "$generation_config" \
    --seed "$round" \
    --timeout "$GPQA_TIMEOUT_SECONDS" \
    --work-dir "$round_dir" \
    --no-timestamp
done

printf '[DONE] 三轮结果保存在 %s；确认每轮样本数为 198，再计算均值和标准差。\n' "$run_root"
