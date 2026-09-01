#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  printf '缺少配置文件 %s；先执行 cp config/env.example .env 并填写内网 URL。\n' "$ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${CANN_RECIPES_URL:?CANN_RECIPES_URL is required}"
: "${CANN_RECIPES_REV:?CANN_RECIPES_REV is required}"
: "${KTRANSFORMERS_URL:?KTRANSFORMERS_URL is required}"
: "${SGLANG_URL:?SGLANG_URL is required}"
: "${LLAMA_CPP_URL:?LLAMA_CPP_URL is required}"

mkdir -p "$SOURCE_ROOT" "$MODEL_ROOT"

sync_repo() {
  local name="$1"
  local url="$2"
  local revision="$3"
  local destination="$SOURCE_ROOT/$name"

  if [[ ! -d "$destination/.git" ]]; then
    git clone "$url" "$destination"
  fi
  git -C "$destination" fetch --all --tags --prune
  git -C "$destination" checkout --detach "$revision"
  printf '[SOURCE] %s %s\n' "$name" "$(git -C "$destination" rev-parse HEAD)"
}

sync_repo cann-recipes-infer "$CANN_RECIPES_URL" "$CANN_RECIPES_REV"
sync_repo ktransformers "$KTRANSFORMERS_URL" "${KTRANSFORMERS_REV:?}"
sync_repo sglang "$SGLANG_URL" "${SGLANG_REV:?}"
sync_repo llama.cpp "$LLAMA_CPP_URL" "${LLAMA_CPP_REV:?}"

if [[ "${CONFIRM_LARGE_DOWNLOAD:-0}" != "1" ]]; then
  printf '[SKIP] 模型下载约 425 GiB。确认路径和内网带宽后设置 CONFIRM_LARGE_DOWNLOAD=1 再运行。\n'
  exit 0
fi

if ! command -v modelscope >/dev/null 2>&1; then
  printf '缺少 modelscope CLI，请从内部 PyPI 安装后重试。\n' >&2
  exit 1
fi
if ! command -v hf >/dev/null 2>&1; then
  printf '缺少 hf CLI，请从内部 PyPI 安装 huggingface_hub 后重试。\n' >&2
  exit 1
fi

modelscope download \
  --model "${W8A8_MODEL_ID:?}" \
  --revision "${W8A8_MODEL_REVISION:-master}" \
  --local_dir "${W8A8_MODEL_PATH:?}"

hf download "${MXFP4_MODEL_ID:?}" \
  --revision "${MXFP4_MODEL_REVISION:-main}" \
  --local-dir "${MXFP4_SOURCE_PATH:?}"

du -sh "${W8A8_MODEL_PATH}" "${MXFP4_SOURCE_PATH}"
printf '[DONE] 源码和两套模型权重已准备；下一步按 DEPLOYMENT.md 打补丁、编译和转换 GGUF。\n'
