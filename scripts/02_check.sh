#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-/data/models/Tensor-0.1-Flash-35B-A3B}"
RESULT_ROOT="${MODEL_PATH}/results/qwen36-vllm-npu3"
CONTAINER_NAME=qwen36-flash-npu3
TIMEOUT="${HEALTH_TIMEOUT:-1800}"

[[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" == true ]] || {
  echo "容器未运行" >&2; exit 1;
}
deadline=$((SECONDS + TIMEOUT))
until curl -fsS http://127.0.0.1:9108/health >/dev/null 2>&1; do
  [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" == true ]] || {
    docker logs --tail 100 "${CONTAINER_NAME}" >&2; exit 1;
  }
  (( SECONDS < deadline )) || { echo "等待服务超时" >&2; exit 1; }
  echo "模型加载中……"
  sleep 10
done

docker logs "${CONTAINER_NAME}" >"${RESULT_ROOT}/logs/service.log" 2>&1
npu-smi info
curl -fsS http://127.0.0.1:9108/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Tensor-0.1-Flash-35B-A3B","messages":[{"role":"user","content":"用三点解释 MoE 路由机制。"}],"temperature":0,"max_tokens":128}' \
  | tee "${RESULT_ROOT}/logs/check-response.json"
echo
echo "检查通过。"
