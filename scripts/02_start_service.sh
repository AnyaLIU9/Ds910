#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/qwen36.conf"
require_container_running

if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${SERVICE_PORT}$"; then
  die "宿主机端口 ${SERVICE_PORT} 已被占用"
fi

if docker exec "${CONTAINER_NAME}" bash -lc 'test -s /results/service.pid && kill -0 "$(cat /results/service.pid)" 2>/dev/null'; then
  die "模型服务已经在运行；不会重复启动"
fi

docker exec \
  -e "SERVICE_PORT=${SERVICE_PORT}" \
  -e "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}" \
  -e "MAX_MODEL_LEN=${MAX_MODEL_LEN}" \
  -e "MAX_NUM_SEQS=${MAX_NUM_SEQS}" \
  -e "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}" \
  -e "MEMORY_UTILIZATION=${MEMORY_UTILIZATION}" \
  "${CONTAINER_NAME}" bash -lc '
    : > /results/logs/service.log
    nohup vllm serve /model \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host 0.0.0.0 --port "${SERVICE_PORT}" \
      --tensor-parallel-size 1 \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --gpu-memory-utilization "${MEMORY_UTILIZATION}" \
      --enforce-eager \
      --trust-remote-code \
      >> /results/logs/service.log 2>&1 &
    echo $! > /results/service.pid
  '

echo "服务已提交，加载模型需要时间。"
echo "日志：tail -f ${RESULT_ROOT}/logs/service.log"
echo "下一步：bash ${DEPLOY_DIR}/03_check_service.sh"
