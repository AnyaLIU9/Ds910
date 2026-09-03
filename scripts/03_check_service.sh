#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/qwen36.conf"
require_container_running

echo "===== 容器"
docker inspect "${CONTAINER_NAME}" \
  --format 'name={{.Name}} image={{.Config.Image}} running={{.State.Running}}'
echo "===== 服务进程"
docker exec "${CONTAINER_NAME}" bash -lc '
  if test -s /results/service.pid && kill -0 "$(cat /results/service.pid)" 2>/dev/null; then
    echo "running pid=$(cat /results/service.pid)"
  else
    echo "not running"
    exit 1
  fi'
echo "===== NPU"
npu-smi info
echo "===== 等待健康检查（最多 ${HEALTH_TIMEOUT:-1800} 秒）"
deadline=$((SECONDS + ${HEALTH_TIMEOUT:-1800}))
until curl --fail --silent "http://127.0.0.1:${SERVICE_PORT}/health" >/dev/null 2>&1; do
  if ! docker exec "${CONTAINER_NAME}" bash -lc \
    'test -s /results/service.pid && kill -0 "$(cat /results/service.pid)" 2>/dev/null'; then
    echo "模型服务进程已退出，最后 100 行日志：" >&2
    tail -n 100 "${RESULT_ROOT}/logs/service.log" >&2 || true
    exit 1
  fi
  (( SECONDS < deadline )) || die "等待健康检查超时；请查看 ${RESULT_ROOT}/logs/service.log"
  echo "模型仍在加载……"
  sleep 10
done
curl --fail --silent --show-error "http://127.0.0.1:${SERVICE_PORT}/health"
echo
echo "===== 单请求"
curl --fail --silent --show-error \
  "http://127.0.0.1:${SERVICE_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"用三点解释 MoE 路由机制。\"}],\"temperature\":0,\"max_tokens\":128}"
echo
echo "检查通过。下一步：bash ${DEPLOY_DIR}/04_benchmark.sh"
