#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-/data/models/Tensor-0.1-Flash-35B-A3B}"
RESULT_ROOT="${MODEL_PATH}/results/qwen36-vllm-npu3"
CONTAINER_NAME=qwen36-flash-npu3

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  label="$(docker inspect -f '{{index .Config.Labels "com.dsv4.qwen36.npu3"}}' "${CONTAINER_NAME}")"
  [[ "${label}" == true ]] || { echo "容器安全标签不匹配，拒绝删除" >&2; exit 1; }
  docker stop --timeout 30 "${CONTAINER_NAME}"
  docker rm "${CONTAINER_NAME}"
  echo "已停止并删除本次容器：${CONTAINER_NAME}"
else
  echo "本次容器不存在，跳过。"
fi

if [[ "${DELETE_RESULTS:-NO}" == YES ]]; then
  [[ "${RESULT_ROOT}" == "/data/models/Tensor-0.1-Flash-35B-A3B/results/qwen36-vllm-npu3" ]] || {
    echo "结果路径不符合安全白名单，拒绝删除：${RESULT_ROOT}" >&2; exit 1;
  }
  if [[ -d "${RESULT_ROOT}" ]]; then
    find "${RESULT_ROOT}" -depth -mindepth 1 -delete
    rmdir "${RESULT_ROOT}"
    echo "已删除本次日志和压测结果：${RESULT_ROOT}（不可恢复）"
  fi
else
  echo "结果已保留：${RESULT_ROOT}"
  echo "如需同时删除结果：DELETE_RESULTS=YES bash ${MODEL_PATH}/deployment/04_cleanup.sh"
fi
