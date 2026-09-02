#!/usr/bin/env bash
set -euo pipefail

# 唯一需要经常修改的参数：宿主机上的物理 NPU 编号。
NPU_ID="${NPU_ID:-5}"

DSV4_ROOT="${DSV4_ROOT:-/home/mem/dsv4}"
IMAGE="${IMAGE:-dsv4-offload-env:cann85-910b2}"
SERVICE_PORT="${SERVICE_PORT:-8020}"
SHM_SIZE="${SHM_SIZE:-64g}"
CONTAINER_NAME="${CONTAINER_NAME:-dsv4-npu${NPU_ID}}"
RELEASE_DIR="${DSV4_ROOT}/code/cann-recipes-infer/integration/sglang/dsv4-flash-single-npu-moe-offload"
LAUNCHER="${RELEASE_DIR}/scripts/launch_dsv4_singleCard_cann8.5.0_910b.sh"

if [[ ! "${NPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "错误：NPU_ID 必须是非负整数，当前值为 ${NPU_ID}" >&2
  exit 2
fi

for path in \
  "/dev/davinci${NPU_ID}" \
  /dev/davinci_manager \
  /dev/devmm_svm \
  /dev/hisi_hdc \
  "${LAUNCHER}" \
  "${DSV4_ROOT}/models"; do
  if [[ ! -e "${path}" ]]; then
    echo "错误：缺少 ${path}" >&2
    exit 1
  fi
done

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "错误：容器 ${CONTAINER_NAME} 已存在。请先停止/删除它，或设置其他 CONTAINER_NAME。" >&2
  exit 1
fi

echo "物理 NPU：${NPU_ID}（/dev/davinci${NPU_ID}）"
echo "容器名称：${CONTAINER_NAME}"
echo "服务端口：${SERVICE_PORT}"
echo "进入容器后，单卡通常使用逻辑编号 NPU_DEVICE_ID=0。"

exec env \
  WORKSPACE="${DSV4_ROOT}/code" \
  MODEL_DIR="${DSV4_ROOT}/models" \
  IMAGE="${IMAGE}" \
  NAME="${CONTAINER_NAME}" \
  SERVICE_PORT="${SERVICE_PORT}" \
  SHM_SIZE="${SHM_SIZE}" \
  NPU_VISIBLE_DEVICES="${NPU_ID}" \
  bash "${LAUNCHER}"
