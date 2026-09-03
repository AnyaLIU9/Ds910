#!/usr/bin/env bash
set -euo pipefail

# 唯一需要经常修改的参数：宿主机上的物理 NPU 编号。
NPU_ID="${NPU_ID:-5}"

DSV4_ROOT="${DSV4_ROOT:-/data/models/dsv4}"
IMAGE="${IMAGE:-dsv4-offload-env:cann85-910b2}"
SERVICE_PORT="${SERVICE_PORT:-9108}"
SHM_SIZE="${SHM_SIZE:-64g}"
CONTAINER_NAME="${CONTAINER_NAME:-dsv4-npu${NPU_ID}}"
RELEASE_DIR="${DSV4_ROOT}/code/cann-recipes-infer/integration/sglang/dsv4-flash-single-npu-moe-offload"
LAUNCHER="${RELEASE_DIR}/scripts/launch_dsv4_singleCard_cann8.5.0_910b.sh"
RUNTIME_ENV="${DSV4_ROOT}/code/dsv4_runtime.env"

if [[ ! "${NPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "错误：NPU_ID 必须是非负整数，当前值为 ${NPU_ID}" >&2
  exit 2
fi

if [[ ! "${SERVICE_PORT}" =~ ^[0-9]+$ ]] || (( SERVICE_PORT < 1 || SERVICE_PORT > 65535 )); then
  echo "错误：SERVICE_PORT 必须是 1-65535，当前值为 ${SERVICE_PORT}" >&2
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

# 该文件位于宿主机挂载目录，进入容器后 source 一次即可得到同一个物理卡号。
printf 'export NPU_DEVICE_ID=%q\nexport ASCEND_RT_VISIBLE_DEVICES=%q\nexport PORT=%q\n' \
  "${NPU_ID}" "${NPU_ID}" "${SERVICE_PORT}" >"${RUNTIME_ENV}"

echo "物理 NPU：${NPU_ID}（/dev/davinci${NPU_ID}）"
echo "容器名称：${CONTAINER_NAME}"
echo "服务端口：${SERVICE_PORT}"
echo "进入容器后执行：source /workspace/code/dsv4_runtime.env"
echo "NPU_DEVICE_ID 使用物理卡号 ${NPU_ID}；运行时会把它映射成进程内逻辑卡 0。"

exec env \
  WORKSPACE="${DSV4_ROOT}/code" \
  MODEL_DIR="${DSV4_ROOT}/models" \
  IMAGE="${IMAGE}" \
  NAME="${CONTAINER_NAME}" \
  SERVICE_PORT="${SERVICE_PORT}" \
  SHM_SIZE="${SHM_SIZE}" \
  NPU_VISIBLE_DEVICES="${NPU_ID}" \
  bash "${LAUNCHER}"
