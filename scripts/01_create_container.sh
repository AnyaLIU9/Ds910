#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/qwen36.conf"

[[ "${NPU_ID}" == "3" ]] || die "安全限制：只允许物理 NPU 3"
[[ -d "${MODEL_PATH}" ]] || die "模型目录不存在：${MODEL_PATH}"
[[ -f "${MODEL_PATH}/config.json" ]] || die "模型目录缺少 config.json"
[[ -e "/dev/davinci${NPU_ID}" ]] || die "缺少 /dev/davinci${NPU_ID}"
docker image inspect "${IMAGE}" >/dev/null 2>&1 || die "本机不存在镜像：${IMAGE}"
docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1 && \
  die "容器 ${CONTAINER_NAME} 已存在；不会覆盖或删除"

echo "请人工确认下方进程表中 NPU 3 空闲："
npu-smi info
[[ "${NPU3_CONFIRMED_FREE:-}" == "YES" ]] || die \
  "确认卡 3 空闲后执行：export NPU3_CONFIRMED_FREE=YES"

mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/metrics" "${RESULT_ROOT}/bench"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --shm-size 16g \
  --device "/dev/davinci${NPU_ID}" \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v "${MODEL_PATH}":/model:ro \
  -v "${RESULT_ROOT}":/results \
  -e "ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}" \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  --entrypoint /bin/bash \
  "${IMAGE}" -lc 'exec sleep infinity'

docker inspect "${CONTAINER_NAME}" \
  --format 'created name={{.Name}} image={{.Config.Image}} running={{.State.Running}}'
echo "下一步：bash ${DEPLOY_DIR}/02_start_service.sh"
