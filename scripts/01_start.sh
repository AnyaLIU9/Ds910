#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-/data/models/Tensor-0.1-Flash-35B-A3B}"
RESULT_ROOT="${MODEL_PATH}/results/qwen36-vllm-npu3"
IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.19.1rc1-openeuler}"
CONTAINER_NAME=qwen36-flash-npu3
QUANTIZATION="${QUANTIZATION:-}"

die() { echo "错误：$*" >&2; exit 1; }
[[ -f "${MODEL_PATH}/config.json" ]] || die "模型路径错误：${MODEL_PATH}"
if grep -Eiq 'modelopt_fp4|nvfp4|"quant_method"[[:space:]]*:[[:space:]]*"modelopt' \
  "${MODEL_PATH}/config.json" "${MODEL_PATH}/quantization_config.json" 2>/dev/null; then
  die "检测到 NVIDIA ModelOpt/NVFP4 权重；vLLM-Ascend/910B2 不支持。请换同模型的 Ascend W8A8 权重"
fi
if [[ -z "${QUANTIZATION}" ]] && grep -Eiq 'w8a8|"quant_method"[[:space:]]*:[[:space:]]*"ascend' \
  "${MODEL_PATH}/config.json" "${MODEL_PATH}/quantization_config.json" 2>/dev/null; then
  QUANTIZATION=ascend
fi
[[ -e /dev/davinci3 ]] || die "缺少 /dev/davinci3"
docker image inspect "${IMAGE}" >/dev/null 2>&1 || die "镜像不存在：${IMAGE}"
docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1 && die "同名容器已存在"
ss -ltn | awk '{print $4}' | grep -Eq '(^|:)9108$' && die "端口 9108 已占用"

npu-smi info
[[ "${NPU3_CONFIRMED_FREE:-}" == YES ]] || die "确认 NPU 3 空闲后设置 NPU3_CONFIRMED_FREE=YES"
mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/bench"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --label com.dsv4.qwen36.npu3=true \
  --network host --shm-size 16g \
  --device /dev/davinci3 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v "${MODEL_PATH}":/model:ro \
  -v "${RESULT_ROOT}":/results \
  -e ASCEND_RT_VISIBLE_DEVICES=3 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e "QUANTIZATION=${QUANTIZATION}" \
  "${IMAGE}" bash -lc 'quant_args=(); \
    [[ -z "${QUANTIZATION}" ]] || quant_args=(--quantization "${QUANTIZATION}"); \
    exec vllm serve /model \
    --served-model-name Tensor-0.1-Flash-35B-A3B \
    --host 0.0.0.0 --port 9108 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager --trust-remote-code \
    "${quant_args[@]}"' \
  | tee "${RESULT_ROOT}/container-id.txt"

echo "已启动。日志：docker logs -f ${CONTAINER_NAME}"
echo "下一步：bash ${MODEL_PATH}/deployment/02_check.sh"
