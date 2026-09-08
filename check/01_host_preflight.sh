#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/data/models/Tensor}"
SOURCE_DIR="${SOURCE_DIR:-${MODEL_ROOT}/test}"
W8A8_MODEL="${W8A8_MODEL:-/data/models/Tensor-W8A8}"
IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.20.2rc1-openeuler}"
NPU_PHYSICAL_ID="${NPU_PHYSICAL_ID:-5}"
SERVICE_PORT="${SERVICE_PORT:-9108}"
MIN_FREE_GIB="${MIN_FREE_GIB:-80}"

fail() { echo "[FAIL] $*" >&2; exit 1; }
pass() { echo "[PASS] $*"; }
need_file() { [[ -f "$1" ]] || fail "缺少文件：$1"; }
need_dir() { [[ -d "$1" ]] || fail "缺少目录：$1"; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || fail "缺少命令：$1"; }

[[ "$NPU_PHYSICAL_ID" =~ ^[0-9]+$ ]] || fail "NPU_PHYSICAL_ID 必须是非负整数"
[[ "$SERVICE_PORT" =~ ^[0-9]+$ ]] || fail "SERVICE_PORT 必须是整数"

need_cmd docker
need_cmd npu-smi
need_cmd find
need_cmd df
need_dir "$MODEL_ROOT"
need_dir "$SOURCE_DIR"

for path in \
  "$SOURCE_DIR/README_ASCEND.md" \
  "$SOURCE_DIR/requirements-ascend.txt" \
  "$SOURCE_DIR/scripts/quantize_qwen3_6_35b_a3b_w8a8.sh" \
  "$SOURCE_DIR/scripts/run_ascend_910b2_offload.sh" \
  "$SOURCE_DIR/scripts/preflight_ascend.sh" \
  "$SOURCE_DIR/tools/ascend_probe.py" \
  "$SOURCE_DIR/tools/validate_ascend_checkpoint.py" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/model.safetensors.index.json"; do
  need_file "$path"
done
need_dir "$SOURCE_DIR/python/prometheus/ascend"

arch="$(uname -m)"
[[ "$arch" == "aarch64" ]] || fail "宿主机架构是 $arch，要求 aarch64"
pass "宿主机架构：$arch"

image_arch="$(docker image inspect "$IMAGE" --format '{{.Architecture}}' 2>/dev/null)" \
  || fail "本机不存在镜像：$IMAGE"
[[ "$image_arch" == "arm64" ]] || fail "镜像架构是 $image_arch，要求 arm64"
pass "镜像存在且架构为 arm64：$IMAGE"

for device in \
  "/dev/davinci${NPU_PHYSICAL_ID}" \
  /dev/davinci_manager \
  /dev/devmm_svm \
  /dev/hisi_hdc; do
  [[ -e "$device" ]] || fail "缺少设备节点：$device"
done
pass "物理 NPU ${NPU_PHYSICAL_ID} 及管理设备节点存在"

shard_count="$(find "$MODEL_ROOT" -maxdepth 1 -type f -name 'model-*-of-*.safetensors' | wc -l | tr -d ' ')"
[[ "$shard_count" == "26" ]] || fail "模型分片数量为 $shard_count，要求 26"
pass "模型目录和 26 个权重分片完整"

free_kib="$(df -Pk /data/models | awk 'NR==2 {print $4}')"
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
(( free_kib >= required_kib )) \
  || fail "/data/models 可用空间不足 ${MIN_FREE_GIB} GiB"
pass "/data/models 可用空间不少于 ${MIN_FREE_GIB} GiB"

if command -v ss >/dev/null 2>&1 && ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${SERVICE_PORT}$"; then
  fail "端口 ${SERVICE_PORT} 已被监听"
fi
pass "端口 ${SERVICE_PORT} 未发现监听进程"

if [[ -e "$W8A8_MODEL" ]] && find "$W8A8_MODEL" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "[WARN] W8A8 输出目录已存在且非空：$W8A8_MODEL"
else
  pass "W8A8 输出目录不存在或为空"
fi

echo
echo "请人工核对下面 npu-smi 输出：物理 NPU ${NPU_PHYSICAL_ID} 必须是 910B2，且没有未知任务占用大量 HBM。"
npu-smi info

echo
echo "宿主机前置检查通过。"
