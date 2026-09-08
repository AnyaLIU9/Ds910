#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/data/models/Tensor}"
SOURCE_DIR="${SOURCE_DIR:-${MODEL_ROOT}/test}"
W8A8_MODEL="${W8A8_MODEL:-/data/models/Tensor-W8A8}"
VENV_DIR="${VENV_DIR:-/data/models/venv}"
RESULT_DIR="${RESULT_DIR:-/data/models/prometheus-results/preflight-w8a8}"
EXPECTED_RT_DEVICE="${EXPECTED_RT_DEVICE:-4}"
SLOTS_PER_LAYER="${SLOTS_PER_LAYER:-16}"

fail() { echo "[FAIL] $*" >&2; exit 1; }

[[ -d "$SOURCE_DIR" ]] || fail "缺少源码目录：$SOURCE_DIR"
[[ -d "$W8A8_MODEL" ]] || fail "缺少 W8A8 模型：$W8A8_MODEL"
[[ -x "$VENV_DIR/bin/python" ]] || fail "虚拟环境不存在：$VENV_DIR"

export ASCEND_RT_VISIBLE_DEVICES="$EXPECTED_RT_DEVICE"
export SOC_VERSION=ascend910b2
export PYTHONPATH="$SOURCE_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
source "$VENV_DIR/bin/activate"
cd "$SOURCE_DIR"
mkdir -p "$RESULT_DIR"

python tools/verify_ascend_environment.py >"$RESULT_DIR/environment.json"
python tools/validate_ascend_checkpoint.py \
  "$W8A8_MODEL" \
  --adapter qwen3_6_35b_a3b \
  --slots-per-layer "$SLOTS_PER_LAYER" \
  | tee "$RESULT_DIR/checkpoint.json"
python tools/ascend_probe.py \
  --output "$RESULT_DIR/probe.json" \
  --expect-soc ascend910b2 \
  --expect-devices 1 \
  --copy-mib "${ASCEND_PROBE_COPY_MIB:-16}" \
  --copy-iters "${ASCEND_PROBE_COPY_ITERS:-1}" \
  --require-op npu_grouped_matmul \
  --require-op npu_dynamic_quant \
  | tee "$RESULT_DIR/probe.log"
python tools/probe_ascend_moe.py | tee "$RESULT_DIR/moe-operator.json"

python - "$RESULT_DIR/checkpoint.json" "$RESULT_DIR/probe.json" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
probe = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

expected = {
    "quantization": "ascend_w8a8",
    "num_hidden_layers": 40,
    "num_experts": 192,
    "top_k": 8,
}
for key, value in expected.items():
    actual = checkpoint.get(key)
    assert actual == value, f"checkpoint {key}: expected {value!r}, got {actual!r}"

assert probe.get("npu_count") == 1, f"expected one NPU, got {probe.get('npu_count')}"
ops = probe.get("ops") or {}
for name in ("npu_grouped_matmul", "npu_dynamic_quant"):
    assert ops.get(name) is True, f"required op missing: {name}"
assert not probe.get("error"), probe.get("error")
print("W8A8 checkpoint、910B2 单卡环境和必需算子检查通过")
PY

echo "结果目录：$RESULT_DIR"
echo "说明：容器预检不调用 npu-smi；物理 NPU 状态只在宿主机检查。"
