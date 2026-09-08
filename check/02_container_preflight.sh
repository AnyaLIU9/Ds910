#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/data/models/Tensor}"
SOURCE_DIR="${SOURCE_DIR:-${MODEL_ROOT}/test}"
VENV_DIR="${VENV_DIR:-/data/models/venv}"
EXPECTED_SOC="${EXPECTED_SOC:-ascend910b2}"
EXPECTED_RT_DEVICE="${EXPECTED_RT_DEVICE:-4}"
MAPPING_SLEEP_SECONDS="${MAPPING_SLEEP_SECONDS:-15}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple}"
PYPI_TRUSTED_HOST="${PYPI_TRUSTED_HOST:-mirrors.huaweicloud.com}"

fail() { echo "[FAIL] $*" >&2; exit 1; }
pass() { echo "[PASS] $*"; }

[[ -d "$SOURCE_DIR" ]] || fail "缺少源码目录：$SOURCE_DIR"
[[ -x "$VENV_DIR/bin/python" ]] || fail "虚拟环境不存在：$VENV_DIR"

export ASCEND_RT_VISIBLE_DEVICES="$EXPECTED_RT_DEVICE"
export SOC_VERSION="$EXPECTED_SOC"
export PIP_INDEX_URL="$PYPI_INDEX_URL"
export PIP_TRUSTED_HOST="$PYPI_TRUSTED_HOST"
unset PIP_EXTRA_INDEX_URL

source "$VENV_DIR/bin/activate"
cd "$SOURCE_DIR"

python - <<'PY'
import os
import torch
import torch_npu
import prometheus

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("prometheus:", prometheus.__file__)
print("ASCEND_RT_VISIBLE_DEVICES:", os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
assert torch.npu.is_available(), "torch.npu.is_available() is false"
assert torch.npu.device_count() == 1, f"expected one visible NPU, got {torch.npu.device_count()}"
torch.npu.set_device(0)
name = torch.npu.get_device_name(0)
print("process device: npu:0")
print("device name:", name)
assert "910B" in name.upper().replace(" ", ""), f"unexpected device: {name}"

required = ("npu_grouped_matmul", "npu_dynamic_quant")
missing = [name for name in required if not hasattr(torch_npu, name)]
assert not missing, f"missing required torch_npu ops: {missing}"
print("required TorchNPU ops: PASS")
PY
pass "Torch、TorchNPU、Prometheus、单卡和 W8A8 必需算子检查通过"

configured_index="$(python -m pip config get global.index-url 2>/dev/null || true)"
[[ "$configured_index" == "$PYPI_INDEX_URL" ]] \
  || fail "venv pip index-url 与预期不一致：期望 $PYPI_INDEX_URL，实际 ${configured_index:-未配置}"
configured_extra="$(python -m pip config get global.extra-index-url 2>/dev/null || true)"
[[ -z "$configured_extra" ]] || fail "检测到禁止的 extra-index-url：$configured_extra"
pass "pip 仅配置指定的 PyPI 镜像：$PYPI_INDEX_URL"

MAPPING_SLEEP_SECONDS="$MAPPING_SLEEP_SECONDS" python - <<'PY'
import os
import time
import torch
import torch_npu

seconds = int(os.environ["MAPPING_SLEEP_SECONDS"])
x = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="npu:0")
print(f"已在进程内 npu:0 分配 128 MiB，将保持 {seconds} 秒。")
print("现在必须在宿主机运行 npu-smi info，确认新增显存位于物理 NPU 5。")
time.sleep(seconds)
del x
torch.npu.synchronize()
PY

echo
echo "容器自动检查通过；物理卡映射仍以宿主机 npu-smi 的人工观察为最终依据。"
