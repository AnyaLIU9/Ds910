#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/data/models/Tensor}"
SOURCE_DIR="${SOURCE_DIR:-${MODEL_ROOT}/test}"
VENV_DIR="${VENV_DIR:-/data/models/venv}"
EXPECTED_SOC="${EXPECTED_SOC:-ascend910b2}"
EXPECTED_RT_DEVICE="${EXPECTED_RT_DEVICE:-5}"
MAPPING_SLEEP_SECONDS="${MAPPING_SLEEP_SECONDS:-15}"

fail() { echo "[FAIL] $*" >&2; exit 1; }
pass() { echo "[PASS] $*"; }
trap 'echo "[FAIL] 容器 NPU 检查失败；请查看上方第一个 Python Traceback/CANN 错误，末尾 ERR99999 UNKNOWN application exception 只是汇总。" >&2' ERR

[[ -d "$SOURCE_DIR" ]] || fail "缺少源码目录：$SOURCE_DIR"
[[ -x "$VENV_DIR/bin/python" ]] || fail "虚拟环境不存在：$VENV_DIR"

export ASCEND_RT_VISIBLE_DEVICES="$EXPECTED_RT_DEVICE"
export SOC_VERSION="$EXPECTED_SOC"
export PYTHONPATH="$SOURCE_DIR/python${PYTHONPATH:+:$PYTHONPATH}"

source "$VENV_DIR/bin/activate"
cd "$SOURCE_DIR"

echo "[CHECK] 容器设备节点（本脚本不会调用 npu-smi）："
ls -l /dev/davinci5 /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc

python - <<'PY'
import os
print("[CHECK] 导入 torch/torch_npu")
import torch
import torch_npu
print("[CHECK] 从源码导入 prometheus")
import prometheus

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("prometheus:", prometheus.__file__)
print("ASCEND_RT_VISIBLE_DEVICES:", os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
assert torch.npu.is_available(), "torch.npu.is_available() is false"
assert torch.npu.device_count() == 1, f"expected one visible NPU, got {torch.npu.device_count()}"
print("[CHECK] 初始化进程设备 npu:0")
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

MAPPING_SLEEP_SECONDS="$MAPPING_SLEEP_SECONDS" python - <<'PY'
import os
import time
import torch
import torch_npu

seconds = int(os.environ["MAPPING_SLEEP_SECONDS"])
x = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="npu:0")
print(f"已在进程内 npu:0 分配 128 MiB，将保持 {seconds} 秒。")
print("如需核对物理卡，请在另一个宿主机终端观察物理 NPU 5；不要在本容器运行 npu-smi。")
time.sleep(seconds)
del x
torch.npu.synchronize()
PY

echo
echo "容器自动检查通过；物理卡映射仍以宿主机 npu-smi 的人工观察为最终依据。"
