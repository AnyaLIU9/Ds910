#!/usr/bin/env bash
set -euo pipefail

failures=0
warnings=0

pass() { printf '[PASS] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; warnings=$((warnings + 1)); }
fail() { printf '[FAIL] %s\n' "$*"; failures=$((failures + 1)); }

printf '[INFO] 本脚本应在准备运行 DeepSeek-V4-Flash 的容器内执行。\n'

if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "aarch64" ]]; then
  pass "容器为 Linux/aarch64"
else
  fail "容器平台为 $(uname -s)/$(uname -m)，需要 Linux/aarch64"
fi

for command_name in python3 pip3 git gcc g++ make cmake pkg-config; do
  if command -v "$command_name" >/dev/null 2>&1; then
    pass "$command_name: $(command -v "$command_name")"
  else
    fail "缺少构建命令 $command_name"
  fi
done

if command -v numactl >/dev/null 2>&1; then
  pass "numactl 可用"
else
  warn "缺少 numactl；编译不一定失败，但无法检查/调优 NUMA"
fi

if pkg-config --exists hwloc 2>/dev/null; then
  pass "hwloc 开发库可由 pkg-config 发现"
elif ldconfig -p 2>/dev/null | grep -q libhwloc; then
  warn "检测到 libhwloc 运行库，但可能缺少开发头文件/libhwloc-dev"
else
  fail "缺少 hwloc；安装运行库和开发头文件"
fi

cann_setenv=""
for candidate in \
  /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash \
  /usr/local/Ascend/ascend-toolkit/set_env.sh \
  /usr/local/Ascend/ascend-toolkit/set_env.bash; do
  if [[ -f "$candidate" ]]; then
    cann_setenv="$candidate"
    break
  fi
done

if [[ -n "$cann_setenv" ]]; then
  # shellcheck disable=SC1090
  if source "$cann_setenv"; then
    pass "已加载 CANN 环境: $cann_setenv"
  else
    fail "CANN环境脚本加载失败: $cann_setenv"
  fi
else
  fail "镜像内未找到 CANN setenv脚本"
fi

if [[ -n "${ASCEND_OPP_PATH:-}" && -d "${ASCEND_OPP_PATH}" ]]; then
  pass "ASCEND_OPP_PATH=${ASCEND_OPP_PATH}"
else
  fail "ASCEND_OPP_PATH 未设置或目录不存在；镜像可能缺少 CANN OPP算子包"
fi

if [[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]]; then
  pass "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
else
  warn "ASCEND_HOME_PATH 未设置；记录镜像实际 CANN安装路径"
fi

if command -v npu-smi >/dev/null 2>&1; then
  pass "容器内 npu-smi 可用"
  npu-smi info || fail "容器内 npu-smi info失败；检查设备和driver挂载"
else
  warn "容器内没有 npu-smi命令；至少要通过 torch_npu确认设备"
fi

python3 - <<'PY' || exit_code=$?
import sys

try:
    import torch
    import torch_npu
except Exception as exc:
    print(f"[PY-FAIL] import torch/torch_npu: {exc}")
    raise

print(f"[PY-INFO] python={sys.version.split()[0]}")
print(f"[PY-INFO] torch={torch.__version__}")
print(f"[PY-INFO] torch_npu={torch_npu.__version__}")
print(f"[PY-INFO] npu_available={torch.npu.is_available()}")
print(f"[PY-INFO] npu_count={torch.npu.device_count()}")

if not torch.npu.is_available() or torch.npu.device_count() < 1:
    raise RuntimeError("No visible NPU inside container")

torch.npu.set_device(0)
x = torch.ones((16, 16), dtype=torch.float16, device="npu")
y = x @ x
torch.npu.synchronize()
if float(y[0, 0].cpu()) != 16.0:
    raise RuntimeError("Unexpected NPU matmul result")
print("[PY-PASS] NPU tensor allocation and matmul succeeded")
PY

if [[ "${exit_code:-0}" -eq 0 ]]; then
  pass "torch/torch_npu及最小 NPU计算通过"
else
  fail "torch/torch_npu或最小 NPU计算失败"
fi

if python3 -c 'import kt_kernel_ext' >/dev/null 2>&1; then
  pass "kt_kernel_ext 已安装且可导入"
else
  warn "kt_kernel_ext 尚不可导入；首次构建前正常，构建完成后必须消除此警告"
fi

if [[ -n "${ASCEND_CUSTOM_OPP_PATH:-}" ]]; then
  printf '[INFO] ASCEND_CUSTOM_OPP_PATH=%s\n' "$ASCEND_CUSTOM_OPP_PATH"
else
  warn "未设置 ASCEND_CUSTOM_OPP_PATH；自定义算子可能尚未安装或由框架动态加载"
fi

printf '[RESULT] failures=%d warnings=%d\n' "$failures" "$warnings"
if (( failures > 0 )); then
  exit 1
fi
