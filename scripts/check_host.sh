#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

DATA_ROOT="${DATA_ROOT:-/data/dsv4}"
NPU_IDS="${NPU_IDS:-3,5,6}"
MIN_MEM_GIB="${MIN_MEM_GIB:-160}"
MIN_DISK_GIB="${MIN_DISK_GIB:-600}"
failures=0

pass() { printf '[PASS] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*"; failures=$((failures + 1)); }

if [[ "$(uname -s)" != "Linux" ]]; then
  fail "必须在目标 Linux 服务器执行，当前为 $(uname -s)"
fi

arch="$(uname -m)"
if [[ "$arch" == "aarch64" ]]; then
  pass "CPU 架构为 aarch64"
else
  fail "CPU 架构为 $arch，需要 aarch64"
fi

if grep -qm1 -w asimddp /proc/cpuinfo 2>/dev/null; then
  pass "检测到 ARM dot-product/SDOT (asimddp)"
else
  fail "未检测到 asimddp，MXFP4 CPU kernel 无法按目标路径运行"
fi

printf '[INFO] online CPUs: %s\n' "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
if command -v numactl >/dev/null 2>&1; then
  numa_nodes="$(numactl --hardware 2>/dev/null | awk '/available:/{print $2}')"
  printf '[INFO] NUMA nodes: %s\n' "${numa_nodes:-unknown}"
  numactl --hardware
else
  warn "未安装 numactl，无法检查 NUMA"
fi

if [[ -r /proc/meminfo ]]; then
  mem_gib="$(awk '/MemAvailable:/{printf "%d", $2/1024/1024}' /proc/meminfo)"
  if (( mem_gib >= MIN_MEM_GIB )); then
    pass "可用内存 ${mem_gib} GiB（单实例要求 >= ${MIN_MEM_GIB} GiB）"
  else
    fail "可用内存仅 ${mem_gib} GiB（单实例要求 >= ${MIN_MEM_GIB} GiB）"
  fi
fi

disk_probe="$DATA_ROOT"
while [[ ! -e "$disk_probe" && "$disk_probe" != "/" ]]; do
  disk_probe="$(dirname "$disk_probe")"
done
disk_gib="$(df -Pk "$disk_probe" | awk 'NR==2{printf "%d", $4/1024/1024}')"
if (( disk_gib >= MIN_DISK_GIB )); then
  pass "${disk_probe} 可用磁盘 ${disk_gib} GiB（转换期要求 >= ${MIN_DISK_GIB} GiB）"
else
  fail "${disk_probe} 可用磁盘仅 ${disk_gib} GiB（转换期要求 >= ${MIN_DISK_GIB} GiB）"
fi

if command -v npu-smi >/dev/null 2>&1; then
  pass "npu-smi 可用"
  npu-smi info || fail "npu-smi info 执行失败"
else
  fail "宿主机找不到 npu-smi；驱动/固件可能未正确安装"
fi

IFS=',' read -r -a npu_id_array <<< "$NPU_IDS"
for id in "${npu_id_array[@]}"; do
  if [[ -e "/dev/davinci${id}" ]]; then
    pass "/dev/davinci${id} 存在"
  else
    fail "/dev/davinci${id} 不存在"
  fi
done

for dev in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  if [[ -e "$dev" ]]; then
    pass "$dev 存在"
  else
    fail "$dev 不存在"
  fi
done

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  pass "Docker daemon 可用"
else
  fail "Docker daemon 不可用"
fi

if [[ -r /usr/local/Ascend/driver/version.info ]]; then
  printf '[INFO] Ascend driver version:\n'
  sed -n '1,20p' /usr/local/Ascend/driver/version.info
else
  warn "未找到 /usr/local/Ascend/driver/version.info，请人工记录驱动版本"
fi

if (( failures > 0 )); then
  printf '[RESULT] %d 项失败，暂不进入模型构建。\n' "$failures"
  exit 1
fi

printf '[RESULT] 基础验机通过。三实例前仍需确认可用 DDR >= 640 GiB。\n'
