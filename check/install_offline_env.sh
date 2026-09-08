#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/data/models/Tensor}"
SOURCE_DIR="${SOURCE_DIR:-${MODEL_ROOT}/test}"
CHECK_DIR="${CHECK_DIR:-${MODEL_ROOT}/check}"
VENV_DIR="${VENV_DIR:-/data/models/venv}"
RESULT_DIR="${RESULT_DIR:-/data/models/prometheus-results}"

PROMPT_WHEEL="${CHECK_DIR}/prompt_toolkit-3.0.53-py3-none-any.whl"
WCWIDTH_WHEEL="${CHECK_DIR}/wcwidth-0.2.13-py2.py3-none-any.whl"
PROMPT_SHA256="01c0891d7f9237d5e339f7d3e42cdae80b7534abb1c7c0e3352efba6231492f2"
WCWIDTH_SHA256="3da69048e4540d84af32131829ff948f1e022c1c6bdb8d6102117aac784f6859"

fail() { echo "[FAIL] $*" >&2; exit 1; }
pass() { echo "[PASS] $*"; }

[[ -d "$SOURCE_DIR" ]] || fail "缺少源码目录：$SOURCE_DIR"
[[ -f "$SOURCE_DIR/requirements-ascend.txt" ]] \
  || fail "缺少文件：$SOURCE_DIR/requirements-ascend.txt"
[[ -f "$PROMPT_WHEEL" ]] || fail "缺少离线包：$PROMPT_WHEEL"
[[ -f "$WCWIDTH_WHEEL" ]] || fail "缺少离线包：$WCWIDTH_WHEEL"

printf '%s  %s\n' "$PROMPT_SHA256" "$PROMPT_WHEEL" | sha256sum -c -
printf '%s  %s\n' "$WCWIDTH_SHA256" "$WCWIDTH_WHEEL" | sha256sum -c -
pass "两个离线 wheel 的 SHA256 正确"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python -m venv --system-site-packages "$VENV_DIR"
  pass "已创建持久化环境：$VENV_DIR"
else
  pass "复用持久化环境：$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python - <<'PY'
import sys
assert sys.version_info[:2] == (3, 11), f"要求容器 Python 3.11，实际为 {sys.version}"
print("Python:", sys.version)
PY

# 安装 check 目录中的所有 wheel，方便以后按同样方式手工补包。
mapfile -t wheels < <(find "$CHECK_DIR" -maxdepth 1 -type f -name '*.whl' -print | sort)
(( ${#wheels[@]} > 0 )) || fail "${CHECK_DIR} 中没有 wheel"
python -m pip install --no-index --no-deps "${wheels[@]}"

mkdir -p "$RESULT_DIR"
missing_file="$RESULT_DIR/requirements-missing.txt"
SOURCE_DIR="$SOURCE_DIR" MISSING_FILE="$missing_file" python - <<'PY'
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from packaging.requirements import Requirement

source = Path(os.environ["SOURCE_DIR"])
missing_file = Path(os.environ["MISSING_FILE"])
missing = []
for raw in (source / "requirements-ascend.txt").read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    requirement = Requirement(line)
    try:
        installed = version(requirement.name)
    except PackageNotFoundError:
        missing.append(line)
        print(f"MISSING   {line}")
        continue
    if installed in requirement.specifier:
        print(f"OK        {requirement.name}=={installed}")
    else:
        missing.append(line)
        print(f"MISMATCH  {requirement.name}=={installed}; require {requirement.specifier}")

missing_file.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
print("missing/mismatch count:", len(missing))
PY

if [[ -s "$missing_file" ]]; then
  echo
  echo "下面这些直接依赖仍缺失或版本不符：" >&2
  cat "$missing_file" >&2
  fail "请手工下载对应的 Python 3.11/aarch64 wheel 到 $CHECK_DIR，然后重新运行本脚本"
fi

# 此时只检查基础镜像和离线 wheel 的依赖闭包；不要先安装带 CUDA 默认元数据的源码包。
python -m pip check

PYTHONPATH="$SOURCE_DIR/python" python - <<'PY'
import prompt_toolkit
import wcwidth
import prometheus

print("prompt_toolkit:", prompt_toolkit.__version__)
print("wcwidth:", wcwidth.__version__)
print("prometheus source:", prometheus.__file__)
PY

pass "持久化环境、Ascend 直接依赖和源码导入检查全部通过"
echo "后续进入新容器后执行：source $VENV_DIR/bin/activate"
