# DeepSeek-V4-Flash 单卡实验：服务器操作

本文从这个状态开始：**代码、权重和 deb 已经解压并上传；Docker 镜像 tar 已放进 `packages/`，但还没有执行 `docker load`**。除这个镜像 tar 外，实验目录不保存其他 ZIP、tar 或 tar.gz。

命令标注“宿主机”的就在服务器 shell 执行；标注“容器内”的要进入容器后执行。

## 先理解 Docker 镜像从哪来

```text
联网电脑下载的 ARM64 tar
        │ docker load
        ▼
lmsysorg/sglang:deepseek-v4-npu-910b       基础镜像：已有 CANN 8.5、torch_npu、算子
        │ docker build，加入 hwloc deb
        ▼
dsv4-offload-env:cann85-910b2               派生镜像：本实验真正使用
        │ docker run
        ▼
dsv4-npu5                                   一次运行出来的容器
```

`/home/mem/dsv4/image/` **不是存 Docker 镜像的地方**。它只是放 `Dockerfile` 和 deb 的构建目录。真正的镜像存在 Docker 的 `DockerRootDir`，通常是 `/var/lib/docker`。

## 0. 实验开始时目录必须是这样

```text
/home/mem/dsv4/
├── packages/
│   └── deepseek-v4-npu-910b-arm64.tar          # 唯一保留的压缩包
├── code/
│   ├── cann-recipes-infer/
│   │   └── integration/sglang/dsv4-flash-single-npu-moe-offload/
│   │       ├── apply_all.sh
│   │       ├── main_repo/
│   │       ├── sglang/
│   │       ├── llama_cpp/
│   │       └── scripts/
│   ├── ktransformers-AK/
│   │   ├── kt-kernel/
│   │   └── third_party/
│   │       ├── pybind11/
│   │       ├── custom_flashinfer/
│   │       ├── sglang/
│   │       └── llama.cpp/
│   ├── scripts/
│   │   ├── start_single_npu_container.sh
│   │   └── bench_concurrency.py
│   └── logs/
├── models/
│   ├── DeepSeek-V4-Flash-W8A8/
│   ├── DeepSeek-V4-Flash/
│   └── cache/                                  # 第一次转换后生成 43 个 GGUF
├── image/
│   ├── Dockerfile
│   └── debs/                                   # 从 Mac 上传来的 17 个 ARM64 deb
└── results/
```

`image/debs/` 中的 deb 只会安装进后面构建的 Docker 派生镜像。**不要在测试服务器宿主机运行 `dpkg -i`。**

`image/Dockerfile` 和 `code/scripts/` 都在 Ds910 仓库中。不要把四个 third-party 仓库再多套一层目录；例如必须直接存在：

```text
ktransformers-AK/third_party/sglang/python/
ktransformers-AK/third_party/llama.cpp/convert-hf-to-gguf.py
ktransformers-AK/third_party/pybind11/CMakeLists.txt
```

## 1. 宿主机检查

```bash
uname -m
lscpu | grep -E 'CPU\(s\)|NUMA node\(s\)|Model name'
npu-smi info
docker --version
docker info --format 'DockerRootDir={{.DockerRootDir}}'
docker system df
```

应满足：

- `uname -m` 是 `aarch64`；
- 能在 `npu-smi info` 看到空闲的 910B2，例如物理卡 3 或 5；
- Docker 能正常工作；
- DockerRootDir 所在磁盘有足够空间。即使 `/home/mem` 很大，DockerRootDir 空间不足仍会构建失败，此时先找管理员迁移 Docker 数据目录。

检查卡 5 的设备节点；换卡时改数字：

```bash
NPU_ID=5
ls -l "/dev/davinci${NPU_ID}" \
  /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
```

## 2. 导入 Docker 基础镜像

这是第一次部署必须执行的启动步骤。宿主机先校验 tar：

```bash
cd /home/mem/dsv4/packages
echo 'cfdb04294636003f6638425df999b4c13b89079356ca6d8f8618abec07c0bbdf  deepseek-v4-npu-910b-arm64.tar' \
  | sha256sum -c -
```

必须显示 `OK`。然后导入：

```bash
docker load -i /home/mem/dsv4/packages/deepseek-v4-npu-910b-arm64.tar
```

导入完成后检查：

```bash
docker image inspect lmsysorg/sglang:deepseek-v4-npu-910b \
  --format 'arch={{.Architecture}} id={{.Id}}'
```

必须看到 `arch=arm64`。`packages/` 中只保留这个镜像 tar，不要再放代码 ZIP、模型压缩包或其他传输包。

## 3. 检查上传目录

宿主机执行：

```bash
export DSV4_ROOT=/home/mem/dsv4
export REPO="$DSV4_ROOT/code/ktransformers-AK"
export RELEASE_DIR="$DSV4_ROOT/code/cann-recipes-infer/integration/sglang/dsv4-flash-single-npu-moe-offload"

test -f "$RELEASE_DIR/apply_all.sh"
test -f "$REPO/kt-kernel/setup.py"
test -f "$REPO/third_party/pybind11/CMakeLists.txt"
test -n "$(find "$REPO/third_party/custom_flashinfer" -mindepth 1 -maxdepth 1 -print -quit)"
test -d "$REPO/third_party/sglang/python/sglang"
test -f "$REPO/third_party/llama.cpp/convert-hf-to-gguf.py"
find "$DSV4_ROOT/image/debs" -maxdepth 1 -name '*.deb' | wc -l
```

最后一条应显示 `17`。任意 `test` 失败就先修目录，不要继续。

检查两份权重有没有缺分片：

```bash
python3 - <<'PY'
import json
from pathlib import Path

for root in map(Path, [
    "/home/mem/dsv4/models/DeepSeek-V4-Flash-W8A8",
    "/home/mem/dsv4/models/DeepSeek-V4-Flash",
]):
    index = root / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    files = sorted(set(data["weight_map"].values()))
    missing = [name for name in files if not (root / name).is_file()]
    print(root.name, "分片", len(files), "缺失", len(missing))
    if missing:
        raise SystemExit(missing[:10])
PY
```

两行都必须显示“缺失 0”。

## 4. 构建本实验使用的派生镜像

宿主机执行：

```bash
cd /home/mem/dsv4/image
docker build -t dsv4-offload-env:cann85-910b2 .
```

这一步不会联网：`FROM` 使用已经导入的基础镜像，`COPY` 把 `image/debs/` 放进镜像，然后 Dockerfile 在镜像内部执行 `dpkg -i`。它不会把 hwloc 安装到测试服务器宿主机。

验收：

```bash
docker run --rm --entrypoint /bin/bash \
  dsv4-offload-env:cann85-910b2 \
  -lc 'test -f /usr/include/hwloc.h && ldconfig -p | grep libhwloc && pkg-config --modversion hwloc'
```

最后应输出 `2.7.0`。到这里，`dsv4-offload-env:cann85-910b2` 才真正存在；它不是提前下载的另一个文件。

## 5. 选择一张卡并进入容器

宿主机执行。使用物理卡 5：

```bash
export NPU_ID=5
bash /home/mem/dsv4/code/scripts/start_single_npu_container.sh
```

改用物理卡 3 时只改为：

```bash
export NPU_ID=3
bash /home/mem/dsv4/code/scripts/start_single_npu_container.sh
```

命令成功后会直接进入容器。脚本同时生成 `/workspace/code/dsv4_runtime.env`，后续不需要再次手写卡号。

## 6. 容器内检查

```bash
source /workspace/code/dsv4_runtime.env
echo "physical NPU=$NPU_DEVICE_ID, visible=$ASCEND_RT_VISIBLE_DEVICES, port=$PORT"
npu-smi info
```

再检查 Python 和 NPU：

```bash
python3 - <<'PY'
import torch
import torch_npu
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("available", torch.npu.is_available())
print("visible device count", torch.npu.device_count())
PY
```

`available` 必须为 `True`。`NPU_DEVICE_ID` 是宿主机物理卡号；`ASCEND_RT_VISIBLE_DEVICES` 会让进程内部把这张卡当作逻辑卡 0。

## 7. 应用补丁（只执行一次）

容器内执行：

```bash
export REPO=/workspace/code/ktransformers-AK
export RELEASE_DIR=/workspace/code/cann-recipes-infer/integration/sglang/dsv4-flash-single-npu-moe-offload

test ! -e "$REPO/.dsv4_patch_applied"
bash "$RELEASE_DIR/apply_all.sh" "$REPO"
mkdir -p "$REPO/tools"
cp -a "$RELEASE_DIR/scripts/tools/." "$REPO/tools/"
touch "$REPO/.dsv4_patch_applied"
```

如果第一条 `test` 失败，说明已经打过补丁，不要再执行 `apply_all.sh`。

## 8. 编译 kt-kernel

容器内执行：

```bash
export PYTHON_BIN=$(command -v python3.11 || command -v python3)
cd "$REPO/kt-kernel"

CPUINFER_USE_ASCEND_NPU=1 \
CPUINFER_ARM_SVE=OFF \
CPUINFER_ARM_BF16=OFF \
CPUINFER_ARM_I8MM=OFF \
"$PYTHON_BIN" setup.py build_ext --inplace \
  2>&1 | tee /workspace/code/logs/kt-kernel-build.log

ln -sfn python "$REPO/kt-kernel/kt_kernel"
export PYTHONPATH="$REPO/third_party/sglang/python:$REPO/kt-kernel${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -c 'import kt_kernel; print("kt_kernel OK")'
```

最后必须输出 `kt_kernel OK`。

## 9. 把 CPU 权重转换成 43 个 GGUF

这一步只做一次，约生成 138 GiB。容器内执行：

```bash
export W8A8_DIR=/workspace/models/DeepSeek-V4-Flash-W8A8
export MXFP4_SRC=/workspace/models/DeepSeek-V4-Flash
export GGUF_CACHE=/workspace/models/cache
mkdir -p "$GGUF_CACHE" /workspace/code/logs
cd "$REPO"

"$PYTHON_BIN" tools/batch_convert_mxfp4_layers_mp.py \
  --input "$MXFP4_SRC" \
  --output-dir "$GGUF_CACHE" \
  --layer-start 0 --layer-end 42 \
  --jobs 16 --verify-sample 3 \
  2>&1 | tee /workspace/code/logs/mxfp4-convert.log
```

转换完成后校验：

```bash
find "$GGUF_CACHE" -maxdepth 1 -name 'dsv4_layer*_mxfp4.gguf' | wc -l

"$PYTHON_BIN" tools/verify_mxfp4_gguf_set.py \
  --dir "$GGUF_CACHE" \
  --sha256-manifest tools/mxfp4_gguf_sha256.txt \
  2>&1 | tee /workspace/code/logs/gguf-verify.log
```

文件数必须为 `43`，全集校验必须通过。校验前不要删除原始 `DeepSeek-V4-Flash` 权重。

## 10. 预检并启动服务

容器内执行：

```bash
source /workspace/code/dsv4_runtime.env
export REPO=/workspace/code/ktransformers-AK
export PYTHON_BIN=$(command -v python3.11 || command -v python3)
export W8A8_DIR=/workspace/models/DeepSeek-V4-Flash-W8A8
export GGUF_CACHE=/workspace/models/cache
export PYTHONPATH="$REPO/third_party/sglang/python:$REPO/kt-kernel${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO"
GGUF_DIR="$GGUF_CACHE" GGUF_SUFFIX=_mxfp4 bash tools/e2e_preflight.sh
```

看到 `PASS` 后启动：

```bash
NPU_DEVICE_ID="$NPU_DEVICE_ID" \
PORT="$PORT" \
PYTHON_BIN="$PYTHON_BIN" \
MODEL_PATH="$W8A8_DIR" \
KT_GGUF_TEMPLATE="$GGUF_CACHE/dsv4_layer{layer_idx}_mxfp4.gguf" \
KT_THREADPOOL_COUNT=8 \
KT_CPUINFER=128 \
KT_NUM_GPU_EXPERTS=32 \
CHUNKED_PREFILL_SIZE=32768 \
bash tools/launch_ds4flash_npu.sh \
  2>&1 | tee /workspace/code/logs/serve-single-npu.log
```

这个命令必须前台运行。保持该终端不要关，另开一个 SSH 终端做下面的测试。

## 11. 测试顺序

### 11.1 先测服务能不能回答

新宿主机终端执行：

```bash
until curl -sf http://127.0.0.1:8020/health >/dev/null; do sleep 5; done

curl -sS -X POST http://127.0.0.1:8020/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"中国的首都是哪里？","sampling_params":{"max_new_tokens":64,"temperature":0}}'
```

能连贯回答“北京”，且服务不退出，才做性能测试。

### 11.2 单请求 decode 吞吐

新宿主机终端进入同一容器：

```bash
docker exec -it "dsv4-npu${NPU_ID:-5}" bash
```

容器内执行：

```bash
source /workspace/code/dsv4_runtime.env
cd /workspace/code/ktransformers-AK
TARGET_TOKENS_LIST="130 1000 8000" MAX_NEW=1000 REPEAT=3 WARMUP=1 \
  PORT="$PORT" bash tools/decode_throughput_test.sh
```

先看 130/1k/8k 三档。结果中的稳态 `tok/s` 才是单请求 decode 速度，第一轮 warmup 不计入结论。

### 11.3 并发 5、10、80 的聚合吞吐

宿主机执行：

```bash
curl -s http://127.0.0.1:8020/v1/models
```

记下返回结果中的模型 `id`，下面用 `<MODEL_ID>` 替换：

```bash
python3 /home/mem/dsv4/code/scripts/bench_concurrency.py \
  --base-url http://127.0.0.1:8020 \
  --model '<MODEL_ID>' \
  --concurrency 5 10 80 \
  --max-tokens 128 \
  --output-dir /home/mem/dsv4/results/concurrency
```

重点看：成功率、`aggregate completion tok/s`、TTFT p95 和 E2E p95。当前服务固定 `--max-running-requests 1`，所以并发 80 主要是在测排队和聚合吞吐，不代表 80 条序列同时 decode。

### 11.4 GPQA-Diamond 准确率

GPQA 最后做。先确保容器内已经离线安装 EvalScope，并且官方 GPQA 数据已按 EvalScope 离线文档放入缓存。然后在容器内执行：

```bash
source /workspace/code/dsv4_runtime.env
cd /workspace/code/ktransformers-AK
REPEATS=3 \
MODEL_PATH=/workspace/models/DeepSeek-V4-Flash-W8A8 \
PORT="$PORT" \
bash tools/gpqa_accuracy_repeat.sh
```

先跑 3 轮验证流程，再决定是否跑默认 10 轮。只和相同 prompt、sampling、reasoning mode 的结果比较，不能拿不同评测协议的官方分数直接判断部署精度。

## 12. 停止

模型服务所在终端按 `Ctrl+C`，然后输入 `exit`。启动脚本使用 `docker run --rm`，退出容器后容器会自动删除；`/home/mem/dsv4` 下的代码、权重、GGUF、日志和结果不会删除。

下次继续实验时不需要重新 `docker load`、构建镜像、打补丁、编译或转换 GGUF，只需重新执行第 5、6、10 步。
