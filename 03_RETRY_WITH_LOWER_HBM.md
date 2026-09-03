# 容器已删除后，用较低显存配置重新启动

这份说明只适用于下面这种情况：

- `/data/models/dsv4` 还在；
- W8A8 模型和原始 MXFP4 模型还在；
- 43 个 MXFP4 GGUF 已经转换完成；
- KTransformers、SGLang 补丁和 kt-kernel 编译已经做过；
- 上一次启动在 `Load weight end` 之后、创建 KV Cache 时因显存不足失败；
- 上一次的容器已经删除。

本次不重新打补丁、不重新编译、不重新转换权重。目标是使用物理 NPU 5、端口 9108，把每层常驻 NPU 的专家数从 32 降到 24，先验证服务能否启动。

## 1. 先理解哪些东西还在

删除 Docker 容器只删除容器本身。下面这些文件位于宿主机 `/data/models/dsv4`，之前通过目录挂载进入容器，所以应当还在：

```text
/data/models/dsv4/
├── code/
│   ├── ktransformers-AK/           # 已打补丁的源码和编译产物
│   ├── cann-recipes-infer/
│   ├── scripts/
│   └── logs/                       # 上一次的启动日志
└── models/
    ├── DeepSeek-V4-Flash-W8A8/     # NPU 权重
    ├── DeepSeek-V4-Flash/          # 原始 MXFP4 权重
    └── cache/                      # 43 个 CPU GGUF
```

Docker 镜像和 Docker 容器也不是一回事。删除容器通常不会删除镜像。

## 2. 宿主机检查 NPU 5、端口和镜像

下面命令都在宿主机执行，不要在容器里执行。

```bash
export DSV4_ROOT=/data/models/dsv4
export NPU_ID=5
export SERVICE_PORT=9108

npu-smi info
ss -ltnp | grep ":${SERVICE_PORT}" || true
docker ps -a --filter 'name=^/dsv4-npu5$'
docker image inspect dsv4-offload-env:cann85-910b2 >/dev/null \
  && echo '派生镜像存在'
```

检查结果：

- NPU 5 不能有其他模型进程占用大量 HBM；
- 9108 端口应当没有输出；
- 不应存在名为 `dsv4-npu5` 的旧容器；
- 最后一条应输出 `派生镜像存在`。

如果 NPU 5 在模型启动前就占用了几 GB，先通过 `npu-smi info` 下方的进程表确认占用者。不要直接结束不认识的业务进程。

如果只有派生镜像不存在，才重新构建镜像：

```bash
cd /data/models/dsv4/image
docker build -t dsv4-offload-env:cann85-910b2 .
```

这一步只构建运行环境，不会重新转换模型权重。

## 3. 宿主机检查保留下来的文件

```bash
test -f /data/models/dsv4/code/ktransformers-AK/.dsv4_patch_applied \
  && echo '补丁标记存在'

find /data/models/dsv4/models/cache \
  -maxdepth 1 -name 'dsv4_layer*_mxfp4.gguf' | wc -l

test -d /data/models/dsv4/models/DeepSeek-V4-Flash-W8A8 \
  && echo 'W8A8 目录存在'
```

GGUF 文件数量必须是 `43`。如果输出不是 43，先不要启动模型，也不要重新执行整个转换流程；先检查具体缺少哪一层。

## 4. 重新创建并进入容器

仍在宿主机执行：

```bash
export NPU_ID=5
export SERVICE_PORT=9108
bash /data/models/dsv4/code/scripts/start_single_npu_container.sh
```

成功后当前终端会进入新容器。终端提示符发生变化属于正常现象。

## 5. 在新容器里检查旧编译产物

从这一节开始，命令都在刚进入的新容器内执行。

```bash
source /workspace/code/dsv4_runtime.env

export REPO=/workspace/code/ktransformers-AK
export PYTHON_BIN=$(command -v python3.11 || command -v python3)
export W8A8_DIR=/workspace/models/DeepSeek-V4-Flash-W8A8
export GGUF_CACHE=/workspace/models/cache
export PYTHONPATH="$REPO/third_party/sglang/python:$REPO/kt-kernel${PYTHONPATH:+:$PYTHONPATH}"

echo "physical NPU=$NPU_DEVICE_ID, port=$PORT"
test -f "$REPO/.dsv4_patch_applied" && echo '补丁标记正常'
"$PYTHON_BIN" -c 'import kt_kernel; print("kt_kernel OK")'
find "$GGUF_CACHE" -maxdepth 1 -name 'dsv4_layer*_mxfp4.gguf' | wc -l
```

应当看到：

```text
physical NPU=5, port=9108
补丁标记正常
kt_kernel OK
43
```

如果 `import kt_kernel` 失败，说明编译产物没有保留下来或当前镜像环境与上次不同。这时只重新执行 `02_EXPERIMENT.md` 的“编译 kt-kernel”一节，不要重新转换 GGUF。

## 6. 运行启动前预检

容器内执行：

```bash
cd "$REPO"
GGUF_DIR="$GGUF_CACHE" GGUF_SUFFIX=_mxfp4 bash tools/e2e_preflight.sh
```

看到 `PASS` 才继续。

## 7. 保存上次日志

本次启动会重新写 `serve-single-npu.log`。先把旧日志改名保存：

```bash
mkdir -p /workspace/code/logs

if [ -f /workspace/code/logs/serve-single-npu.log ]; then
  mv /workspace/code/logs/serve-single-npu.log \
    "/workspace/code/logs/serve-single-npu-previous-$(date +%Y%m%d-%H%M%S).log"
fi
```

这些日志实际保存在宿主机：

```text
/data/models/dsv4/code/logs/
```

## 8. 使用 24 个 NPU 常驻专家重新启动

容器内执行完整命令：

```bash
cd "$REPO"

NPU_DEVICE_ID="$NPU_DEVICE_ID" \
PORT="$PORT" \
PYTHON_BIN="$PYTHON_BIN" \
MODEL_PATH="$W8A8_DIR" \
KT_GGUF_TEMPLATE="$GGUF_CACHE/dsv4_layer{layer_idx}_mxfp4.gguf" \
KT_THREADPOOL_COUNT=8 \
KT_CPUINFER=128 \
KT_NUM_GPU_EXPERTS=24 \
CHUNKED_PREFILL_SIZE=8192 \
MEM_FRACTION=0.81 \
bash tools/launch_ds4flash_npu.sh \
  2>&1 | tee /workspace/code/logs/serve-single-npu.log
```

这次与上一次的主要区别是：

```text
KT_NUM_GPU_EXPERTS: 32 → 24
CHUNKED_PREFILL_SIZE: 32768 → 8192
KV Cache dtype: 仍使用默认 BF16，不修改
```

降低常驻专家数会让更多专家在 CPU 上计算，可能降低 decode 吞吐，但能明显释放 HBM。当前目标是先让服务完整启动，不是立即测 32k prompt 的最高性能。

这个启动命令保持在前台运行，不要退出当前容器终端。另开一个 SSH 终端执行后面的检查。

## 9. 观察是否越过上一次错误

在第二个宿主机终端执行：

```bash
tail -f /data/models/dsv4/code/logs/serve-single-npu.log
```

另一个宿主机终端可以观察 NPU 5：

```bash
watch -n 2 npu-smi info
```

重点观察日志是否依次出现类似信息：

```text
Load weight end ...
Using KV cache dtype bfloat16
SWAC4C128KVPool mem usage: ...
Capture npu graph end ...
```

`Using KV cache dtype bfloat16` 是正常信息。成功创建 `SWAC4C128KVPool`，就表示已经越过上一次的 AssertionError。

## 10. 检查服务并发送一个短请求

在第二个宿主机终端执行：

```bash
curl -sS http://127.0.0.1:9108/health
```

如果服务还在加载，可以等待几分钟后重试。健康检查成功后发送短请求：

```bash
curl -sS -X POST http://127.0.0.1:9108/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"中国的首都是哪里？","sampling_params":{"max_new_tokens":64,"temperature":0}}'
```

能返回连贯内容，并且模型进程不退出，才算本轮启动成功。

## 11. 如果 24 个专家仍然报同一个 AssertionError

先确认失败日志仍然是：

```text
assert c128_max_total_num_tokens > 0
```

然后依次处理：

1. 确认 NPU 5 启动前没有其他进程占用 HBM；
2. 保持 KV Cache 为 BF16，不要擅自改成 FP8；
3. 把上面启动命令中的 `KT_NUM_GPU_EXPERTS=24` 改成 `16` 再试；
4. 仍然失败时，保存下面命令的输出再分析：

```bash
grep -nE \
  'Load weight end|mem usage|available_gpu_mem|available.*memory|KV cache dtype|SWAC4C128|reserved ND streaming|AssertionError' \
  /workspace/code/logs/serve-single-npu.log
```

不要通过反复提高 `MEM_FRACTION` 作为长期解决办法。提高它虽然可能给 KV 池更多空间，却会压缩长 prefill 所需的激活余量，后面可能在请求阶段 OOM。

## 12. 如何停止并删除这次容器

如果模型在前台运行，在模型终端按一次 `Ctrl+C`，等待子进程退出，然后执行：

```bash
exit
```

启动脚本使用了 `--rm`，正常退出后容器会自动删除。宿主机确认：

```bash
docker ps -a --filter 'name=^/dsv4-npu5$'
```

如果模型终端失去响应，在宿主机执行：

```bash
docker stop --time 60 dsv4-npu5
```

停止或删除容器不会删除 `/data/models/dsv4` 中的权重、GGUF、代码和日志。
