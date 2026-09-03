# 已有镜像和转换产物：重新部署

这份说明适用于下面这种情况：

- `/data/models/dsv4` 还在；
- 派生镜像 `dsv4-offload-env:cann85-910b2` 已经构建好；
- W8A8 模型和原始 MXFP4 模型还在；
- 43 个 MXFP4 GGUF 已经转换完成；
- KTransformers、SGLang 补丁和 kt-kernel 编译已经做过；
- 上一次容器已经停止或删除，无论上一次成功还是失败。

本次不重新导入或构建镜像、不重新打补丁、不重新编译、不重新转换权重。目标是使用物理 NPU 5、端口 9108 重新创建容器。首次重启先用每层 24 个 NPU 常驻专家验证；确认物理卡映射和生成结果后，再恢复 32 个专家测正式性能。

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
export ASCEND_LOGICAL_ID=4
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

这里的两个卡号不能混用：

```text
NPU_ID=5                 宿主机物理卡，对应 /dev/davinci5
ASCEND_LOGICAL_ID=4      容器逻辑卡；这台服务器缺少物理卡4
```

如果把 `ASCEND_RT_VISIBLE_DEVICES` 错设成 5，实际会落到物理 NPU 6。NPU 6 只有约 50 GB HBM 时，32 个常驻专家会在 KV Cache 创建阶段失败。仓库启动脚本会自动计算逻辑编号；这里显式写出 4 是为了再次部署时更容易核对。
CANN 8.5 对三种设备编号的官方说明：
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/appdevgapi/aclcppdevg_03_2059.html

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
export ASCEND_LOGICAL_ID=4
export SERVICE_PORT=9108
bash /data/models/dsv4/code/scripts/start_single_npu_container.sh
```

启动前脚本必须输出：

```text
物理 NPU：5（/dev/davinci5）
容器逻辑 NPU：4（进程内会成为 npu:0）
```

成功后当前终端会进入新容器。终端提示符发生变化属于正常现象。如果脚本显示物理 5 对应逻辑 5，先停止，不要加载权重。

## 5. 在新容器里检查旧编译产物

从这一节开始，命令都在刚进入的新容器内执行。

```bash
source /workspace/code/dsv4_runtime.env

export REPO=/workspace/code/ktransformers-AK
export PYTHON_BIN=$(command -v python3.11 || command -v python3)
export W8A8_DIR=/workspace/models/DeepSeek-V4-Flash-W8A8
export GGUF_CACHE=/workspace/models/cache
export PYTHONPATH="$REPO/third_party/sglang/python:$REPO/kt-kernel${PYTHONPATH:+:$PYTHONPATH}"

echo "physical=$NPU_PHYSICAL_ID, container logical=$ASCEND_LOGICAL_ID, visible=$ASCEND_RT_VISIBLE_DEVICES, port=$PORT"
test -f "$REPO/.dsv4_patch_applied" && echo '补丁标记正常'
"$PYTHON_BIN" -c 'import kt_kernel; print("kt_kernel OK")'
find "$GGUF_CACHE" -maxdepth 1 -name 'dsv4_layer*_mxfp4.gguf' | wc -l
```

应当看到：

```text
physical=5, container logical=4, visible=4, port=9108
补丁标记正常
kt_kernel OK
43
```

如果 `import kt_kernel` 失败，说明编译产物没有保留下来或当前镜像环境与上次不同。这时只重新执行 `02_EXPERIMENT.md` 的“编译 kt-kernel”一节，不要重新转换 GGUF。

加载大模型前，先用一小块显存确认物理卡映射。容器内执行：

```bash
"$PYTHON_BIN" - <<'PY'
import time
import torch
import torch_npu

x = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="npu:0")
print("已在进程内 npu:0 分配 128 MiB；请在宿主机观察 15 秒")
time.sleep(15)
PY
```

这 15 秒内在另一个宿主机终端执行 `npu-smi info`。临时进程必须出现在物理 NPU 5；如果出现在 NPU 6，不要继续加载权重。

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
ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
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

权重加载期间必须看到物理 NPU 5 的显存上升。如果显存出现在 NPU 6，立即按 `Ctrl+C` 停止；这表示容器逻辑编号仍然设置错了。

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
curl -sS --max-time 5 -o /dev/null \
  -w 'health HTTP %{http_code}\n' \
  http://127.0.0.1:9108/health
```

如果服务还在加载，可以等待几分钟后重试。输出 `health HTTP 200` 即成功；健康接口没有正文属于正常现象。然后发送短请求：

```bash
curl -sS -X POST http://127.0.0.1:9108/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"中国的首都是哪里？","sampling_params":{"max_new_tokens":64,"temperature":0}}'
```

能返回连贯内容，并且模型进程不退出，才算本轮启动成功。

## 11. 在物理 NPU 5 上恢复 32 个专家

24 个专家能回答后，说明重新部署和卡号映射正确。若要按原教程测试性能，先正常停止并删除当前容器，再按第 4 节重新创建容器，然后把第 8 节启动命令改为：

```text
KT_NUM_GPU_EXPERTS=32
```

其他参数先保持不变。物理 NPU 5 有完整 64 GB HBM，32 个专家大概率可以创建 KV Cache；仍需以日志中的 `SWAC4C128KVPool mem usage` 和实际生成请求为准。需要测试 32k prompt 时，再将 `CHUNKED_PREFILL_SIZE` 从 8192 调到 32768。

## 12. 如果 24 个专家仍然报同一个 AssertionError

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

## 13. 本次实测踩坑复盘（最终已跑通）

### 13.1 表面上是显存不足，真正起因是设备编号映射错了

本次计划使用宿主机物理 NPU 5，但服务器的设备节点中没有编号 4：

```text
宿主机物理设备：0 1 2 3 5 6 ...
容器逻辑设备：  0 1 2 3 4 5 ...
```

旧启动脚本直接执行了：

```text
NPU_ID=5
ASCEND_RT_VISIBLE_DEVICES=5
```

其中第一个 `5` 是宿主机物理编号，第二个 `5` 却是容器逻辑编号。容器逻辑 5 实际对应宿主机物理 NPU 6，因此模型没有加载到计划中的物理 NPU 5。

物理 NPU 6 当时只有约 50 GB 可用 HBM。使用 32 个 NPU 常驻专家时，权重虽然已经完成加载，但剩余空间不足以创建 KV Cache，最终触发：

```text
assert c128_max_total_num_tokens > 0
```

正确关系是：

```text
宿主机 NPU_ID=5
容器 ASCEND_LOGICAL_ID=4
进程内部使用 npu:0
```

修正映射并确认显存出现在物理 NPU 5 后，服务完成启动和请求验证，本次单卡 CPU/NPU offload 部署最终跑通。

### 13.2 为什么一开始容易误判成 offload 或量化问题

本次日志顺序大致是：

```text
Load weight end ...
Using KV cache dtype bfloat16
创建 SWAC4C128 KV 内存池
AssertionError
unclosed zmq.Socket ...
```

这些信息应这样理解：

| 现象 | 正确含义 |
|---|---|
| `Load weight end` | 权重加载阶段已经完成，不代表整个服务已经就绪 |
| `Using KV cache dtype bfloat16` | 正常默认配置，不是量化精度错误 |
| `c128_max_total_num_tokens > 0` 断言失败 | 当前实际 NPU 的剩余 HBM 不够创建最低 KV 池 |
| `unclosed zmq.Socket` | Scheduler 异常退出后的清理提示，不是根因 |
| `Only CUDA/HIP/XPU support AWQ currently` | 当前路线使用 compressed-tensors W8A8，与本次失败无关 |
| 物理 NPU 5 没有进程、NPU 6 显存上涨 | 设备映射错误，应立即停止加载 |

完整 W8A8 权重约 275 GiB，不可能全部常驻一张 64 GB NPU。日志已经走到 KV 内存池阶段时，“完全没有执行 CPU MoE offload”不是首要怀疑方向；应先确认实际占用的是哪张物理卡，以及这张卡真实可用的 HBM。

### 13.3 `/health` 看起来没输出，其实已经成功

修正显存配置后，日志持续出现：

```text
INFO ... GET /health HTTP/1.1 200 OK
```

同时 Uvicorn 显示：

```text
Uvicorn running on http://0.0.0.0:9108
```

这已经说明 HTTP 服务可访问。`/health` 可以返回空响应体，所以普通 `curl -s` 会表现为终端没有文字。以后统一使用：

```bash
curl -sS --max-time 5 -o /dev/null \
  -w 'health HTTP %{http_code}\n' \
  http://127.0.0.1:9108/health
```

输出 `health HTTP 200` 表示健康检查通过。最终仍要再发送一次 `/generate` 请求并确认回答连贯，才算端到端部署成功。

### 13.4 以后重跑的最短检查顺序

1. 宿主机用 `ls /dev/davinci*` 检查物理编号是否连续；
2. 使用 `NPU_ID=5 ASCEND_LOGICAL_ID=4` 创建容器；
3. 确认脚本打印“物理 5、容器逻辑 4”；
4. 先做第 5 节的 128 MiB 小分配，在宿主机确认进程落到物理 NPU 5；
5. 再加载大模型，同时用 `npu-smi info` 观察目标卡；
6. 日志到达 `SWAC4C128KVPool` 后检查是否成功创建 KV 池；
7. 用带状态码的 `/health` 命令验证 HTTP；
8. 用 `/generate` 验证真实推理输出。

这次最大的经验是：**看到显存错误时，先确认“实际物理卡”，再调整专家数、KV Cache 或量化参数。**

## 14. 如何停止并删除这次容器

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
