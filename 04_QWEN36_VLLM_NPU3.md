# Tensor-0.1-Flash-35B-A3B：单卡 NPU 3 vLLM-Ascend 部署

这套操作与仓库里原有的 DeepSeek-V4/SGLang/KTransformers offload 实验相互独立。目标是只把 `/data/models/Tensor-0.1-Flash-35B-A3B` 以只读方式挂入新容器，只暴露当前空闲且约有 59GB 可用 HBM 的物理 NPU 3。部署脚本、压测脚本、日志和结果都收口到这个模型目录。

服务器上的 `quay.io/ascend/vllm-ascend:v0.19.1rc1-openeuler` 已成功部署过 Qwen3-14B，因此本方案优先复用这套镜像，不新建或修改 Conda 推理环境。Qwen3.6-35B-A3B 的官方最低 vLLM-Ascend 支持版本是 v0.18.0rc1，v0.19.1rc1 在版本线上满足要求。仍需现场验证定制 Tensor 权重的 `architectures` 和量化格式；“版本够新”不等于每种第三方量化权重都受支持。

默认值：

```text
镜像     quay.io/ascend/vllm-ascend:v0.19.1rc1-openeuler
容器     qwen36-flash-npu3
物理卡   NPU 3（当前约 59GB 可用）
端口     9108
上下文   8192
服务端最大序列数 32（覆盖 30 并发压测）
```

脚本不会停止、删除或覆盖任何已有容器，也不会结束 NPU 进程。发现同名容器或端口占用时直接退出，并要求操作者查看 `npu-smi` 后显式确认卡 3 空闲。

## 1. 服务器目录布局

统一使用下面的布局；模型文件本身保持原样，新增内容只进入 `deployment/` 和 `results/`：

```text
/data/models/Tensor-0.1-Flash-35B-A3B/
├── config.json、*.safetensors 等模型原文件
├── deployment/
│   ├── 04_QWEN36_VLLM_NPU3.md
│   ├── qwen36.conf
│   ├── 01_create_container.sh
│   ├── 02_start_service.sh
│   ├── 03_check_service.sh
│   ├── 04_benchmark.sh
│   └── bench_concurrency.py
└── results/
    └── qwen36-vllm-npu3/
        ├── logs/
        ├── metrics/
        └── bench/
```

然后执行：

```bash
chmod +x /data/models/Tensor-0.1-Flash-35B-A3B/deployment/{01_create_container,02_start_service,03_check_service,04_benchmark}.sh
chmod +x /data/models/Tensor-0.1-Flash-35B-A3B/deployment/bench_concurrency.py
```

## 2. 只读预检

```bash
export MODEL_PATH=/data/models/Tensor-0.1-Flash-35B-A3B
export IMAGE=quay.io/ascend/vllm-ascend:v0.19.1rc1-openeuler

npu-smi info
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'
ss -ltnp | grep ':9108' || true
docker image inspect "$IMAGE" --format 'arch={{.Architecture}} id={{.Id}}'

du -sh "$MODEL_PATH"
find "$MODEL_PATH" -maxdepth 1 -type f -printf '%f %s\n' | sort
sed -n '1,240p' "$MODEL_PATH/config.json"
test ! -f "$MODEL_PATH/quantization_config.json" || \
  sed -n '1,240p' "$MODEL_PATH/quantization_config.json"
```

必须人工确认：NPU 3 没有别的业务进程、9108 未占用、镜像架构是 ARM64、模型包含完整 config/tokenizer/权重分片。17GB 很可能是量化权重；能否加载最终取决于其量化方法是否被这个 vLLM-Ascend 镜像支持。

进一步检查镜像版本，不启动模型：

```bash
docker run --rm \
  --device /dev/davinci3 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  "$IMAGE" bash -lc '
    python - <<"PY"
import torch, torch_npu, vllm
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("vllm", vllm.__version__)
print("npu available", torch.npu.is_available())
print("visible devices", torch.npu.device_count())
PY'
```

## 3. 依次执行四个脚本

### 3.1 创建独立容器

```bash
export NPU_ID=3
export MODEL_PATH=/data/models/Tensor-0.1-Flash-35B-A3B
export IMAGE=quay.io/ascend/vllm-ascend:v0.19.1rc1-openeuler
export CONTAINER_NAME=qwen36-flash-npu3
export SERVICE_PORT=9108

## 看过 npu-smi 输出并确认物理卡 3 没有其他进程后才设置
export NPU3_CONFIRMED_FREE=YES

bash /data/models/Tensor-0.1-Flash-35B-A3B/deployment/01_create_container.sh
```

该脚本只创建一个绑定 NPU 3 的空闲容器，不会启动模型服务。模型目录只读挂载，不会修改权重。

### 3.2 启动模型服务

```bash
bash /data/models/Tensor-0.1-Flash-35B-A3B/deployment/02_start_service.sh
```

服务参数：TP=1、8K context、服务端最大序列数 32、eager、显存利用率 0.85。启动日志：

```bash
docker logs -f qwen36-flash-npu3 2>&1 | \
  tee /data/models/Tensor-0.1-Flash-35B-A3B/results/qwen36-vllm-npu3/logs/manual-follow.log
```

另一个终端观察卡 3：

```bash
watch -n 2 npu-smi info
```

如果日志显示不认识 `--trust-remote-code` 或其他参数，先执行以下命令确认该镜像实际 CLI，再根据输出调整；不要盲目升级正在使用的镜像：

```bash
docker run --rm "$IMAGE" vllm serve --help | less
```

### 3.3 健康检查与单请求

```bash
bash /data/models/Tensor-0.1-Flash-35B-A3B/deployment/03_check_service.sh
```

如果模型的 chat template 不完整，改测 `/v1/completions`，或者修正模型目录中的 tokenizer/chat-template 配置副本；不要原地破坏唯一权重。

### 3.4 运行 5/10/20/30 并发压测

压测程序使用 Python 标准库，无须 Conda 或 pip 安装。每档默认请求数为 `max(20, concurrency)`，为了使 30 并发有更稳定的吞吐统计，现场建议每档 60 个请求、固定输出 256 token：

```bash
bash /data/models/Tensor-0.1-Flash-35B-A3B/deployment/04_benchmark.sh
```

第四个脚本自动使用 `vllm_bench`，每档 60 个请求、固定最多输出 256 token。需要覆盖时，例如：

```bash
BENCH_REQUESTS=100 BENCH_OUTPUT_TOKENS=512 \
  bash /data/models/Tensor-0.1-Flash-35B-A3B/deployment/04_benchmark.sh
```

结果包含：

- 输入 token 吞吐（prompt tokens / 档位墙钟时间）；
- 输出 token 吞吐（completion tokens / 档位墙钟时间）；
- 总 token 吞吐、请求吞吐；
- E2E latency、TTFT、TPOT 的 mean/p50/p95/p99；
- 每个请求的 token 数、延迟和错误。

每档生成一份 JSON。服务必须返回 OpenAI `usage` 才会计算真实 token 吞吐；脚本不会用字符数或 SSE chunk 数冒充 token。

压测时建议同时保存资源采样：

```bash
mkdir -p /data/models/Tensor-0.1-Flash-35B-A3B/results/qwen36-vllm-npu3/metrics
npu-smi info watch -i 3 -c 1 \
  > /data/models/Tensor-0.1-Flash-35B-A3B/results/qwen36-vllm-npu3/metrics/npu3-watch.log 2>&1 &
echo $! > /data/models/Tensor-0.1-Flash-35B-A3B/results/qwen36-vllm-npu3/metrics/npu-smi-watch.pid
```

不同 `npu-smi` 版本的 watch 参数可能不同；先运行 `npu-smi info watch -h`。不支持时使用另一个终端的 `watch -n 2 npu-smi info`，压测结束后手动停止。

## 4. 常见失败顺序

1. **模型架构不支持**：日志出现 `unsupported architecture/model type`。`v0.18.0rc1` 是 Qwen3.6-35B-A3B 的最低支持线，但定制的 Tensor 模型 config 可能不同；优先比较其 `architectures` 与官方模型。
2. **量化方法不支持**：出现 `Unknown quantization method` 或 CUDA-only AWQ/GPTQ/FP4 算子错误。需要与制作者确认权重格式，或转换为该版本支持的 Ascend W8A8/W4A8，不能靠加启动参数解决。
3. **启动 OOM**：依次降为 `MAX_MODEL_LEN=4096`、`MAX_NUM_SEQS=1`、`MAX_NUM_BATCHED_TOKENS=2048`、`MEMORY_UTILIZATION=0.80`；保持 eager。注意较低的 `gpu-memory-utilization` 会缩小 KV cache，但无法解决权重本身放不下的问题。
4. **驱动/CANN 不匹配**：比较这次镜像与服务器上已正常运行的同标签容器。不要在宿主机或现有容器里升级 CANN/torch-npu。
5. **30 并发排队或失败**：初次启动的 `MAX_NUM_SEQS=8` 只允许有限并行调度。服务验证稳定后，可停止本容器并以 `MAX_NUM_SEQS=32` 重建；若显存不足则设 16。并发客户端数量不等于服务端能同时执行的序列数量。

以更低显存配置重新创建时，先停止本次独立容器：

```bash
docker stop qwen36-flash-npu3
```

确认这是本次容器且日志已经保存后，才手动删除：

```bash
docker inspect qwen36-flash-npu3 --format '{{.Name}} {{.Config.Image}}'
docker rm qwen36-flash-npu3

MAX_MODEL_LEN=4096 \
MAX_NUM_SEQS=1 \
MAX_NUM_BATCHED_TOKENS=2048 \
MEMORY_UTILIZATION=0.80 \
NPU3_CONFIRMED_FREE=YES \
bash /data/models/Tensor-0.1-Flash-35B-A3B/deployment/01_create_container.sh
```

## 5. Conda 备用路线

只有现成镜像确实不支持权重时才考虑新环境。不要修改已在服务的 Conda 环境：

```bash
conda create -n qwen36-vllm-ascend python=3.11 -y
conda activate qwen36-vllm-ascend
```

随后必须依据宿主机驱动/CANN 版本选择成套的 `torch`、`torch-npu`、`vllm`、`vllm-ascend`，不可混装。现场先记录：

```bash
cat /usr/local/Ascend/driver/version.info
cat /usr/local/Ascend/ascend-toolkit/latest/*-linux/data/version.info 2>/dev/null || true
python -V
```

优先复制“同机上已正常运行的 v0.19.1rc1-openEuler 容器”的版本组合；不要直接安装最新版覆盖已有环境。`vllm_bench` 只用于从宿主机发压测请求，不要往里面安装 `torch-npu` 或让它承担模型推理。
