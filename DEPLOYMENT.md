# DeepSeek-V4-Flash：910B2 单卡部署与实验说明

## 1. 实验目标和边界

目标硬件：

- 2 × Ascend 910B2（64 GiB HBM），物理设备号为 3 和 5；先用物理卡 3 做单卡实验。
- Kunpeng-920，aarch64，256 核，8 NUMA，支持 `asimddp`/SDOT。
- 宿主机安装 Ascend 驱动和固件；CANN、PyTorch NPU、框架及自定义算子位于容器。

采用 CANN Recipes 的单 NPU + CPU MoE offload 路径：Attention、Router、Shared Expert 和热 Routed Experts 在 NPU 上以 W8A8 计算，其余 Routed Experts 在 CPU 上读取原生 MXFP4 GGUF 计算。

当前交付是单发路径，单实例通常固定 `--max-running-requests 1`。客户端并发 100 主要测排队；两卡启动两个独立实例时，最多约 2 条请求同时 decode。它不是原生 100 序列 batching。

## 2. 仓库原则

本仓库只保存：

- 部署和实验说明。
- 环境配置模板。
- 验机、物料下载、在线压测和 GPQA-Diamond 脚本。
- 固定版本的 GPQA 数据集子模块。

不保存 Docker 镜像、模型权重、GGUF 产物、第三方框架源码、Python wheel、日志或评测结果。

## 3. 克隆及 GPQA 数据

```bash
git clone --recurse-submodules <DS910_REPOSITORY_URL> Ds910
cd Ds910
cp config/env.example .env
```

GPQA 固定到 ModelScope `modelscope/gpqa` 的提交：

```text
3481c2b00a22d8256f6b845f256aa4cd4b654132
```

如果普通克隆时未拉子模块：

```bash
git submodule sync
git submodule update --init --depth 1 benchmarks/gpqa
```

内网有自己的 GPQA Git 镜像时：

```bash
git config submodule.benchmarks/gpqa.url \
  https://git.internal.example.com/datasets/gpqa.git
git submodule update --init benchmarks/gpqa
```

不要把 GPQA 题目打印到公开日志或报告中；报告只保存汇总分数、样本数、配置和失败样本编号。

## 4. 内网配置

修改 `.env`，至少填好：

```bash
HF_ENDPOINT=https://hf.internal.example.com
MODELSCOPE_ENDPOINT=https://modelscope.internal.example.com
PIP_INDEX_URL=https://pypi.internal.example.com/simple

CANN_RECIPES_URL=https://git.internal.example.com/cann/cann-recipes-infer.git
KTRANSFORMERS_URL=https://git.internal.example.com/kvcache-ai/ktransformers.git
SGLANG_URL=https://git.internal.example.com/iforgetmyname/sglang.git
LLAMA_CPP_URL=https://git.internal.example.com/ggerganov/llama.cpp.git

DOCKER_IMAGE=harbor.internal.example.com/ai/sglang:deepseek-v4-npu-910b
```

不要把 token、密码或内部证书内容写入 `.env` 后提交；`.env` 已被忽略。

完全离线后可设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 5. 验机

```bash
bash scripts/check_host.sh .env
```

单实例门槛：

| 项目 | 最低/建议值 |
|---|---:|
| 可用 DDR | 最低 160 GiB，建议 256 GiB |
| 转换期磁盘 | 600 GiB |
| 服务常驻磁盘 | 约 415 GiB |
| CPU | ARMv8.2-A + NEON dotprod/SDOT |
| NPU | 单张 64 GiB 910B2 |

双实例会共同争抢 DDR，建议至少 384 GiB 可用内存，最好 512 GiB 或更高。Linux page cache 可能共享部分只读 GGUF 页面，但运行时转换、NUMA 本地缓存和两个进程仍可能产生额外内存占用，不能按单实例简单估算。

宿主机不要求安装 CANN Toolkit 或算子库，但必须满足：

- `npu-smi info` 正常。
- 指定 `/dev/davinci*`、`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc` 存在。
- 宿主机驱动与镜像内 CANN 8.5 兼容。
- 容器启动时挂载驱动和上述设备。

## 6. 下载物料

先只下载固定源码，不下载约 425 GiB 权重：

```bash
bash scripts/fetch_assets.sh .env
```

确认内网带宽、模型路径和磁盘后，将 `.env` 改为：

```bash
CONFIRM_LARGE_DOWNLOAD=1
```

再次运行：

```bash
bash scripts/fetch_assets.sh .env
```

需要的两套模型权重是：

| 位置 | 模型 | 约占用 |
|---|---|---:|
| NPU W8A8 | `sgl-npu/DeepSeek-V4-Flash-W8A8` | 275 GiB |
| CPU 转换源 | `deepseek-ai/DeepSeek-V4-Flash` 原生 MXFP4 | 150 GiB |
| CPU 转换产物 | 43 层 MXFP4 GGUF | 138 GiB |

首次复现不要替换为 0731、第三方 W8A8、MTP 权重或重新量化权重。NPU W8A8 和 CPU MXFP4 必须对应同一模型版本及量化基底。

镜像使用：

```text
lmsysorg/sglang:deepseek-v4-npu-910b
```

若镜像已在内部 Harbor，只需把 `.env` 的 `DOCKER_IMAGE` 指向现有标签，不需要宿主机再安装算子库。

## 7. 打补丁和编译

交付目录：

```text
$SOURCE_ROOT/cann-recipes-infer/
  integration/sglang/dsv4-flash-single-npu-moe-offload/
```

当前固定基线：

| 仓库 | Revision |
|---|---|
| CANN Recipes delivery | `1a7fbd34` |
| KTransformers | `d7b5b49` |
| SGLang | `298193eb3` |
| llama.cpp | `a94e6ff` |

进入交付目录，严格按照该 revision 自带的使用指南布置三仓目录，然后执行：

```bash
bash apply_all.sh "$REPO"
```

不要在补丁失败时强行继续。编译 `kt-kernel` 时必须明确关闭不适用的 ARM 路径：

```bash
export CPUINFER_USE_ASCEND_NPU=1
export CPUINFER_ARM_SVE=OFF
export CPUINFER_ARM_BF16=OFF
export CPUINFER_ARM_I8MM=OFF

python setup.py build_ext --inplace
pip install -e .
```

编译日志验收：

```text
DOTPROD=ON
SVE=OFF
BF16=OFF
I8MM=OFF
```

910B 走交付中的 CANN 8.5 容器脚本，不要使用 A3/CANN 9.0 路径。

## 8. MXFP4 转 GGUF及预检

使用交付的：

```text
batch_convert_mxfp4_layers_mp.py
verify_mxfp4_gguf_set.py
cpu_moe_reference_check_mxfp4.py
e2e_preflight.sh
```

输出必须包含：

```text
dsv4_layer0_mxfp4.gguf
...
dsv4_layer42_mxfp4.gguf
```

转换和验证顺序：

1. 从原生 MXFP4 转43层 GGUF，建议先用16个转换进程。
2. 执行 GGUF 集合 SHA256与 bit-exact 对账。
3. 执行单层 CPU kernel参考对账，官方参考 cosine约 `0.999939`。
4. 执行 `GGUF_DIR=... bash tools/e2e_preflight.sh`，退出码必须为0。

在模型输出连贯且 GPQA验收完成前，不要删除原生 MXFP4转换源。

## 9. 单卡启动与冒烟

先使用物理卡3：

```bash
export NPU_DEVICE_ID=3
export PORT=8020
export MODEL_PATH=/workspace/models/DeepSeek-V4-Flash-W8A8
export KT_GGUF_TEMPLATE='/workspace/models/gguf/dsv4_layer{layer_idx}_mxfp4.gguf'

export KT_THREADPOOL_COUNT=8
export KT_CPUINFER=128
export KT_NUM_GPU_EXPERTS=32

export KT_DYNAMIC_RESIDENT=1
export KT_PREFILL_STREAM=1
export KT_SIDE_STREAM=1
export KT_MXFP4_DEPOOL=1
export KT_MXFP4_GGUF_DEDUP=1
export KT_DECODE_TIMING=1

bash tools/launch_ds4flash_npu.sh
```

检查：

```bash
curl -f http://127.0.0.1:8020/health
curl -s http://127.0.0.1:8020/v1/models
```

再发送一条 `temperature=0`、`max_tokens=64` 的中文请求，要求输出连贯且无乱码、NaN或重复异常。当前版本 NPU静态占用可接近50多 GiB；不要按早期文章的16～20 GiB估算。

## 10. 单卡性能和并发

先发送5～10次预热请求，让 GGUF page cache、图和热专家路径稳定。然后执行：

```bash
bash scripts/bench_serving.sh .env
```

默认测试：

```text
input=1024 tokens
output=128 tokens
concurrency=5,10,80
prompts=200
```

如果负载机没有 vLLM 客户端，可以改用仓库中只依赖 Python 3 标准库的独立脚本。一次只跑一个档位：

```bash
python3 scripts/bench_concurrency.py \
  --base-url http://127.0.0.1:8020 \
  --model DeepSeek-V4-Flash \
  --concurrency 5 \
  --requests 20 \
  --max-tokens 128

python3 scripts/bench_concurrency.py \
  --base-url http://127.0.0.1:8020 \
  --model DeepSeek-V4-Flash \
  --concurrency 10 \
  --requests 20 \
  --max-tokens 128

python3 scripts/bench_concurrency.py \
  --base-url http://127.0.0.1:8020 \
  --model DeepSeek-V4-Flash \
  --concurrency 80 \
  --requests 80 \
  --max-tokens 128
```

脚本统计精确的聚合输出 tok/s、总 tok/s、request/s、成功率以及 E2E/TTFT/TPOT 分位数，JSON 结果写入 `results/concurrency/`。如果服务没有返回 OpenAI `usage`，tok/s 会显示 `N/A`，不会使用不准确的文本或 SSE 块估算。并发80对当前单发服务主要测排队和聚合吞吐，不代表80路同时 decode。

必须记录：

- 请求吞吐与输出 token/s。
- TTFT、TPOT、ITL、E2E 的 p50/p95/p99。
- 成功率、超时率。
- `cpu_moe_wall`。
- CPU、NPU、DDR、NUMA和 page fault监控。

监控命令：

```bash
watch -n 1 npu-smi info
pidstat -u -r -d -p ALL 1
vmstat 1
iostat -xz 1
numastat -p <SERVER_PID>
```

910B早期单流 decode参考约13～16 token/s。提高客户端并发不会让单实例获得同等程度的计算并发；当输出吞吐趋于平坦而 TTFT/E2E持续增加时，说明请求正在排队。

## 11. 两卡聚合

单卡稳定后，在物理卡3和5分别启动独立实例：

| 卡 | 端口 | `KT_CPUINFER` | `KT_THREADPOOL_COUNT` |
|---:|---:|---:|---:|
| 3 | 8020 | 96 | 8 |
| 5 | 8021 | 96 | 8 |

两个实例总工作线程先控制在192，给系统和负载工具保留约64核；不要直接使用两个实例各128线程把256核全部占满。

最大聚合吞吐测试可在两个端口同时运行负载，客户端并发分配为50/50，输出 token/s直接求和。真实入口延迟测试则在两个端口前配置 `least_conn`负载均衡，再对统一入口运行 `bench_serving.sh`。

理想两卡线性值约26～32 token/s；DDR争抢后可能只有约18～29 token/s。该区间是容量规划预估，不是验收保证。计算扩展效率：

```text
双实例聚合输出 TPS / (2 × 单实例输出 TPS)
```

若扩展效率低，优先排查 DDR带宽、远端 NUMA访问、swap、专家缓存复制和 CPU线程超订阅，而不是只看 NPU利用率。

## 12. GPQA-Diamond

先确保服务以非 thinking模式运行，然后执行：

```bash
bash scripts/run_gpqa.sh .env
```

脚本固定：

- GPQA-Diamond，198题。
- API并发1。
- `temperature=1.0`。
- 三轮独立运行。
- 任一错误即中止，不跳过错误样本。

官方910B参考：

| 轮次 | 分数 |
|---|---:|
| R1 | 69.19% |
| R2 | 72.73% |
| R3 | 73.23% |
| 均值 | 71.72% |
| 标准差 | 1.80 pp |

GPQA-Diamond只有198题，单次结果存在明显抽样波动；应比较三轮均值和标准差，不要求逐轮完全相同。每轮必须确认报告样本数为198、无空回复、无超时。

## 13. 停止服务和删除实验容器

模型服务在前台时先按 `Ctrl+C`。回到宿主机，先用精确名称确认目标：

```bash
CONTAINER_NAME=dsv4-npu5  # 按实际容器名修改
docker ps -a --filter "name=^/${CONTAINER_NAME}$"
docker stop --time 60 "$CONTAINER_NAME"
```

停止不会删除容器。需要恢复检查时：

```bash
docker start "$CONTAINER_NAME"
docker exec -it "$CONTAINER_NAME" bash
```

模型服务通常需要进入容器重新执行启动命令。确定不再需要容器后才删除：

```bash
docker stop --time 60 "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME"
docker ps -a --filter "name=^/${CONTAINER_NAME}$"
```

最后一条应无输出。删除容器不会删除绑定挂载的模型、源码、GGUF和日志，也不会删除基础/派生镜像。不要使用模糊名称、通配符或批量删除命令。

## 14. 结果归档

每次实验至少记录：

```text
宿主机、CPU、NUMA和DDR配置
NPU型号、驱动和固件版本
Docker镜像digest和CANN版本
四个源码revision
模型revision和文件hash
所有KT环境变量
输入/输出长度、并发、请求数
TTFT/TPOT/ITL/E2E/TPS
CPU/NPU/NUMA监控日志
GPQA每轮样本数、分数、均值和标准差
```

不要只保存最后一个 TPS 数字；没有版本、输入长度、并发和延迟分位数的结果不可复现。
