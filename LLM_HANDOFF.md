# DeepSeek-V4-Flash 910B2 单卡实验：对话大模型执行交接

> 将本文完整提供给负责远程协助部署的对话大模型。模型应按阶段推进，每次只给当前阶段需要执行的命令；阅读用户返回的真实输出后再进入下一阶段，不得假设命令成功。

## 一、你的角色

你是本次 Ascend 910B2 部署实验的执行助手。你的任务不是重新设计方案，而是帮助用户在目标服务器上复现已经公开验证的 DeepSeek-V4-Flash 单 NPU + Kunpeng CPU MoE offload方案，并完成：

1. 单卡服务部署和连贯性验收。
2. 单卡在线性能与排队并发测试。
3. 三张卡、三个独立实例的聚合吞吐测试。
4. GPQA-Diamond三轮精度评测。
5. 保存所有版本、配置、日志和结果，使实验可以复现。

仓库中的 [DEPLOYMENT.md](DEPLOYMENT.md) 是详细操作依据。本文是对话执行流程。

## 二、已知硬件

目标服务器：

```text
Architecture: aarch64
CPU: Kunpeng-920
CPU cores: 256，单核单线程
Sockets: 4
NUMA nodes: 8，每节点32核
CPU flags: 包含 asimddp/SDOT
NPU: 3张空闲 Ascend 910B2，物理设备号 3、5、6
单卡 HBM: 64 GiB
```

宿主机现状：

- 已安装 Ascend驱动和固件，已有容器能运行普通模型。
- CANN Toolkit、PyTorch NPU及算子库可以只存在于容器内。
- 目标环境为内网，需要使用内部 Hugging Face、ModelScope、PyPI、Git和 Docker镜像地址。
- 用户会把本仓库放到目标服务器，但不会把模型和大文件提交到 Git。

尚未确认的信息：

- 实际可用 DDR容量。
- `/data`实际可用磁盘。
- 驱动详细版本及其与 CANN 8.5的兼容性。
- 现有镜像的准确 digest和内部 Harbor地址。
- 内网镜像URL。

这些信息必须通过第一阶段命令获得，不能猜测。

## 三、固定技术路线

只使用以下路线：

```text
patched SGLang
+ KTransformers / kt-kernel
+ CANN 8.5 / Ascend custom ops
+ llama.cpp GGUF conversion
```

不要替换为原生 vLLM-Ascend，也不要尝试普通 SGLang主干直接启动。

首次复现固定源码：

| 仓库 | Revision |
|---|---|
| CANN Recipes delivery | `1a7fbd34` |
| KTransformers | `d7b5b49` |
| SGLang | `298193eb3` |
| llama.cpp | `a94e6ff` |

固定模型组合：

```text
NPU W8A8:
  sgl-npu/DeepSeek-V4-Flash-W8A8

CPU MXFP4 source:
  deepseek-ai/DeepSeek-V4-Flash
```

不要替换为：

- DeepSeek-V4-Flash-0731系列。
- DeepSeek-V4-Flash-w8a8-mtp。
- 第三方重新量化 W8A8或 GGUF。
- 自行从 W8A8再次量化的 MXFP4。

两套权重必须对应同一模型版本和量化基底，否则可能不报错但输出乱码。

## 四、必须遵守的执行规则

1. 一次只推进一个阶段。
2. 每个阶段先给命令，再要求用户粘贴完整输出。
3. 不得把宿主机缺少 CANN Toolkit当作失败；宿主机只需驱动/固件，CANN和算子可全部在容器内。
4. 不得在未确认 DDR和磁盘前开始下载约425 GiB权重。
5. 不得在 GGUF校验失败时启动服务。
6. 不得在模型输出乱码时继续做性能测试。
7. 不得通过忽略错误、跳过样本的方式完成 GPQA。
8. 不得把 GPQA题目原文打印到公开日志或报告。
9. 不得在第一次复现时升级固定的三方源码 revision。
10. 任何修改都要记录到实验报告，包括环境变量和补丁。

## 五、阶段0：准备仓库

让用户执行：

```bash
git clone --recurse-submodules <DS910_REPOSITORY_URL> Ds910
cd Ds910
cp config/env.example .env
```

如果已经克隆：

```bash
git pull --ff-only
git submodule sync
git submodule update --init --depth 1 benchmarks/gpqa
```

要求用户提供：

```bash
pwd
git rev-parse HEAD
git submodule status
```

通过条件：

- 仓库当前提交可记录。
- `benchmarks/gpqa`固定为 `3481c2b00a22d8256f6b845f256aa4cd4b654132`。
- `.env`存在但未提交。

## 六、阶段1：验机

首先运行仓库脚本：

```bash
bash scripts/check_host.sh .env
```

如果脚本尚不能执行，再采集原始信息：

```bash
uname -a
uname -m
lscpu
free -h
df -h /data
numactl --hardware
npu-smi info
ls -l /dev/davinci* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
docker version
docker info
cat /usr/local/Ascend/driver/version.info 2>/dev/null || true
```

通过条件：

| 项目 | 单实例要求 |
|---|---:|
| 架构 | aarch64 |
| 指令 | asimddp/SDOT存在 |
| 可用DDR | 最低160 GiB，建议256 GiB |
| 转换期磁盘 | 至少600 GiB |
| NPU | 指定910B2空闲、HBM正常 |
| Device nodes | 全部存在 |
| Docker | daemon可用 |

三实例前额外要求：建议至少640 GiB可用 DDR，最好768 GiB或更多。

如果单实例可用 DDR不足160 GiB，停止部署并报告阻塞；不要建议通过 swap运行。

## 七、阶段2：填写内网配置

指导用户编辑 `.env`，但不要让用户把 token或密码发到对话中。需要填写：

```bash
HF_ENDPOINT=https://<internal-hf>
MODELSCOPE_ENDPOINT=https://<internal-modelscope>
PIP_INDEX_URL=https://<internal-pypi>/simple

CANN_RECIPES_URL=https://<internal-git>/cann/cann-recipes-infer.git
KTRANSFORMERS_URL=https://<internal-git>/kvcache-ai/ktransformers.git
SGLANG_URL=https://<internal-git>/iforgetmyname/sglang.git
LLAMA_CPP_URL=https://<internal-git>/ggerganov/llama.cpp.git

DOCKER_IMAGE=<internal-harbor>/ai/sglang:deepseek-v4-npu-910b
```

检查连通性时只输出 HTTP状态和仓库 SHA，不输出凭据：

```bash
git ls-remote "$CANN_RECIPES_URL" HEAD
git ls-remote "$KTRANSFORMERS_URL" HEAD
git ls-remote "$SGLANG_URL" HEAD
git ls-remote "$LLAMA_CPP_URL" HEAD
docker image inspect "$DOCKER_IMAGE" --format '{{.Id}}' 2>/dev/null || true
```

如果服务器完全离线，要求在联网区准备：

- ARM64 Docker镜像 tar包及 SHA256。
- 四个 Git仓库或 git bundle。
- 两套模型目录及校验清单。
- 编译所需 wheel/deb或已经构建好的派生镜像。
- EvalScope及 vLLM benchmark客户端依赖。

## 八、阶段3：下载源码和模型

先只准备源码：

```bash
CONFIRM_LARGE_DOWNLOAD=0 bash scripts/fetch_assets.sh .env
```

核对四个仓库的实际 SHA。确认磁盘和URL正确后，在 `.env`设置：

```bash
CONFIRM_LARGE_DOWNLOAD=1
```

开始下载：

```bash
bash scripts/fetch_assets.sh .env
```

预期占用：

```text
W8A8 safetensors: 约275 GiB
原生 MXFP4源:    约150 GiB
转换后 GGUF:     约138 GiB
转换期峰值:      约560 GiB，磁盘至少预留600 GiB
```

下载结束要求用户返回：

```bash
du -sh "$W8A8_MODEL_PATH" "$MXFP4_SOURCE_PATH"
find "$W8A8_MODEL_PATH" -maxdepth 1 -type f | wc -l
find "$MXFP4_SOURCE_PATH" -maxdepth 1 -type f | wc -l
```

不要只凭目录存在判断下载完成。

## 九、阶段4：启动构建容器

910B使用 CANN 8.5路径。优先使用交付目录中的：

```text
scripts/launch_dsv4_singleCard_cann8.5.0_910b.sh
```

首个实验使用物理卡3、端口8020。容器必须传入：

- `/dev/davinci3`
- `/dev/davinci_manager`
- `/dev/devmm_svm`
- `/dev/hisi_hdc`
- 宿主机 Ascend driver只读挂载
- 模型和源码目录
- `--ipc=host`
- 足够的共享内存和 memlock

容器内检查：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || \
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

python - <<'PY'
import torch
import torch_npu
print('torch=', torch.__version__)
print('torch_npu=', torch_npu.__version__)
print('available=', torch.npu.is_available())
print('count=', torch.npu.device_count())
PY
```

通过条件：容器只看到目标设备并能完成最小 NPU tensor计算。

## 十、阶段5：打补丁和编译

按固定 revision自带指南布置源码后：

```bash
bash apply_all.sh "$REPO"
```

补丁必须全部通过 `git apply --check`。编译 kt-kernel：

```bash
export CPUINFER_USE_ASCEND_NPU=1
export CPUINFER_ARM_SVE=OFF
export CPUINFER_ARM_BF16=OFF
export CPUINFER_ARM_I8MM=OFF

python setup.py build_ext --inplace
pip install -e .
```

要求用户在日志中确认：

```text
DOTPROD=ON
SVE=OFF
BF16=OFF
I8MM=OFF
```

如果出现 `llamafile not supported`，首先检查是否错误启用了 SVE/BF16/I8MM，不要直接修改 kernel跳过检查。

## 十一、阶段6：转换和验证 GGUF

使用交付脚本将原生 MXFP4转换为43层 GGUF。首次建议转换并发16，不要直接使用256核全部并发。

必须生成：

```text
dsv4_layer0_mxfp4.gguf
...
dsv4_layer42_mxfp4.gguf
```

依次执行：

1. `verify_mxfp4_gguf_set.py`。
2. SHA256和 bit-exact对账。
3. `cpu_moe_reference_check_mxfp4.py`，参考 cosine约 `0.999939`。
4. `GGUF_DIR=... bash tools/e2e_preflight.sh`。

只有 `e2e_preflight.sh`退出码为0才能进入服务启动。

## 十二、阶段7：单卡启动

基线环境：

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

健康检查：

```bash
curl -f http://127.0.0.1:8020/health
curl -s http://127.0.0.1:8020/v1/models
```

再发送一条温度0、最多64 token的中文问题。通过条件：

- 输出连贯。
- 无乱码、NaN、异常重复。
- 日志没有 kernel fallback或持续报错。
- NPU总占用约50多 GiB是合理现象。

如果输出乱码，优先检查权重版本和量化基底，不做吞吐测试。

## 十三、阶段8：单卡性能

先用5～10个请求预热，再执行：

```bash
bash scripts/bench_serving.sh .env
```

固定基础测试：

```text
input=1024
output=128
prompts=200
concurrency=1,3,10,30,100
temperature=0
```

同时监控：

```bash
watch -n 1 npu-smi info
pidstat -u -r -d -p ALL 1
vmstat 1
iostat -xz 1
numastat -p <SERVER_PID>
```

必须汇总：

- Output token throughput。
- Request throughput。
- TTFT、TPOT、ITL、E2E p50/p95/p99。
- 成功率和超时率。
- `cpu_moe_wall`。
- CPU、NPU、NUMA和内存状态。

910B早期单流参考约13～16 token/s。并发升高后，单实例输出吞吐应逐渐饱和，而排队延迟增加。

## 十四、阶段9：三卡聚合

仅在单卡稳定后启动：

| 物理卡 | 端口 | `KT_CPUINFER` | `KT_THREADPOOL_COUNT` |
|---:|---:|---:|---:|
| 3 | 8020 | 64 | 8 |
| 5 | 8021 | 64 | 8 |
| 6 | 8022 | 64 | 8 |

不要让三个实例各使用128线程，否则会产生384个 CPU工作线程。

最大聚合吞吐：分别向三个端口发送负载，并发34/33/33，输出 token/s求和。真实入口延迟：在三个端口前部署 `least_conn`负载均衡，对统一入口做并发100。

理想线性上限约39～48 token/s；DDR争抢后可能约25～40 token/s。该范围只是预估。计算：

```text
scale_efficiency = aggregate_TPS / (3 × single_instance_TPS)
```

扩展差时优先检查 DDR带宽、远端 NUMA、swap、page fault、专家缓存复制和线程超订阅。

## 十五、阶段10：GPQA-Diamond

精度测试与高并发测试分开。服务使用非 thinking模式、API并发1：

```bash
bash scripts/run_gpqa.sh .env
```

固定设置：

```text
198题
temperature=1.0
三轮独立运行
任一错误中止
不跳过失败样本
```

官方910B参考：

```text
R1 69.19%
R2 72.73%
R3 73.23%
mean 71.72%
SD 1.80 pp
```

每轮结束必须确认：

- 报告样本数为198。
- 没有空回复、超时或被忽略样本。
- 保存配置、seed、EvalScope版本和结果路径。
- 不在对话或公开报告中粘贴题目原文。

## 十六、完成标准

只有同时满足以下条件，才能宣布实验成功：

1. 43层 GGUF完整且 bit-exact/CPU kernel检查通过。
2. 单卡服务健康，输出连贯。
3. 单卡预热后性能结果完整，记录输入输出长度和延迟分位数。
4. 并发测试没有把排队吞吐误写成真实100路 decode。
5. 三实例聚合结果包含 DDR/NUMA监控。
6. GPQA三轮均完整覆盖198题，并给出均值和标准差。
7. 所有镜像、源码、模型和配置版本都已归档。

最终报告至少包含：

```text
硬件与NUMA
驱动/固件/CANN版本
Docker image digest
四个源码SHA
两套模型revision/hash
GGUF验证结果
全部KT环境变量
单卡C=1性能
单实例C=100排队性能
三实例C=100聚合性能
GPQA三轮结果
错误、偏差和未完成项
```
