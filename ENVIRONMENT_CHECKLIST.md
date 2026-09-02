# DeepSeek-V4-Flash 910B2 环境检查清单（新手版）

## 1. 先理解“装在哪里”

这套部署分宿主机和容器两层。不要看到宿主机没有 CANN或算子库就直接判定环境不完整。

| 组件 | 宿主机必须安装 | 容器内必须安装 |
|---|---:|---:|
| Ascend驱动 | 是 | 通过目录挂载使用 |
| NPU固件 | 是 | 否 |
| Docker | 是 | 否 |
| CANN Toolkit/Runtime | 否 | 是 |
| CANN OPP基础算子包 | 否 | 是 |
| PyTorch、torch_npu | 否 | 是 |
| gcc/cmake/hwloc等编译工具 | 否 | 构建镜像内需要 |
| patched SGLang/KTransformers | 否 | 是 |
| `kt_kernel_ext`和本方案自定义算子 | 否 | 容器内编译/安装 |
| W8A8/MXFP4/GGUF权重 | 否 | 通过宿主机目录挂载 |

最简单的判断原则：

```text
宿主机负责“让容器能访问910B2”
容器负责“模型框架、CANN、算子和计算环境”
```

## 2. 当前目标设备

```text
CPU: Kunpeng-920，aarch64，256核，8 NUMA
NPU: 2张 Ascend 910B2
物理设备号: 3、5
首个单卡实验: 卡3
第二个聚合实例: 卡5
```

仓库默认配置已经改为：

```bash
NPU_IDS=3,5
```

## 3. 第一步：宿主机检查

在宿主机、容器外执行：

```bash
cd Ds910
cp config/env.example .env
set -a
source .env
set +a
bash scripts/check_host.sh .env
```

如果脚本失败，再逐项执行下面的命令。

### 3.1 操作系统和CPU

```bash
uname -a
uname -m
lscpu
grep -m1 '^Features\|^flags' /proc/cpuinfo
```

预期：

```text
uname -m = aarch64
CPU = Kunpeng-920
CPU online = 256
NUMA = 8
Features包含 asimddp
```

`asimddp`代表 CPU具备本方案 MXFP4 kernel需要的 dot-product/SDOT指令。如果没有，停止部署。

### 3.2 内存和NUMA

```bash
free -h
numactl --hardware
for node in /sys/devices/system/node/node*/meminfo; do
  grep MemFree "$node"
done
swapon --show
```

检查：

- 单实例至少160 GiB可用 DDR，建议256 GiB。
- 两实例建议至少384 GiB，最好512 GiB或更多。
- 8个 NUMA节点都应有可用内存。
- 正式性能测试期间不要依赖 swap。

如果总内存充足但某些 NUMA节点几乎没有空闲内存，需要先检查其他业务进程。

### 3.3 磁盘

```bash
df -h /data
df -i /data
```

需要：

```text
W8A8约275 GiB
原生MXFP4约150 GiB
GGUF约138 GiB
转换期至少预留600 GiB
```

除容量外也要检查 inode，避免磁盘有空间但无法继续创建文件。

### 3.4 驱动、固件和卡状态

```bash
npu-smi info
cat /usr/local/Ascend/driver/version.info 2>/dev/null || true
ls -l /dev/davinci3 /dev/davinci5
ls -l /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
```

通过条件：

- `npu-smi info`正常返回。
- 卡3和卡5没有其他计算进程占用。
- 两卡无告警、温度和功耗正常。
- 五个所需 device node存在。
- 驱动版本被记录到实验报告。

如果宿主机 `npu-smi info`失败，这是驱动/固件问题；不要进容器继续排查算子。

### 3.5 Docker

```bash
docker version
docker info
docker image ls
```

确认已有镜像：

```bash
docker image inspect "$DOCKER_IMAGE" \
  --format 'id={{.Id}} arch={{.Architecture}} os={{.Os}}'
```

必须是 `linux/arm64`。记录 image ID或 digest，不能只记录可变 tag。

## 4. 第二步：检查容器设备挂载

首个实验只向容器暴露物理卡3。使用官方启动脚本时检查它至少传入：

```text
/dev/davinci3
/dev/davinci_manager
/dev/devmm_svm
/dev/hisi_hdc
/usr/local/Ascend/driver（通常只读挂载）
模型目录
源码目录
--ipc=host
足够的memlock/shared memory
```

进入容器后执行：

```bash
ls -l /dev/davinci* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
npu-smi info
```

如果宿主机能看到卡、容器看不到，问题通常是 Docker设备或 driver目录没有挂载，不是模型权重问题。

## 5. 第三步：容器基础环境自动检查

把仓库挂载进容器，然后在容器内执行：

```bash
bash scripts/check_container.sh
```

这个脚本检查：

- Linux/aarch64。
- Python、pip、git、gcc/g++、make、cmake、pkg-config。
- hwloc运行库和开发环境。
- CANN setenv脚本。
- `ASCEND_OPP_PATH`和基础 OPP算子目录。
- torch/torch_npu导入。
- NPU可见性和一次真实 FP16矩阵乘法。
- `kt_kernel_ext`是否已完成构建。
- 自定义算子路径是否已设置。

第一次进入原始镜像时允许出现这两个警告：

```text
kt_kernel_ext 尚不可导入
ASCEND_CUSTOM_OPP_PATH 未设置
```

因为它们可能要在打补丁和编译后才出现。但是服务启动前必须再次执行检查，届时 `kt_kernel_ext`不能继续缺失。

## 6. 手工检查 CANN和基础算子库

容器内执行：

```bash
find /usr/local/Ascend/ascend-toolkit -maxdepth 3 \
  -type f \( -name 'setenv.bash' -o -name 'set_env.sh' \) -print

source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
echo "ASCEND_OPP_PATH=$ASCEND_OPP_PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

test -d "$ASCEND_OPP_PATH" && echo 'OPP directory OK'
find "$ASCEND_OPP_PATH" -maxdepth 2 -type d | head -n 30
```

如何判断：

| 现象 | 结论 |
|---|---|
| 找不到 CANN setenv | 镜像没有完整 CANN Toolkit/Runtime |
| `ASCEND_OPP_PATH`为空 | 环境脚本没加载或镜像缺 OPP包 |
| OPP目录不存在 | 基础算子包不完整 |
| 宿主机没有 OPP、容器内有 | 正常，不需要在宿主机补装 |

## 7. 检查 PyTorch NPU

容器内执行：

```bash
python3 - <<'PY'
import torch
import torch_npu

print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('available:', torch.npu.is_available())
print('device_count:', torch.npu.device_count())

torch.npu.set_device(0)
x = torch.ones((16, 16), dtype=torch.float16, device='npu')
y = x @ x
torch.npu.synchronize()
print('matmul:', y[0, 0].cpu().item())
PY
```

预期最后输出：

```text
available: True
device_count: 1
matmul: 16.0
```

如果 `torch_npu`导入出现 undefined symbol，通常是 torch、torch_npu、CANN或宿主机driver版本不匹配，不能通过随便升级其中一个包解决。

## 8. 检查编译依赖

```bash
python3 --version
pip3 --version
gcc --version
g++ --version
cmake --version
make --version
pkg-config --version
pkg-config --modversion hwloc || true
ldconfig -p | grep hwloc || true
```

至少需要：

```text
git
gcc/g++
make
cmake
pkg-config
hwloc运行库
hwloc开发头文件
Python/pip
```

只有 `ldconfig`能看到 `libhwloc`、但 `pkg-config`找不到 hwloc时，通常说明只装了运行库，没装 `libhwloc-dev`。

## 9. 检查本方案自定义算子

这部分不是普通4B模型能跑就一定存在。DeepSeek单卡 offload还需要 patched SGLang、kt-kernel以及 MXFP4/NPU相关实现。

打补丁后检查：

```bash
git -C "$SOURCE_ROOT/ktransformers" rev-parse HEAD
git -C "$SOURCE_ROOT/sglang" rev-parse HEAD
git -C "$SOURCE_ROOT/llama.cpp" rev-parse HEAD

python3 -c 'import kt_kernel_ext; print(kt_kernel_ext.__file__)'
```

编译日志必须显示：

```text
DOTPROD=ON
SVE=OFF
BF16=OFF
I8MM=OFF
```

然后执行官方预检：

```bash
GGUF_DIR="$GGUF_DIR" bash tools/e2e_preflight.sh
echo $?
```

退出码必须为0。这个预检比仅查看算子目录更可靠，因为它会实际检查 GGUF集合和 `kt_kernel_ext`加载。

常见错误对应关系：

| 报错/现象 | 优先检查 |
|---|---|
| `npu-smi`失败 | 宿主机驱动、固件 |
| 容器无 `/dev/davinci*` | Docker `--device`挂载 |
| `import torch_npu`失败 | 镜像torch/torch_npu/CANN兼容性 |
| `ASCEND_OPP_PATH`不存在 | 镜像缺基础 OPP算子包 |
| `import kt_kernel_ext`失败 | kt-kernel未编译、Python路径或动态库缺失 |
| `llamafile not supported` | 错误启用 SVE/BF16/I8MM |
| undefined symbol | ABI或动态库版本不匹配 |
| 模型能启动但输出乱码 | W8A8和MXFP4版本/量化基底不匹配 |
| NPU利用率低、CPU很忙 | CPU MoE/DDR带宽成为瓶颈 |

## 10. 检查两套模型权重

需要且只需要：

```text
NPU: sgl-npu/DeepSeek-V4-Flash-W8A8
CPU: deepseek-ai/DeepSeek-V4-Flash 原生MXFP4
```

检查目录：

```bash
du -sh "$W8A8_MODEL_PATH" "$MXFP4_SOURCE_PATH"
find "$W8A8_MODEL_PATH" -maxdepth 1 -type f | sort | sed -n '1,20p'
find "$MXFP4_SOURCE_PATH" -maxdepth 1 -type f | sort | sed -n '1,20p'
```

转换后检查43层：

```bash
find "$GGUF_DIR" -maxdepth 1 -name 'dsv4_layer*_mxfp4.gguf' | wc -l
du -sh "$GGUF_DIR"
```

预期数量为43、总大小约138 GiB。数量正确仍不代表内容正确，必须继续执行官方 bit-exact和 CPU kernel对账。

## 11. 检查内网依赖

不要在开始编译后才发现某个URL不通。宿主机或下载机先检查：

```bash
git ls-remote "$CANN_RECIPES_URL" HEAD
git ls-remote "$KTRANSFORMERS_URL" HEAD
git ls-remote "$SGLANG_URL" HEAD
git ls-remote "$LLAMA_CPP_URL" HEAD

curl -I "$HF_ENDPOINT"
curl -I "$MODELSCOPE_ENDPOINT"
curl -I "$PIP_INDEX_URL"
```

还需要确认：

- 内部 Harbor中存在 ARM64镜像。
- 内部 PyPI有 EvalScope、Hugging Face Hub、ModelScope客户端及构建依赖。
- Git镜像包含固定 revision，不只是最新 master。
- 下载完成后推理进程可以启用离线模式。

## 12. Benchmark工具检查

负载机或评测容器：

```bash
vllm bench serve --help >/dev/null
evalscope eval --help >/dev/null
git submodule status benchmarks/gpqa
```

GPQA应固定为：

```text
3481c2b00a22d8256f6b845f256aa4cd4b654132
```

验证样本数：

```bash
python3 - <<'PY'
import csv
with open('benchmarks/gpqa/gpqa_diamond.csv', newline='', encoding='utf-8') as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
```

预期为198。

## 13. 最终绿灯条件

开始下载大模型前：

- [ ] 宿主机 `npu-smi info`正常。
- [ ] 物理卡3和5空闲。
- [ ] 驱动和固件版本已记录。
- [ ] 可用 DDR达到单实例门槛。
- [ ] `/data`至少600 GiB可用。
- [ ] ARM64 Docker镜像存在。
- [ ] 五类内网URL可访问。

开始编译前：

- [ ] 容器内 CANN环境可加载。
- [ ] `ASCEND_OPP_PATH`存在。
- [ ] torch/torch_npu版本已记录。
- [ ] 最小 NPU矩阵乘法通过。
- [ ] gcc/cmake/pkg-config/hwloc齐全。
- [ ] 四个源码 revision准确。

启动模型前：

- [ ] `apply_all.sh`成功。
- [ ] `DOTPROD=ON`，其余三个 ARM扩展为OFF。
- [ ] `kt_kernel_ext`可导入。
- [ ] 43层 GGUF完整。
- [ ] bit-exact和 CPU kernel对账通过。
- [ ] `e2e_preflight.sh`退出码为0。

全部绿灯后，才进入 [DEPLOYMENT.md](DEPLOYMENT.md) 的单卡启动步骤。
