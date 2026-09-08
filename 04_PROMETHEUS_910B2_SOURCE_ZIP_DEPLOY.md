# Prometheus：单卡 910B2 源码包部署 W8A8

本文只处理 W8A8 转换、启动和压测，不测试 BF16 服务。源码已经部署完成，本文只检查服务器上的目录是否正确，不处理打包、上传或解压。

固定环境：

```text
CPU：Kunpeng 920（aarch64）
NPU：Ascend 910B2，使用宿主机物理 NPU 5
镜像：quay.io/ascend/vllm-ascend:v0.20.2rc1-openeuler
量化输入：/data/models/Tensor
源码目录：/data/models/Tensor/test
检查脚本：/data/models/Tensor/check
持久化环境：/data/models/venv
W8A8 输出：/data/models/Tensor-W8A8
端口：9108
```

Prometheus 使用自己的 `prom serve`，不是 `vllm serve`。现有 vLLM-Ascend 镜像只作为 CANN、PyTorch 和 torch_npu 基础环境。

仓库生成的是路由专家 W8A8 Dynamic：路由专家权重为 INT8 per-channel，激活为 INT8 per-token；attention、GDN、router 和 shared expert 保持 BF16。它不是 MindIE/msModelSlim 全模型 W8A8。

## 1. 检查服务器目录

确认目录结构：

```text
/data/models/Tensor/
├── config.json
├── model.safetensors.index.json
├── model-00001-of-00026.safetensors ... model-00026-of-00026.safetensors
├── test/
│   ├── python/
│   ├── scripts/
│   ├── tools/
│   └── requirements-ascend.txt
└── check/
    ├── 01_host_preflight.sh
    ├── 02_container_preflight.sh
    └── 03_w8a8_preflight.sh
```

`/data/models/Tensor/test` 必须直接包含 `python/`、`scripts/`、`tools/` 和 `requirements-ascend.txt`，不能再多套一层目录。源码目录位于模型目录内不影响 checkpoint 读取。

## 2. 宿主机预检查

```bash
bash /data/models/Tensor/check/01_host_preflight.sh
```

确认宿主机和镜像分别是 `aarch64`、`arm64`，物理 NPU 5 是 910B2 且没有其他任务占用大量 HBM，9108 端口空闲。转换前建议 `/data/models` 至少有 80 GiB 可用空间。

`npu-smi info` 只在物理机执行。openEuler 容器不要求存在 `npu-smi`，也不把宿主机的 `npu-smi` 可执行文件挂进容器。容器内使用 `torch.npu.is_available()`、`torch.npu.device_count()`、短显存分配和必需 TorchNPU 算子检查来验收运行环境。

这台服务器缺少物理设备 4。采用手工设备节点挂载时：

```text
宿主机物理设备：0 1 2 3 5 6 ...
容器 CANN 逻辑：0 1 2 3 4 5 ...
物理 NPU 5 → 容器逻辑 4 → 进程内 npu:0
```

## 3. 创建一次性环境安装容器

如果之前按旧版文档创建过容器，先清理旧容器名；带 `--rm` 的安装容器在 `exit` 后通常已经自动消失：

```bash
docker ps -a --filter 'name=^/prometheus-b2-setup$' \
  --filter 'name=^/prometheus-b2-npu5$'
docker rm -f prometheus-b2-setup prometheus-b2-npu5 2>/dev/null || true
```

这只删除 Docker 容器，不删除镜像，也不删除宿主机 `/data/models` 中的绑定挂载数据。

可在启动安装容器前设置自己的 index URL；不设置则使用华为云公网镜像：

```bash
export PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple}"
export PYPI_TRUSTED_HOST="${PYPI_TRUSTED_HOST:-mirrors.huaweicloud.com}"
```

```bash
docker run --rm -it \
  --name tensor-w8a8-setup \
  --network host --ipc host --shm-size 32g \
  --device=/dev/davinci5 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=4 \
  -e SOC_VERSION=ascend910b2 \
  -e PYPI_INDEX_URL \
  -e PYPI_TRUSTED_HOST \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /data/models:/data/models \
  -w /data/models/Tensor/test \
  quay.io/ascend/vllm-ascend:v0.20.2rc1-openeuler bash
```

不要混用手工 `--device` 和 Ascend Docker Runtime。如果服务器必须使用 Ascend Runtime，则改为 `--runtime=ascend -e ASCEND_VISIBLE_DEVICES=5`，移除四个 `--device` 和 `ASCEND_RT_VISIBLE_DEVICES=4`，进入容器后以设备检测结果为准。

## 4. 在 `/data/models/venv` 创建持久化环境

仍在安装容器内执行：

```bash
cd /data/models/Tensor/test
python -m venv --system-site-packages /data/models/venv
source /data/models/venv/bin/activate

export PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple}"
export PYPI_TRUSTED_HOST="${PYPI_TRUSTED_HOST:-mirrors.huaweicloud.com}"
export PIP_INDEX_URL="$PYPI_INDEX_URL"
export PIP_TRUSTED_HOST="$PYPI_TRUSTED_HOST"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-10}"
unset PIP_EXTRA_INDEX_URL

python -m pip config --site set global.index-url "$PIP_INDEX_URL"
python -m pip config --site set global.trusted-host "$PIP_TRUSTED_HOST"
python -m pip config --site set global.timeout "$PIP_DEFAULT_TIMEOUT"
python -m pip config --site set global.retries "$PIP_RETRIES"
python -m pip config --site unset global.extra-index-url 2>/dev/null || true
python -m pip config list

python -m pip install --no-cache-dir 'setuptools>=77' wheel
python -m pip install --no-cache-dir -r requirements-ascend.txt
python -m pip install --no-cache-dir . --no-build-isolation --no-deps

python - <<'PY'
import torch, torch_npu, prometheus
print(torch.__version__, torch_npu.__version__, prometheus.__file__)
assert torch.npu.is_available() and torch.npu.device_count() == 1
PY
```

安装日志里的下载地址必须全部来自你指定的 `PYPI_INDEX_URL`。如果依赖在该镜像中不存在或暂未同步，停止并记录缺失的包和版本；不要临时增加其他 `extra-index-url`。

可以在创建容器前自行选择允许访问的 index URL。例如华为云内网 PyPI：

```bash
export PYPI_INDEX_URL=http://mirrors.myhuaweicloud.com/pypi/web/simple
export PYPI_TRUSTED_HOST=mirrors.myhuaweicloud.com
```

也可以设置为单位提供的华为制品代理地址。`PYPI_TRUSTED_HOST` 应填写 URL 中的主机名；后续检查脚本使用同名变量核对配置。

不要执行 `scripts/build-release-wheels.sh`，它包含 CUDA release/kernel-cache 构建逻辑。`requirements-ascend.txt` 刻意不安装 torch、torch_npu、vLLM、Triton、flashlib 或 CUDA 包，这些组件必须继续使用基础镜像中已经匹配好的版本。

安装完成后，在容器内执行统一环境和卡映射检查：

```bash
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple}" \
PYPI_TRUSTED_HOST="${PYPI_TRUSTED_HOST:-mirrors.huaweicloud.com}" \
  bash /data/models/Tensor/check/02_container_preflight.sh
```

脚本会保持 128 MiB NPU 分配 15 秒。与此同时必须另开物理机终端执行 `npu-smi info`，确认新增显存位于物理 NPU 5。不要在容器内执行 `npu-smi`。若显存出现在 NPU 6，立即停止。进程内部始终使用 `npu:0`。

## 5. 转换 W8A8

只校验量化输入格式，不启动 BF16 服务：

```bash
PYTHONPATH=python python tools/validate_ascend_checkpoint.py \
  /data/models/Tensor \
  --adapter qwen3_6_35b_a3b \
  --slots-per-layer 16

test ! -e /data/models/Tensor-W8A8
```

转换并保存日志：

```bash
SLOTS_PER_LAYER=16 CHUNK_EXPERTS=1 \
  bash scripts/quantize_qwen3_6_35b_a3b_w8a8.sh \
    /data/models/Tensor \
    /data/models/Tensor-W8A8 \
  2>&1 | tee /data/models/prometheus-logs/quantize-w8a8.log

du -sh /data/models/Tensor /data/models/Tensor-W8A8
python -m json.tool /data/models/Tensor-W8A8/config.json \
  | grep -A20 quantization_config
```

脚本会在转换后再次校验 W8A8。不要删除 `/data/models/Tensor`，至少保留到 W8A8 验收完成。转换完成后执行 `exit` 退出安装容器；源码、环境、模型和日志都在宿主机 `/data/models`，不会随 `--rm` 丢失。

当前量化器会复制输入模型目录中的非 safetensors 文件和子目录。因此源码位于 `/data/models/Tensor/test` 时，转换结果中也会出现 `/data/models/Tensor-W8A8/test`。这不影响按索引加载 W8A8 权重，但会额外占用磁盘；转换前后用下面命令记录大小：

```bash
du -sh \
  /data/models/Tensor/test \
  /data/models/Tensor-W8A8/test 2>/dev/null || true
```

本文不自动删除复制出来的目录。若需要避免这份复制，应先修改量化器的非权重文件复制白名单并重新测试，而不是在部署过程中临时移动源码。

## 6. 创建长期运行容器

宿主机执行：

```bash
docker run -d \
  --name tensor-w8a8-npu5 \
  --restart unless-stopped \
  --network host --ipc host --shm-size 32g \
  --device=/dev/davinci5 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=4 \
  -e SOC_VERSION=ascend910b2 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /data/models:/data/models \
  -w /data/models/Tensor/test \
  quay.io/ascend/vllm-ascend:v0.20.2rc1-openeuler sleep infinity

docker exec tensor-w8a8-npu5 bash -lc '
  source /data/models/venv/bin/activate
  python -c "import torch, torch_npu; print(torch.npu.device_count(), torch.npu.get_device_name(0))"
'
```

## 7. W8A8 预检

```bash
docker exec -it tensor-w8a8-npu5 \
  bash /data/models/Tensor/check/03_w8a8_preflight.sh
```

这个脚本不会调用源码中的 `scripts/preflight_ascend.sh`，因为后者当前强制要求运行环境中存在 `npu-smi`。检查脚本会直接执行 checkpoint 校验、TorchNPU 环境检查、H2D 探测和 W8A8 必需算子检查；物理卡状态继续由第 2 节的宿主机脚本负责。

检查：

```bash
cat /data/models/prometheus-results/preflight-w8a8/checkpoint.json
cat /data/models/prometheus-results/preflight-w8a8/probe.json
cat /data/models/prometheus-results/preflight-w8a8/moe-operator.json
```

应显示 910B2、单个可见设备、40 层、192 专家、top-k 8 和 `ascend_w8a8`。`probe.json` 和 `moe-operator.json` 还必须确认 `npu_grouped_matmul` 与 `npu_dynamic_quant` 都存在；最新启动脚本会把缺少任一算子视为硬错误。

## 8. W8A8 eager 冒烟

```bash
docker exec -d tensor-w8a8-npu5 bash -lc '
  set -euo pipefail
  source /data/models/venv/bin/activate
  cd /data/models/Tensor/test
  export ASCEND_RT_VISIBLE_DEVICES=4 SOC_VERSION=ascend910b2
  export PROMETHEUS_ASCEND_MOE_KERNEL=eager
  export SLOTS_PER_LAYER=16 MAX_RUNNING_REQUESTS=1
  export MAX_SEQ_LEN=2048 MAX_PREFILL_LENGTH=256 PORT=9108
  bash scripts/run_ascend_910b2_offload.sh /data/models/Tensor-W8A8 \
    >>/data/models/prometheus-logs/w8a8-eager.log 2>&1
'

tail -f /data/models/prometheus-logs/w8a8-eager.log
```

另一个终端执行：

```bash
curl --fail --max-time 10 http://127.0.0.1:9108/health
curl -sS http://127.0.0.1:9108/v1/models | python3 -m json.tool
```

把 `model` 替换成 `/v1/models` 返回的实际 `id`：

```bash
curl --fail --max-time 600 http://127.0.0.1:9108/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"/data/models/Tensor-W8A8",
    "messages":[{"role":"user","content":"请用一句话解释企业现金流预测。"}],
    "temperature":0,
    "max_tokens":64,
    "stream":false
  }'

docker exec tensor-w8a8-npu5 bash -lc \
  'pkill -TERM -f "prometheus.cli serve" || true'
sleep 10
```

## 9. W8A8 GMM 正式启动

正式压测强制使用 GMM，不允许 `auto` 悄悄降级：

```bash
docker exec -d tensor-w8a8-npu5 bash -lc '
  set -euo pipefail
  source /data/models/venv/bin/activate
  cd /data/models/Tensor/test
  export ASCEND_RT_VISIBLE_DEVICES=4 SOC_VERSION=ascend910b2
  export PROMETHEUS_ASCEND_MOE_KERNEL=gmm
  export SLOTS_PER_LAYER=16 MAX_RUNNING_REQUESTS=1
  export MAX_SEQ_LEN=2048 MAX_PREFILL_LENGTH=256 PORT=9108
  bash scripts/run_ascend_910b2_offload.sh /data/models/Tensor-W8A8 \
    >>/data/models/prometheus-logs/w8a8-gmm.log 2>&1
'

tail -f /data/models/prometheus-logs/w8a8-gmm.log
```

重复健康检查和短请求。若 eager 成功但 GMM 失败，保留 `moe-operator.json`、`w8a8-eager.log` 和 `w8a8-gmm.log`，先处理 CANN/torch_npu grouped-matmul 兼容问题，不要继续压测。

## 10. 单请求服务压测

复用本仓脚本：

```text
/data/models/Ds910/scripts/bench_concurrency.py
```

若服务器没有整个 `Ds910` 目录，把该脚本放到 `/data/models/Tensor/test/bench_concurrency.py` 并修改下方路径。

取得实际模型 ID：

```bash
export SERVED_MODEL_NAME=$(curl -sS http://127.0.0.1:9108/v1/models \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
```

最新的 `run_ascend_910b_offload.sh` 强制要求 `MAX_RUNNING_REQUESTS=1`；设置为 2 或更高会在启动阶段直接退出。因此这里只测试单流延迟、吞吐和长时间稳定性，不宣称多并发能力。

短压测：

```bash
python3 /data/models/Ds910/scripts/bench_concurrency.py \
  --base-url http://127.0.0.1:9108 \
  --model "$SERVED_MODEL_NAME" \
  --concurrency 1 \
  --requests 10 \
  --max-tokens 128 \
  --timeout 1800 \
  --output-dir /data/models/prometheus-results/w8a8-single
```

稳定性压测仍保持并发 1，只增加请求数量和输出长度：

```bash
python3 /data/models/Ds910/scripts/bench_concurrency.py \
  --base-url http://127.0.0.1:9108 \
  --model "$SERVED_MODEL_NAME" \
  --concurrency 1 \
  --requests 100 \
  --max-tokens 256 \
  --timeout 1800 \
  --output-dir /data/models/prometheus-results/w8a8-concurrency
```

当前 GDN、SDPA、专家拷贝/计算重叠还未完成 910B2 生产级优化，且启动器主动禁止多请求。不要使用并发 `2 5 10 20 30`。压测期间运行：

```bash
watch -n 1 npu-smi info
tail -f /data/models/prometheus-logs/w8a8-gmm.log
```

至少保存 TTFT、TPOT、输出 token/s、请求吞吐、P50/P95/P99、成功率、NPU HBM 峰值和 CPU 内存峰值。

## 11. 常见问题

- 显存出现在 NPU 6：手工挂载模式应为物理 `/dev/davinci5`、`ASCEND_RT_VISIBLE_DEVICES=4`、进程内 `npu:0`。
- `torch.npu.device_count()` 不是 1：卡隔离未生效，不要加载模型。
- W8A8 eager 成功而 GMM 失败：当前 CANN/torch_npu GMM 接口与代码预期不匹配。
- 服务 OOM：先确认物理卡正确，再把 `MAX_SEQ_LEN` 从 2048 降到 1024、`MAX_PREFILL_LENGTH` 从 256 降到 128、`SLOTS_PER_LAYER` 从 16 降到 12 或 8；槽数不能低于 top-k 8。
- 安装后 torch_npu 失效：pip 覆盖了镜像匹配好的 torch。重建 `/data/models/venv`，只安装 `requirements-ascend.txt`，源码使用 `--no-deps`。

停止服务或容器：

```bash
docker exec tensor-w8a8-npu5 bash -lc \
  'pkill -TERM -f "prometheus.cli serve" || true'
docker stop tensor-w8a8-npu5
```
