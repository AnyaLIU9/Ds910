# 单卡 Ascend 910B2：源码运行、W8A8 转换与压测

本文只处理 W8A8，不处理 BF16 服务，也不处理源码包的制作、上传或解压。

固定配置：

```text
CPU              Kunpeng 920（aarch64）
NPU              宿主机物理 NPU 5，单卡 Ascend 910B2
镜像             quay.io/ascend/vllm-ascend:v0.20.2rc1-openeuler
原始模型         /data/models/Tensor
源码             /data/models/Tensor/test
检查和离线包     /data/models/Tensor/check
持久化 venv      /data/models/venv
W8A8 输出         /data/models/Tensor-W8A8
服务容器         tensor-w8a8-npu5
服务端口         9108
```

## 先看执行位置

本文只使用两个容器：

1. `tensor-w8a8-setup`：一次性安装、检查和量化容器。从第 3 节进入，到第 6 节完成后才 `exit`。
2. `tensor-w8a8-npu5`：W8A8 服务容器，在量化完成后创建。

每个标题都明确写了 `[宿主机]` 或 `[setup 容器内]`。看到下一处 `docker run` 之前，不要自行退出当前容器，也不要在容器里再次执行 `docker run`。

Prometheus 使用自己的 Python 启动入口，不是 `vllm serve`。vLLM-Ascend 镜像只提供 CANN、PyTorch 和 `torch_npu`。

## 1. [有外网的电脑] 手工下载两个补充包

检查当前源码的 `requirements-ascend.txt`，并结合服务器上已有包的审计结果，已知只需要手工补下面两个纯 Python wheel：

### prompt_toolkit 3.0.53

- [PyPI 文件页](https://pypi.org/project/prompt-toolkit/#files)
- [wheel 直接下载](https://files.pythonhosted.org/packages/54/6f/84908cad2d6aa5144abcf7b42709fe4fdb459bc640ec7ac5786e7693dabc/prompt_toolkit-3.0.53-py3-none-any.whl)
- 文件名：`prompt_toolkit-3.0.53-py3-none-any.whl`
- SHA256：`01c0891d7f9237d5e339f7d3e42cdae80b7534abb1c7c0e3352efba6231492f2`

### wcwidth 0.2.13

`wcwidth` 是 `prompt_toolkit` 的运行依赖。只下载 `prompt_toolkit` 后出现 `No module named wcwidth`，就是少了这个包。

- [PyPI 文件页](https://pypi.org/project/wcwidth/0.2.13/#files)
- [wheel 直接下载](https://files.pythonhosted.org/packages/fd/84/fd2ba7aafacbad3c4201d395674fc6348826569da3c0937e75505ead3528/wcwidth-0.2.13-py2.py3-none-any.whl)
- 文件名：`wcwidth-0.2.13-py2.py3-none-any.whl`
- SHA256：`3da69048e4540d84af32131829ff948f1e022c1c6bdb8d6102117aac784f6859`

把两个文件都放到：

```text
/data/models/Tensor/check/
```

不要改文件名。命令中的名称是 `prompt_toolkit` 和 `__version__`，不要输入 Markdown 转义后的 `prompt\_toolkit` 或 `**version**`。

## 2. [宿主机] 检查目录、镜像和物理 NPU

服务器目录应为：

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
    ├── 03_w8a8_preflight.sh
    ├── install_offline_env.sh
    ├── prompt_toolkit-3.0.53-py3-none-any.whl
    └── wcwidth-0.2.13-py2.py3-none-any.whl
```

`test` 必须直接包含 `python/`、`scripts/`、`tools/` 和 `requirements-ascend.txt`，不能再套一层源码目录。

执行宿主机检查：

```bash
# 宿主机
bash /data/models/Tensor/check/01_host_preflight.sh
```

该脚本会检查：

- 宿主机为 `aarch64`，镜像为 `arm64`；
- 源码、26 个模型分片、两个 wheel 和安装脚本均存在；
- 两个 wheel 的 SHA256 正确；
- `/data/models` 可用空间和 9108 端口；
- 物理 `/dev/davinci5` 及管理设备节点；
- 最后在宿主机执行 `npu-smi info`。

人工确认 NPU 5 是 910B2，而且没有未知任务大量占用 HBM。`npu-smi` 只在宿主机运行；容器中没有该命令不算错误。

物理卡编号不连续不表示设备 ID 会自动前移；手工挂载 `/dev/davinci5` 时使用设备 ID 5：

```text
宿主机物理 NPU 5 → ASCEND_RT_VISIBLE_DEVICES=5 → Python 进程内 npu:0
```

## 3. [宿主机] 进入唯一的 setup 容器

先删除可能残留的同名容器。此操作不删除镜像，也不删除 `/data/models`：

```bash
# 宿主机
docker rm -f tensor-w8a8-setup 2>/dev/null || true

docker run --rm -it \
  --name tensor-w8a8-setup \
  --network host --ipc host --shm-size 32g \
  --device=/dev/davinci5 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=5 \
  -e SOC_VERSION=ascend910b2 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /data/models:/data/models \
  -w /data/models/Tensor/test \
  quay.io/ascend/vllm-ascend:v0.20.2rc1-openeuler bash
```

命令结束后，终端提示符已经属于容器。接下来的第 4、5、6 节全部在这个容器里执行，不要退出。

不要混用手工 `--device` 和 Ascend Docker Runtime。如果服务器被管理员配置为必须使用 Ascend Runtime，应改用管理员给出的 runtime 参数，而不是同时保留两套映射。

## 4. [setup 容器内] 一条命令建立离线环境

执行：

```bash
# tensor-w8a8-setup 容器内
bash /data/models/Tensor/check/install_offline_env.sh
```

这个脚本会自动完成：

1. 校验两个 wheel；
2. 不存在时创建 `/data/models/venv`，已存在时直接复用；
3. 使用容器 Python 3.11；
4. 从零开始，使用 `--no-index --no-deps` 一次性安装 `prompt_toolkit` 和 `wcwidth`，完全不访问镜像；
5. 检查 `requirements-ascend.txt` 中所有直接依赖的版本；
6. 验证 `prompt_toolkit`、`wcwidth` 和源码导入。

不要再执行 `pip install -r requirements-ascend.txt`，也不要执行 `pip install .`。源码的 `pyproject.toml` 描述的是 CUDA 默认依赖；Ascend 运行脚本会直接设置 `PYTHONPATH`，无需把源码安装进 venv。

脚本内执行的离线安装等价于下面这条命令；这是说明，不需要再重复执行：

```bash
python -m pip install --no-index --no-deps \
  /data/models/Tensor/check/wcwidth-0.2.13-py2.py3-none-any.whl \
  /data/models/Tensor/check/prompt_toolkit-3.0.53-py3-none-any.whl
```

成功时应看到：

```text
missing/mismatch count: 0
[PASS] 持久化环境、Ascend 直接依赖及 prompt_toolkit/wcwidth 导入检查全部通过
```

当前源码声明的直接依赖已经全部纳入检查；结合基础镜像现状，补充清单就是 `prompt_toolkit` 和它的依赖 `wcwidth`。如果脚本仍列出其他 `MISSING` 或 `MISMATCH`，不要继续量化，先按实际输出补对应 wheel。

## 5. [setup 容器内] 检查 Python、NPU 和卡映射

仍在同一个 setup 容器内执行：

```bash
# tensor-w8a8-setup 容器内
bash /data/models/Tensor/check/02_container_preflight.sh
```

脚本本身不会调用 `npu-smi`。它会显示每个检查阶段，并在进程内 `npu:0` 分配 128 MiB、保持 15 秒。看到提示时，在另一个宿主机终端执行：

```bash
# 另一个宿主机终端
npu-smi info
```

新增显存必须出现在物理 NPU 5。若出现在其他卡，停止后续操作。不要在容器里运行 `npu-smi info`。

## 6. [setup 容器内] 转换 W8A8

仍在同一个 setup 容器内执行：

```bash
# tensor-w8a8-setup 容器内
source /data/models/venv/bin/activate
cd /data/models/Tensor/test
export PYTHONPATH=/data/models/Tensor/test/python
mkdir -p /data/models/prometheus-logs /data/models/prometheus-results

python tools/validate_ascend_checkpoint.py \
  /data/models/Tensor \
  --adapter qwen3_6_35b_a3b \
  --slots-per-layer 16

test ! -e /data/models/Tensor-W8A8

SLOTS_PER_LAYER=16 CHUNK_EXPERTS=1 \
  bash scripts/quantize_qwen3_6_35b_a3b_w8a8.sh \
    /data/models/Tensor \
    /data/models/Tensor-W8A8 \
  2>&1 | tee /data/models/prometheus-logs/quantize-w8a8.log

du -sh /data/models/Tensor /data/models/Tensor-W8A8
python -m json.tool /data/models/Tensor-W8A8/config.json \
  | grep -A20 quantization_config
```

`test ! -e /data/models/Tensor-W8A8` 失败说明输出目录已经存在。不要覆盖旧结果；先确认旧目录是否还需要，再决定如何处理。

量化器会复制输入模型目录中的非 safetensors 文件，因此输出目录可能同时出现 `test/` 和 `check/`。这不影响权重加载，但会额外占用磁盘。

量化和检查成功后才退出 setup 容器：

```bash
# tensor-w8a8-setup 容器内
exit
```

setup 容器带 `--rm`，退出后容器会自动删除。模型、venv、源码和日志都位于宿主机 `/data/models`，不会丢失。

如果中途误退出，重新执行第 3 节完整的 `docker run`，然后从未完成的小节继续。不要启动一个缺少 NPU 挂载的简化容器。

## 7. [宿主机] 创建 W8A8 服务容器

现在回到宿主机：

```bash
# 宿主机
docker rm -f tensor-w8a8-npu5 2>/dev/null || true

docker run -d \
  --name tensor-w8a8-npu5 \
  --restart unless-stopped \
  --network host --ipc host --shm-size 32g \
  --device=/dev/davinci5 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=5 \
  -e SOC_VERSION=ascend910b2 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /data/models:/data/models \
  -w /data/models/Tensor/test \
  quay.io/ascend/vllm-ascend:v0.20.2rc1-openeuler sleep infinity

docker exec tensor-w8a8-npu5 bash -lc '
  source /data/models/venv/bin/activate
  export PYTHONPATH=/data/models/Tensor/test/python
  python -c "import torch, torch_npu, prometheus; print(torch.npu.device_count(), torch.npu.get_device_name(0), prometheus.__file__)"
'
```

这里出现第二个 `docker run` 是正常的：setup 容器已经完成并删除，现在创建的是长期服务容器。

## 8. [宿主机] W8A8 预检

```bash
# 宿主机
docker exec -it tensor-w8a8-npu5 \
  bash /data/models/Tensor/check/03_w8a8_preflight.sh
```

检查结果：

```bash
# 宿主机
cat /data/models/prometheus-results/preflight-w8a8/checkpoint.json
cat /data/models/prometheus-results/preflight-w8a8/probe.json
cat /data/models/prometheus-results/preflight-w8a8/moe-operator.json
```

结果应确认：单个可见 910B2、40 层、192 专家、top-k 8、`ascend_w8a8`，并且 `npu_grouped_matmul` 与 `npu_dynamic_quant` 都存在。

## 9. [宿主机] eager 冒烟

```bash
# 宿主机
docker exec -d tensor-w8a8-npu5 bash -lc '
  set -euo pipefail
  source /data/models/venv/bin/activate
  cd /data/models/Tensor/test
  export ASCEND_RT_VISIBLE_DEVICES=5 SOC_VERSION=ascend910b2
  export PROMETHEUS_ASCEND_MOE_KERNEL=eager
  export SLOTS_PER_LAYER=16 MAX_RUNNING_REQUESTS=1
  export MAX_SEQ_LEN=2048 MAX_PREFILL_LENGTH=256 PORT=9108
  bash scripts/run_ascend_910b2_offload.sh /data/models/Tensor-W8A8 \
    >>/data/models/prometheus-logs/w8a8-eager.log 2>&1
'

tail -f /data/models/prometheus-logs/w8a8-eager.log
```

服务就绪后按 `Ctrl+C` 退出 `tail`，不会停止服务。检查：

```bash
# 宿主机
curl --fail --max-time 10 http://127.0.0.1:9108/health
curl -sS http://127.0.0.1:9108/v1/models | python3 -m json.tool

curl --fail --max-time 600 http://127.0.0.1:9108/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"/data/models/Tensor-W8A8",
    "messages":[{"role":"user","content":"请用一句话解释企业现金流预测。"}],
    "temperature":0,
    "max_tokens":64,
    "stream":false
  }'
```

冒烟成功后停止 eager 服务：

```bash
# 宿主机
docker exec tensor-w8a8-npu5 bash -lc \
  'pkill -TERM -f "prometheus.cli serve" || true'
sleep 10
```

## 10. [宿主机] GMM 正式启动

正式压测强制使用 GMM，不允许 `auto` 自动降级：

```bash
# 宿主机
docker exec -d tensor-w8a8-npu5 bash -lc '
  set -euo pipefail
  source /data/models/venv/bin/activate
  cd /data/models/Tensor/test
  export ASCEND_RT_VISIBLE_DEVICES=5 SOC_VERSION=ascend910b2
  export PROMETHEUS_ASCEND_MOE_KERNEL=gmm
  export SLOTS_PER_LAYER=16 MAX_RUNNING_REQUESTS=1
  export MAX_SEQ_LEN=2048 MAX_PREFILL_LENGTH=256 PORT=9108
  bash scripts/run_ascend_910b2_offload.sh /data/models/Tensor-W8A8 \
    >>/data/models/prometheus-logs/w8a8-gmm.log 2>&1
'

tail -f /data/models/prometheus-logs/w8a8-gmm.log
```

服务就绪后按 `Ctrl+C` 退出 `tail`，再重复健康检查和短请求。若 eager 成功但 GMM 失败，保存预检 JSON、`w8a8-eager.log` 和 `w8a8-gmm.log`，先处理 CANN/`torch_npu` GMM 兼容问题，不要继续压测。

## 11. [宿主机] 单请求压测

当前 Ascend 启动器强制 `MAX_RUNNING_REQUESTS=1`，所以只做单流延迟、吞吐和稳定性测试。

假设 Ds910 的压测脚本位于：

```text
/data/models/Ds910/scripts/bench_concurrency.py
```

取得服务实际模型 ID：

```bash
# 宿主机
export SERVED_MODEL_NAME="$(curl -sS http://127.0.0.1:9108/v1/models \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
```

短压测：

```bash
# 宿主机
python3 /data/models/Ds910/scripts/bench_concurrency.py \
  --base-url http://127.0.0.1:9108 \
  --model "$SERVED_MODEL_NAME" \
  --concurrency 1 \
  --requests 10 \
  --max-tokens 128 \
  --timeout 1800 \
  --output-dir /data/models/prometheus-results/w8a8-single
```

稳定性压测：

```bash
# 宿主机
python3 /data/models/Ds910/scripts/bench_concurrency.py \
  --base-url http://127.0.0.1:9108 \
  --model "$SERVED_MODEL_NAME" \
  --concurrency 1 \
  --requests 100 \
  --max-tokens 256 \
  --timeout 1800 \
  --output-dir /data/models/prometheus-results/w8a8-stability
```

压测期间另开宿主机终端观察：

```bash
# 宿主机
watch -n 1 npu-smi info
tail -f /data/models/prometheus-logs/w8a8-gmm.log
```

不要测试并发 2、5、10、20 或 30；当前代码会拒绝 `MAX_RUNNING_REQUESTS>1`。

## 12. [宿主机] 停止和删除容器

只停止服务进程：

```bash
# 宿主机
docker exec tensor-w8a8-npu5 bash -lc \
  'pkill -TERM -f "prometheus.cli serve" || true'
```

停止容器：

```bash
# 宿主机
docker stop tensor-w8a8-npu5
```

彻底删除容器：

```bash
# 宿主机
docker rm -f tensor-w8a8-npu5
```

以上命令不会删除镜像，也不会删除绑定挂载的 `/data/models`。若要重新启动已停止但未删除的容器，执行 `docker start tensor-w8a8-npu5`。

## 常见错误

- `No module named prompt_toolkit` 或 `No module named wcwidth`：确认两个 wheel 都在 `check` 目录，然后在 setup 容器中重新运行 `install_offline_env.sh`；脚本会同时安装两者。
- `No module named prometheus`：确认使用最新版检查脚本；运行源码时必须设置 `PYTHONPATH=/data/models/Tensor/test/python`。
- 容器里没有 `npu-smi`：正常；只在宿主机检查物理卡。
- `torch.npu.device_count()` 不是 1：卡隔离失败，不要加载模型。
- 显存不在物理 NPU 5：设备映射不符合预期，立即停止。
- MindStudio、`mindstudio-kpp` 或 `plotly` 缺失：这是基础镜像中无关工具的依赖状态，不属于本项目依赖，不需要补装；最新版安装脚本不再执行全局 `pip check`。
- 服务 OOM：先将 `MAX_SEQ_LEN` 从 2048 降到 1024、`MAX_PREFILL_LENGTH` 从 256 降到 128，再将 `SLOTS_PER_LAYER` 从 16 降到 12 或 8；不得低于 top-k 8。
