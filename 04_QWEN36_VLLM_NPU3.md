# NPU 3 单卡部署：现场操作

模型：`/data/models/Tensor-0.1-Flash-35B-A3B`

镜像：`quay.io/ascend/vllm-ascend:v0.19.1rc1-openeuler`
端口：`9108`；容器：`qwen36-flash-npu3`。

把这五个文件放入模型的 `deployment/`：

```text
01_start.sh
02_check.sh
03_benchmark.sh
04_cleanup.sh
bench_concurrency.py
```

依次执行：

```bash
cd /data/models/Tensor-0.1-Flash-35B-A3B/deployment
chmod +x *.sh bench_concurrency.py

npu-smi info
export NPU3_CONFIRMED_FREE=YES
bash 01_start.sh       # 创建容器并启动服务
bash 02_check.sh       # 等待服务、检查并发送样例
bash 03_benchmark.sh   # 测试 5/10/20/30 并发
bash 04_cleanup.sh     # 停止并删除容器，保留结果
```

结果位于：

```text
/data/models/Tensor-0.1-Flash-35B-A3B/results/qwen36-vllm-npu3/
```

如果还要删除本次日志和压测结果：

```bash
DELETE_RESULTS=YES bash 04_cleanup.sh
```

清理脚本只认本次容器的安全标签，只允许删除上述结果目录；不会删除模型权重、`deployment/` 或其他容器。

常用命令：

```bash
docker logs -f qwen36-flash-npu3
HEALTH_TIMEOUT=3600 bash 02_check.sh
BENCH_REQUESTS=100 BENCH_OUTPUT_TOKENS=512 bash 03_benchmark.sh
```

启动失败时执行：

```bash
docker logs --tail 200 qwen36-flash-npu3
```
