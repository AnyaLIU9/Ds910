# Ds910

DeepSeek-V4-Flash 在 Ascend 910B2 + Kunpeng-920 上进行单卡 NPU/CPU MoE offload、在线性能压测和 GPQA-Diamond 精度评测的最小复现仓库。

仓库不保存模型权重、Docker 镜像、第三方源码或运行结果。完整步骤见 [DEPLOYMENT.md](DEPLOYMENT.md)；完全离线并指定物理 NPU 5 时使用 [DSV4_OFFLINE_NPU5_RUNBOOK.md](DSV4_OFFLINE_NPU5_RUNBOOK.md)；开始前按 [ENVIRONMENT_CHECKLIST.md](ENVIRONMENT_CHECKLIST.md) 验机。需要交给其他对话大模型继续执行时，直接提供 [LLM_HANDOFF.md](LLM_HANDOFF.md)。

```bash
git clone --recurse-submodules <DS910_REPOSITORY_URL> Ds910
cd Ds910
cp config/env.example .env
```

然后修改 `.env` 中的内网 URL 和服务器路径，从验机开始：

```bash
bash scripts/check_host.sh .env
```

服务启动后，可直接测试并发 5、10、80 的聚合 tok/s：

```bash
python3 scripts/bench_concurrency.py \
  --base-url http://127.0.0.1:8020 \
  --model DeepSeek-V4-Flash \
  --concurrency 5 10 80
```
