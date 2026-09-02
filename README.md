# Ds910

DeepSeek-V4-Flash 在 Ascend 910B2 + Kunpeng-920 上进行单卡 NPU/CPU MoE offload、在线性能压测和 GPQA-Diamond 精度评测的最小复现仓库。

仓库不保存模型权重、Docker 镜像、第三方源码或运行结果。完整步骤见 [DEPLOYMENT.md](DEPLOYMENT.md)；需要交给其他对话大模型继续执行时，直接提供 [LLM_HANDOFF.md](LLM_HANDOFF.md)。

```bash
git clone --recurse-submodules git@github.com:AnyaLIU9/Ds910.git
cd Ds910
cp config/env.example .env
```

然后修改 `.env` 中的内网 URL 和服务器路径，从验机开始：

```bash
bash scripts/check_host.sh .env
```
