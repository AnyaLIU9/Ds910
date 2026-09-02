# DeepSeek-V4-Flash 单卡实验：下载清单

这份文件只回答一件事：**联网电脑需要下载什么**。服务器上的操作看 `02_EXPERIMENT.md`。

固定路线：Atlas 910B2 + Kunpeng-920 + CANN 8.5 镜像 + SGLang/KTransformers CPU MoE offload。

## 1. Docker 基础镜像

镜像名：

```text
lmsysorg/sglang:deepseek-v4-npu-910b
```

- Docker Hub：https://hub.docker.com/r/lmsysorg/sglang/tags
- 已上传的 ARM64 备份：https://huggingface.co/datasets/Lyyyy1818/haha/tree/main
- 备份文件：`deepseek-v4-npu-910b-arm64.tar`
- SHA-256：`cfdb04294636003f6638425df999b4c13b89079356ca6d8f8618abec07c0bbdf`

如果重新从 Docker Hub 制作备份，必须明确拉 ARM64：

```bash
docker pull --platform linux/arm64 lmsysorg/sglang:deepseek-v4-npu-910b
docker save -o deepseek-v4-npu-910b-arm64.tar \
  lmsysorg/sglang:deepseek-v4-npu-910b
shasum -a 256 deepseek-v4-npu-910b-arm64.tar
```

把这个 tar 上传到：

```text
/home/mem/dsv4/packages/deepseek-v4-npu-910b-arm64.tar
```

服务器实验流程会先校验它，再执行 `docker load`。导入后镜像进入 Docker 自己的存储；tar 可以继续留在 `packages/` 作为离线备份。

## 2. 两份模型权重

只用下面两份，不用 0731-W8A8，也不用 MTP 版。

| 用途 | 固定模型 | 下载页 |
|---|---|---|
| NPU attention、router、shared expert 和热专家 | `DeepSeek-V4-Flash-W8A8` | https://modelscope.cn/models/sgl-npu/DeepSeek-V4-Flash-W8A8 |
| CPU 专家转换源，原生 MXFP4 | `DeepSeek-V4-Flash` | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash |

下载完成后目录名保持为：

```text
DeepSeek-V4-Flash-W8A8
DeepSeek-V4-Flash
```

## 3. 固定版本代码

| 代码 | 固定提交 | 下载链接 |
|---|---|---|
| CANN Recipes 交付仓 | `1a7fbd348f4d8be4aecdb369d4b0fe89d433f5a5` | [仓库内固定 ZIP](./cann-recipes-infer-1a7fbd34.zip) |
| KTransformers | `d7b5b49a3ef214a822aba613423551dd56416557` | https://github.com/kvcache-ai/ktransformers/archive/d7b5b49a3ef214a822aba613423551dd56416557.zip |
| SGLang fork | `298193eb34c9d87debbdb5957edead0a8b9ec988` | https://github.com/iforgetmyname/sglang/archive/298193eb34c9d87debbdb5957edead0a8b9ec988.zip |
| llama.cpp | `a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f` | https://github.com/ggml-org/llama.cpp/archive/a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f.zip |
| pybind11 | `bb05e0810b87e74709d9f4c4545f1f57a1b386f5` | https://github.com/pybind/pybind11/archive/bb05e0810b87e74709d9f4c4545f1f57a1b386f5.zip |
| custom_flashinfer | `fd94393fb5b8ba8bae9c0bd6ab1c2a429d81ac76` | https://github.com/kvcache-ai/custom_flashinfer/archive/fd94393fb5b8ba8bae9c0bd6ab1c2a429d81ac76.zip |

CANN Recipes ZIP 的校验文件在：[cann-recipes-infer-1a7fbd34.zip.sha256](./cann-recipes-infer-1a7fbd34.zip.sha256)。

KTransformers ZIP 不带子模块内容，所以它里面的 `third_party/` 出现空目录是正常的。SGLang、llama.cpp、pybind11 和 custom_flashinfer 必须分别下载，最终摆放方式见 `02_EXPERIMENT.md`。

## 4. `hwloc` 离线依赖

这些 deb **不是安装到 Mac 本机，也不是安装到测试服务器宿主机**。Mac 只负责暂存和上传文件。

联网 Mac 上已经准备好这个目录：

```text
/Users/lyy/Downloads/dsv4-hwloc-arm64-debs/
```

里面应有 17 个 ARM64 Ubuntu 22.04 `.deb` 文件。把这 17 个文件原样上传到测试服务器：

```text
/home/mem/dsv4/image/debs/
```

后续在测试服务器执行 `docker build` 时，`image/Dockerfile` 才会把它们安装进 `dsv4-offload-env:cann85-910b2` 派生镜像。不要在 Mac 或服务器宿主机执行 `dpkg -i`，也不要单独下载一个 `libhwloc.so`。

```text
Mac 暂存目录
  → 上传 17 个 deb
测试服务器 /home/mem/dsv4/image/debs/
  → docker build
派生 Docker 镜像里的 Ubuntu 环境
```

## 5. GPQA-Diamond（模型跑通后再准备）

- EvalScope：https://pypi.org/project/evalscope/
- EvalScope 文档：https://evalscope.readthedocs.io/
- GPQA 官方数据：https://huggingface.co/datasets/Idavidrein/gpqa

GPQA 是 gated dataset。必须先登录 Hugging Face、接受数据集条款，再在联网环境下载。内网运行还需要提前准备 EvalScope 的 ARM64/Python 3.11 wheels 及数据缓存。

## 不要下载

这条复现路线不使用：`DeepSeek-V4-Flash-0731-W8A8`、`DeepSeek-V4-Flash-W8A8-MTP`、vLLM-Ascend、CUDA、最新版 SGLang 或最新版 KTransformers。
