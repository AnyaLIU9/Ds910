# DeepSeek-V4-Flash：`/home/mem` 最终目录与 NPU 5 单卡部署 Runbook

本文是本次实验的主操作说明。后续按本文执行，不再混用 A3、0731、vLLM-Ascend 或其他模型的部署命令。

## 0. 本次实验范围

- 服务器：aarch64，Kunpeng-920，256 CPU 核，8 NUMA
- NPU：Ascend 910B2
- 第一阶段只使用物理 NPU 5
- 宿主机只需要驱动、固件和 Docker
- 容器使用 CANN 8.5.0 的 DSV4 指定镜像
- 推理框架：打过交付补丁的 SGLang + KTransformers/kt-kernel
- NPU 权重：固定版官方 W8A8
- CPU 权重：固定版官方原生 MXFP4，转换为 43 个 GGUF
- 第一阶段只测单请求，不测 100 并发

---

## 0.1 Docker 版本和存储要求

原始交付说明没有固定 Docker 的精确版本。当前服务器是 **Docker 19.03.10**。这不是当前实验的阻塞项，可以先直接使用，不必为了版本号立刻升级：

- 当前准备的 `deepseek-v4-npu-910b-arm64.tar` 已检查过，里面既有传统 Docker `manifest.json`，也有 OCI 索引；它不是只有新版 Docker 才认识的纯 OCI 归档。
- Docker 19.03 已支持本实验要用的 `docker load`、`--device`、目录挂载、`--network host`、`--ipc` 和 `--shm-size`。
- 因此先以“能否成功导入镜像并映射 NPU”为准，不因 19.03.10 这个版本号直接中止实验。
- Docker 19.03 已经很老；如果以后把服务长期对外提供，应安排升级。但这和本次离线复现实验能否启动是两件事。
- 推荐使用普通 rootful Docker，不使用 rootless Docker。
- 不需要 NVIDIA Container Runtime。
- 交付脚本通过 `--device` 映射 Ascend 设备，因此必须能够映射 `/dev/davinci5`、`/dev/davinci_manager`、`/dev/devmm_svm` 和 `/dev/hisi_hdc`。

服务器检查：

```bash
docker version
```

```bash
docker info --format 'root={{.DockerRootDir}} driver={{.Driver}} security={{json .SecurityOptions}}'
```

重点确认输出里没有 `rootless`。

Docker 镜像不会存进 `/home/mem/dsv4`。执行 `docker load` 后，解包数据默认写入 Docker Root Dir，通常是：

```text
/var/lib/docker
```

指定基础镜像在 Docker 中展开约占 20 GB，派生层还会增加少量空间。导入前检查 Docker Root Dir 所在磁盘：

```bash
export DSV4_DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}')
echo "$DSV4_DOCKER_ROOT"
df -h "$DSV4_DOCKER_ROOT"
```

建议至少保留 30 GiB 可用空间。模型权重、GGUF 和日志仍放 `/home/mem/dsv4`，不进入 Docker Root Dir。

如果只有 `/home` 有空间，而 Docker Root Dir 所在分区不足 30 GiB，先停止。需要让服务器管理员评估把 Docker `data-root` 改到 `/home/mem/docker-data`。修改 `data-root` 会影响机器上的所有现有镜像和容器，不要在共享服务器上自行修改或移动 `/var/lib/docker`。

版本判断：

| Docker Server 版本 | 处理 |
|---|---|
| 20.10 或更高 | 可以继续 |
| **19.03.10（当前服务器）** | **先继续；只有实际导入或设备映射失败时才处理版本问题** |
| 低于 19.03 | 先升级 Docker |
| rootless | 不建议映射 Ascend 设备，换 rootful Docker |

### 0.1.1 当前服务器先执行的 Docker 验证

注意：`docker version` 会同时显示 Client 和 Server。真正决定容器能力的是 **Server 端的 19.03.10**。

先看版本和 Docker 数据盘空间：

```bash
docker version
docker info --format 'root={{.DockerRootDir}} driver={{.Driver}} security={{json .SecurityOptions}}'
export DSV4_DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}')
df -h "$DSV4_DOCKER_ROOT"
```

再校验并导入已经上传到服务器的 tar：

```bash
cd /home/mem/dsv4/packages
sha256sum -c deepseek-v4-npu-910b-arm64.tar.sha256
docker load -i deepseek-v4-npu-910b-arm64.tar
docker image inspect lmsysorg/sglang:deepseek-v4-npu-910b \
  --format 'arch={{.Architecture}} id={{.Id}}'
```

正常结果应满足：

- SHA-256 显示 `OK`；
- `docker load` 成功并出现镜像名；
- inspect 输出包含 `arch=arm64`。

满足这三项，就继续后面的派生镜像和 NPU 5 映射步骤，不需要升级 Docker。

如果 `docker load` 明确报 `unsupported manifest`、`unsupported media type` 或归档格式不支持，再采用下面任一方案：

1. 让管理员把 Docker Engine 升级到 20.10 或更高版本；
2. 在同为 ARM64、Docker 19.03 的联网环境重新 `docker pull`、`docker save` 后再传入。

不要在尚未执行 `docker load` 前，仅凭版本较旧就重装 Docker。

Docker 版本和 CANN 版本不是一回事：Docker 负责容器和设备映射；CANN 8.5.0、torch_npu、自定义算子在镜像内部。宿主机还需要与 CANN 8.5.0 路径兼容的 Ascend 驱动和固件。

---

## 1. 你现在还缺什么

| 内容 | 当前状态 | 是否必须 |
|---|---|---|
| `deepseek-v4-npu-910b` ARM64 Docker tar | 已上传到 `/home/mem/dsv4/packages/` | 必须 |
| `hwloc` Ubuntu 22.04 ARM64 离线 deb 包 | 本说明已在本机生成，待上传 | 必须 |
| `DeepSeek-V4-Flash-W8A8` | 已下载或正在完成 | 必须 |
| 原版 `DeepSeek-V4-Flash` MXFP4 | 已下载 | 必须 |
| ktransformers `d7b5b49` | 已下载 | 必须 |
| sglang `298193eb3` | 已下载 | 必须 |
| llama.cpp `a94e6ff` | 已下载 | 必须 |
| 完整 `cann-recipes-infer` 交付目录 | **需要确认/下载** | 必须 |
| pybind11 `bb05e081...` | **需要确认/下载** | 必须 |
| custom_flashinfer `fd94393f...` | **需要确认/下载** | 必须 |
| evalscope | 暂时不用下载 | GPQA 时才需要 |

### 1.1 必须补齐的下载地址

#### CANN Recipes 完整交付仓

- 仓库：https://gitcode.com/cann/cann-recipes-infer
- 当前官方仓库页面：https://gitcode.com/cann/cann-recipes-infer/tree/master
- 对应 DSV4 完整功能合并记录：https://gitcode.com/cann/cann-recipes-infer/pull/682
- 本实验把 `1a7fbd34` 作为整个 `cann-recipes-infer` 交付仓的固定版本，不要直接长期跟随会继续变化的 `master`。
- 设计文档位于 `docs/integration/sglang/dsv4-flash-single-npu-moe-offload/`，只用于阅读；真正要执行的补丁、转换器和启动脚本位于不带 `docs` 的 `integration/sglang/dsv4-flash-single-npu-moe-offload/`。
- 本实验必须存在的目录：

```text
integration/sglang/dsv4-flash-single-npu-moe-offload/
```

该目录必须包含：

```text
apply_all.sh
main_repo/*.patch
sglang/*.patch
llama_cpp/*.patch
scripts/launch_dsv4_singleCard_cann8.5.0_910b.sh
scripts/tools/*
```

只有 `dsv4_flash_single_card_inference_guide.md` 不够，必须下载完整交付目录。

不要把这里的 `1a7fbd34` 和设计文档中出现的 `c5cc95e` 混为一谈：

- `1a7fbd34`：本实验要保存的 **完整 cann-recipes-infer 交付仓版本**，910B2 路线使用它。
- `c5cc95e`：A3/910C 从干净 CANN 9.0 镜像构建第三方融合算子时，由环境脚本钉住的一个算子源码版本；它不是本实验顶层交付仓应 checkout 的版本。
- 当前机器是 910B2，并使用 CANN 8.5.0 专用镜像，因此不要额外把 `c5cc95e` clone 成第二个顶层 `cann-recipes-infer`。

在线取得代码时执行：

```bash
mkdir -p /home/mem/dsv4/code
cd /home/mem/dsv4/code
git clone https://gitcode.com/cann/cann-recipes-infer.git cann-recipes-infer
git -C cann-recipes-infer checkout 1a7fbd34
git -C cann-recipes-infer rev-parse --short HEAD
```

最后一行必须输出 `1a7fbd34`。如果内网无法 clone，就在联网电脑打开官方 `master` 页面，下载完整源码压缩包，上传并解压到：

注意：GitCode 当前的网页路由不支持用 `tree/1a7fbd34` 打开这个短 SHA，该地址会显示 404；这不等于 Git commit 不存在。精确固定版本使用上面的 `git checkout 1a7fbd34`。如果只能在网页手工下载，就从当前 `master` 下载完整仓库，并按下面的关键文件清单验收；官方目录页显示 DSV4 交付目录最近一次合并记录为 `1a7fbd34`（PR #682）。

```text
/home/mem/dsv4/code/cann-recipes-infer/
```

解压后不要多套一层目录。最终必须能直接访问：

```text
/home/mem/dsv4/code/cann-recipes-infer/integration/sglang/dsv4-flash-single-npu-moe-offload/apply_all.sh
```

它和 `ktransformers-AK` 是同级目录，不放进 `ktransformers-AK/third_party/`。

#### ktransformers

- 仓库：https://github.com/kvcache-ai/ktransformers
- 固定提交：https://github.com/kvcache-ai/ktransformers/commit/d7b5b49a3ef214a822aba613423551dd56416557
- 官方固定版本 ZIP：https://github.com/kvcache-ai/ktransformers/archive/d7b5b49a3ef214a822aba613423551dd56416557.zip

#### sglang

- 仓库：https://github.com/iforgetmyname/sglang
- 固定提交：https://github.com/iforgetmyname/sglang/commit/298193eb34c9d87debbdb5957edead0a8b9ec988
- 官方固定版本 ZIP：https://github.com/iforgetmyname/sglang/archive/298193eb34c9d87debbdb5957edead0a8b9ec988.zip

#### llama.cpp

- 仓库：https://github.com/ggml-org/llama.cpp
- 固定提交：https://github.com/ggml-org/llama.cpp/commit/a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f
- 官方固定版本 ZIP：https://github.com/ggml-org/llama.cpp/archive/a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f.zip

旧说明中的 `ggerganov/llama.cpp` 已重定向到 `ggml-org/llama.cpp`，提交 SHA 不变。

#### pybind11 子模块

- 仓库：https://github.com/pybind/pybind11
- 固定提交：https://github.com/pybind/pybind11/commit/bb05e0810b87e74709d9f4c4545f1f57a1b386f5
- 官方固定版本 ZIP：https://github.com/pybind/pybind11/archive/bb05e0810b87e74709d9f4c4545f1f57a1b386f5.zip

#### custom_flashinfer 子模块

- 仓库：https://github.com/kvcache-ai/custom_flashinfer
- 固定提交：https://github.com/kvcache-ai/custom_flashinfer/commit/fd94393fb5b8ba8bae9c0bd6ab1c2a429d81ac76
- 官方固定版本 ZIP：https://github.com/kvcache-ai/custom_flashinfer/archive/fd94393fb5b8ba8bae9c0bd6ab1c2a429d81ac76.zip

#### 两份权重

- NPU W8A8：https://modelscope.cn/models/sgl-npu/DeepSeek-V4-Flash-W8A8
- CPU 原生 MXFP4：https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash

#### Docker 镜像

- 镜像名：`lmsysorg/sglang:deepseek-v4-npu-910b`
- Docker Hub：https://hub.docker.com/r/lmsysorg/sglang

本次已经导出了 ARM64 tar，不需要在内网服务器 `docker pull`。

#### GPQA 可选依赖

- evalscope：https://pypi.org/project/evalscope/
- 项目：https://github.com/modelscope/evalscope

模型没跑通以前不要安装 evalscope。

---

## 2. 最终目录必须长这样

服务器上所有实验材料统一放在：

```text
/home/mem/dsv4/
```

完整结构：

```text
/home/mem/dsv4/
├── packages/
│   ├── deepseek-v4-npu-910b-arm64.tar
│   ├── deepseek-v4-npu-910b-arm64.tar.sha256
│   ├── dsv4-code-bundle.tar.gz                 # 可选：自己制作的传输包，不是官方文件
│   ├── wheels/                                 # GPQA 阶段才使用，可为空
│   └── debs/                                   # 仅容器不能联网安装 hwloc 时准备
│
├── image/
│   └── Dockerfile                              # 自己创建；不是下载物
│
├── code/
│   ├── cann-recipes-infer/
│   │   └── integration/sglang/
│   │       └── dsv4-flash-single-npu-moe-offload/
│   │           ├── apply_all.sh
│   │           ├── main_repo/
│   │           ├── sglang/
│   │           ├── llama_cpp/
│   │           └── scripts/
│   │
│   ├── ktransformers-AK/
│   │   ├── .git/
│   │   ├── kt-kernel/
│   │   └── third_party/
│   │       ├── pybind11/                       # bb05e081...
│   │       ├── custom_flashinfer/               # fd94393f...
│   │       ├── sglang/                          # 298193eb3...
│   │       └── llama.cpp/                       # a94e6ff8...
│   │
│   └── logs/
│       ├── image_check.log                     # 运行检查时生成
│       ├── kt-kernel-build.log                 # 编译时生成
│       ├── mxfp4-convert.log                   # 转换时生成
│       ├── gguf-verify.log                     # 校验时生成
│       └── serve-npu5.log                      # 启动服务时生成
│
└── models/
    ├── DeepSeek-V4-Flash-W8A8/
    │   ├── config.json
    │   ├── model.safetensors.index.json
    │   ├── tokenizer_config.json
    │   └── *.safetensors
    │
    ├── DeepSeek-V4-Flash/
    │   ├── config.json
    │   ├── model.safetensors.index.json
    │   ├── tokenizer_config.json
    │   └── *.safetensors
    │
    └── cache/
        ├── dsv4_layer0_mxfp4.gguf
        ├── dsv4_layer1_mxfp4.gguf
        ├── ...
        └── dsv4_layer42_mxfp4.gguf
```

注意：

- `dsv4-code-bundle.tar.gz` 不是需要从网上寻找的官方产物，只是把
  `cann-recipes-infer/` 和 `ktransformers-AK/` 一次传入内网时可自行制作的压缩包。
  如果代码已经分别放到下述最终位置，该文件可以完全不存在。
- `sglang` 和 `llama.cpp` 必须最终位于 `ktransformers-AK/third_party/`。
- `pybind11` 和 `custom_flashinfer` 不能是空目录。
- 权重目录不要重复嵌套一层同名目录。
- `image/Dockerfile` 是我们自己写的环境说明文件，不从任何仓库下载。
- `code/logs/` 开始时可以是空目录，里面五个 `.log` 文件均由后续命令的 `tee` 或输出重定向自动创建，不需要下载或提前创建空文件。
- 所有编译和服务日志写进 `/home/mem/dsv4/code/logs/`。

---

## 3. 基础镜像检查与派生镜像

“派生镜像”不是另一份需要从厂商下载的镜像。它的含义是：

```text
官方基础镜像
  + 本实验缺少的少量系统依赖
  = 我们自己构建并命名的派生镜像
```

本机已经对 `lmsysorg/sglang:deepseek-v4-npu-910b` 做过无 NPU 启动检查：镜像内已有 `git`、`gcc/g++`、`make`、`cmake`、`pkg-config` 和 Python 3.11，但没有 `/usr/include/hwloc.h`，也没有 `libhwloc.so`。因此编译 kt-kernel 前必须补齐 `libhwloc-dev` 与对应运行库。

`/home/mem/dsv4/image/Dockerfile` 是我们自己创建的文本文件，不需要提前下载；`image/` 目录也可以等基础镜像导入成功后再创建。

### 3.1 先判断容器能否访问 Ubuntu 软件源

服务器执行完 `docker load` 后测试：

```bash
docker run --rm \
  --entrypoint /bin/bash \
  lmsysorg/sglang:deepseek-v4-npu-910b \
  -lc 'timeout 30 apt-get update'
```

如果退出码为 0，走 §3.2；如果无法联网，走 §3.3。无需重新下载官方基础镜像。

### 3.2 容器能访问软件源：现场构建薄派生镜像

创建 `/home/mem/dsv4/image/Dockerfile`：

```dockerfile
FROM lmsysorg/sglang:deepseek-v4-npu-910b
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends libhwloc-dev libhwloc15 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
CMD ["/bin/bash"]
```

构建：

```bash
cd /home/mem/dsv4/image
docker build -t dsv4-offload-env:cann85-910b2 .
```

这次 build 使用本机已经 `docker load` 的基础镜像，只从 Ubuntu 源下载少量 hwloc 包，不会再次 `docker pull` 5 GB 镜像。

### 3.3 容器不能访问软件源：提前准备 ARM64 deb 包

你现在应走这一节。不要在服务器上执行 `apt-get update`，也不要单独从网页只下载一个 `libhwloc.so`。`libhwloc.so` 是运行期动态库，`hwloc.h` 和无版本的链接库则来自开发包；编译 kt-kernel 两者都要。

本机已经以同一份 `lmsysorg/sglang:deepseek-v4-npu-910b` ARM64 镜像为基准，使用 Ubuntu 22.04 官方软件源解析并实装验证了完整依赖。需要上传这两个很小的文件：

```text
dsv4-hwloc-arm64-debs.tar.gz
dsv4-hwloc-arm64-debs.tar.gz.sha256
```

压缩包约 2.3 MiB，SHA-256 是：

```text
94d065cdbd57938dd6ae423084900507151c05d5031970856567822a4f372ebe
```

把它们上传到：

```text
/home/mem/dsv4/packages/
```

服务器校验并解压：

```bash
cd /home/mem/dsv4/packages
sha256sum -c dsv4-hwloc-arm64-debs.tar.gz.sha256

mkdir -p /home/mem/dsv4/packages/debs
tar -xzf dsv4-hwloc-arm64-debs.tar.gz \
  -C /home/mem/dsv4/packages/debs \
  --strip-components=1

find /home/mem/dsv4/packages/debs -maxdepth 1 -name '*.deb' | wc -l
```

最后应显示 `17`。这些包是 Ubuntu 22.04 的 `arm64` 或架构无关的 `all` 包，不能换成 x86_64/amd64 包。

创建离线派生镜像的构建目录。这里复制一次只有约 2.3 MiB，目的是避免把数百 GiB 权重作为 Docker build context 发给 Docker 19.03：

```bash
mkdir -p /home/mem/dsv4/image/debs
cp -a /home/mem/dsv4/packages/debs/. /home/mem/dsv4/image/debs/
```

用 `vi /home/mem/dsv4/image/Dockerfile` 创建下面这个文件：

```dockerfile
FROM lmsysorg/sglang:deepseek-v4-npu-910b
USER root
COPY debs/ /tmp/dsv4-debs/
RUN dpkg -i /tmp/dsv4-debs/*.deb && \
    ldconfig && \
    rm -rf /tmp/dsv4-debs
WORKDIR /workspace
CMD ["/bin/bash"]
```

然后构建：

```bash
cd /home/mem/dsv4/image
docker build -t dsv4-offload-env:cann85-910b2 .
```

这个过程完全离线：`FROM` 使用已经 `docker load` 的本地基础镜像，`COPY` 使用刚上传的 deb 包，Dockerfile 里没有 `apt-get`、`curl` 或 `git clone`。

本离线包实际包含 `libhwloc-dev`、`libhwloc15` 及基础镜像缺失的间接依赖。已经在同一基础镜像中验证：

```text
/usr/include/hwloc.h                    存在
/lib/aarch64-linux-gnu/libhwloc.so      存在
/lib/aarch64-linux-gnu/libhwloc.so.15   存在
pkg-config --modversion hwloc            输出 2.7.0
```

### 3.4 派生镜像验收

```bash
docker run --rm \
  --entrypoint /bin/bash \
  dsv4-offload-env:cann85-910b2 \
  -lc 'set -e; test -f /usr/include/hwloc.h; ldconfig -p | grep libhwloc; pkg-config --modversion hwloc'
```

最后一行应输出 `2.7.0`。该命令成功后，后续实验使用 `dsv4-offload-env:cann85-910b2`；基础镜像仍保留作为可回退版本。

---

## 4. 在联网机准备最终代码结构

如果三个仓已经分别下载，仍要将它们按 KTransformers 的目录要求摆放。

### 4.0 没有 Git：在官网按固定提交手工下载（推荐给你）

服务器宿主机没有 Git 不影响本实验：下载和整理在联网电脑完成；后面执行补丁时，基础 Docker 镜像内已经有 `/usr/bin/git`。

在联网电脑浏览器中分别打开下面五个 **官方固定提交页面**。页面标题旁先确认短 SHA，然后点 `Browse files`（浏览此时的仓库）→ `Code` → `Download ZIP`。也可以直接点表中的“固定 ZIP”：

| 最终用途 | 固定提交页面 | 固定 ZIP |
|---|---|---|
| KTransformers 主仓 | https://github.com/kvcache-ai/ktransformers/commit/d7b5b49a3ef214a822aba613423551dd56416557 | https://github.com/kvcache-ai/ktransformers/archive/d7b5b49a3ef214a822aba613423551dd56416557.zip |
| SGLang fork | https://github.com/iforgetmyname/sglang/commit/298193eb34c9d87debbdb5957edead0a8b9ec988 | https://github.com/iforgetmyname/sglang/archive/298193eb34c9d87debbdb5957edead0a8b9ec988.zip |
| llama.cpp | https://github.com/ggml-org/llama.cpp/commit/a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f | https://github.com/ggml-org/llama.cpp/archive/a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f.zip |
| pybind11 | https://github.com/pybind/pybind11/commit/bb05e0810b87e74709d9f4c4545f1f57a1b386f5 | https://github.com/pybind/pybind11/archive/bb05e0810b87e74709d9f4c4545f1f57a1b386f5.zip |
| custom_flashinfer | https://github.com/kvcache-ai/custom_flashinfer/commit/fd94393fb5b8ba8bae9c0bd6ab1c2a429d81ac76 | https://github.com/kvcache-ai/custom_flashinfer/archive/fd94393fb5b8ba8bae9c0bd6ab1c2a429d81ac76.zip |

必须是上表中的完整 SHA。不要在每个仓库首页直接下载默认分支，也不要下载 `sgl-project/sglang` 主仓来代替 `iforgetmyname/sglang` fork。

KTransformers 的 GitHub ZIP **不会包含 Git 子模块内容**，所以解压后 `third_party/pybind11` 和 `third_party/custom_flashinfer` 为空是正常的；上表最后两个 ZIP 就是用来补这两个空目录的。SGLang 和 llama.cpp 也要用上表中的独立 ZIP 放进去。

CANN Recipes 这样手工下载：

1. 打开官方仓库：https://gitcode.com/cann/cann-recipes-infer/tree/master
2. 点页面仓库区的 `ZIP`/`下载 ZIP`，下载完整仓库，不要只保存网页上的 Markdown。
3. 同时打开官方 PR #682：https://gitcode.com/cann/cann-recipes-infer/pull/682，确认 DSV4 单卡 MoE offload 合并记录为 `1a7fbd34`。
4. 当前 `master` 已包含该合并；本实验只使用 `integration/sglang/dsv4-flash-single-npu-moe-offload/` 交付目录。下载后按 §1.1 的文件清单验收。由于 GitCode 的 `tree/1a7fbd34` 短 SHA 网页会 404，手工下载不再依赖这个地址。

下载完成后，先在联网电脑上双击解压这六个 ZIP，再按下面结构整理。目录后面的长 SHA 是 GitHub 自动生成的解压目录名；先借它确认版本，再改成最终短名字：

```text
dsv4-code-bundle/
├── cann-recipes-infer/                         # GitCode ZIP 解压目录改名
└── ktransformers-AK/                           # ktransformers-d7b5... 解压目录改名
    └── third_party/
        ├── pybind11/                           # pybind11-bb05... 的内容
        ├── custom_flashinfer/                  # custom_flashinfer-fd94... 的内容
        ├── sglang/                             # sglang-2981... 的内容
        └── llama.cpp/                          # llama.cpp-a94e... 的内容
```

这里的“内容”是指把解压目录本身改名/移动过去，不能再多套一层，例如下面是错误的：

```text
third_party/sglang/sglang-298193eb.../python/    # 错误：多了一层
```

下面才正确：

```text
third_party/sglang/python/                       # 正确
```

整理好后运行 §4.6 的结构检查并打成一个 `dsv4-code-bundle.tar.gz` 上传。因为网页 ZIP 没有 `.git` 历史，`git rev-parse HEAD` 报错是预期现象；版本依据是下载 URL/原始 ZIP 文件名中的完整 SHA。`apply_all.sh` 后面仍可使用镜像里的 `git apply` 对普通目录打补丁。

如果你已经按这些 SHA 下载过 KTransformers、SGLang、llama.cpp，不需要重复下载，只需补齐两个空子模块和 CANN Recipes 交付仓。

### 4.1 准备 KTransformers 主仓

本节是“联网电脑有 Git”时的可选方法；没有 Git 就跳过 §4.1～§4.5 的 Git 命令，使用 §4.0。

```bash
git clone https://github.com/kvcache-ai/ktransformers.git ktransformers-AK
git -C ktransformers-AK checkout d7b5b49a3ef214a822aba613423551dd56416557
```

### 4.2 下载两个原始子模块

最简单的方式：

```bash
git -C ktransformers-AK submodule update --init \
  third_party/pybind11 \
  third_party/custom_flashinfer
```

检查：

```bash
git -C ktransformers-AK/third_party/pybind11 rev-parse HEAD
git -C ktransformers-AK/third_party/custom_flashinfer rev-parse HEAD
```

预期：

```text
bb05e0810b87e74709d9f4c4545f1f57a1b386f5
fd94393fb5b8ba8bae9c0bd6ab1c2a429d81ac76
```

如果 `git submodule` 无法使用，也可以分别下载上述两个固定 commit，然后复制到对应目录。

### 4.3 放入固定版 SGLang

不要使用 KTransformers `.gitmodules` 默认的 sglang，使用本交付指定的 fork：

```bash
git clone https://github.com/iforgetmyname/sglang.git sglang-fixed
git -C sglang-fixed checkout 298193eb34c9d87debbdb5957edead0a8b9ec988
```

保证目标目录不存在或为空后，再复制：

```bash
cp -a sglang-fixed ktransformers-AK/third_party/sglang
```

### 4.4 放入固定版 llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp-fixed
git -C llama.cpp-fixed checkout a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f
```

保证目标目录不存在或为空后，再复制：

```bash
cp -a llama.cpp-fixed ktransformers-AK/third_party/llama.cpp
```

### 4.5 检查四个目录

```bash
git -C ktransformers-AK rev-parse HEAD
git -C ktransformers-AK/third_party/pybind11 rev-parse HEAD
git -C ktransformers-AK/third_party/custom_flashinfer rev-parse HEAD
git -C ktransformers-AK/third_party/sglang rev-parse HEAD
git -C ktransformers-AK/third_party/llama.cpp rev-parse HEAD
```

依次应为：

```text
d7b5b49a3ef214a822aba613423551dd56416557
bb05e0810b87e74709d9f4c4545f1f57a1b386f5
fd94393fb5b8ba8bae9c0bd6ab1c2a429d81ac76
298193eb34c9d87debbdb5957edead0a8b9ec988
a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f
```

### 4.6 可选：把代码制成一个离线传输包

这一步不是下载要求。如果服务器已经分别取得两个完整目录，可跳过本节。

如需一次上传，将已经准备好的 `cann-recipes-infer` 和 `ktransformers-AK` 放进同一个临时目录：

```bash
mkdir -p dsv4-code-bundle
cp -a cann-recipes-infer dsv4-code-bundle/cann-recipes-infer
cp -a ktransformers-AK dsv4-code-bundle/ktransformers-AK
```

检查结构：

```text
dsv4-code-bundle/
├── cann-recipes-infer/
└── ktransformers-AK/
```

```bash
test -f dsv4-code-bundle/cann-recipes-infer/integration/sglang/dsv4-flash-single-npu-moe-offload/apply_all.sh
test -d dsv4-code-bundle/ktransformers-AK/third_party/pybind11
test -d dsv4-code-bundle/ktransformers-AK/third_party/custom_flashinfer
test -d dsv4-code-bundle/ktransformers-AK/third_party/sglang
test -d dsv4-code-bundle/ktransformers-AK/third_party/llama.cpp
echo $?
```

应输出 `0`。

打包：

```bash
tar -czf dsv4-code-bundle.tar.gz dsv4-code-bundle
```

```bash
shasum -a 256 dsv4-code-bundle.tar.gz \
  > dsv4-code-bundle.tar.gz.sha256
```

将代码包和校验文件上传到 Hugging Face 私有 dataset 仓库。

---

## 5. 服务器建立目录

以下步骤开始在 910B2 服务器执行。

```bash
export DSV4_ROOT=/home/mem/dsv4
```

```bash
mkdir -p \
  "$DSV4_ROOT/packages" \
  "$DSV4_ROOT/image" \
  "$DSV4_ROOT/code" \
  "$DSV4_ROOT/code/logs" \
  "$DSV4_ROOT/models/DeepSeek-V4-Flash-W8A8" \
  "$DSV4_ROOT/models/DeepSeek-V4-Flash" \
  "$DSV4_ROOT/models/cache"
```

检查磁盘和内存：

```bash
df -h /home/mem
free -h
```

---

## 6. 服务器导入基础镜像并离线派生

你已经把镜像 tar 放在 `/home/mem/dsv4/packages/`，先看真实文件名：

```bash
cd /home/mem/dsv4/packages
ls -lh *.tar *.tar.sha256 2>/dev/null
```

如果文件名是本说明此前生成的名字，校验并导入：

```bash
cd /home/mem/dsv4/packages
sha256sum -c deepseek-v4-npu-910b-arm64.tar.sha256
docker load -i deepseek-v4-npu-910b-arm64.tar
```

若你上传时改过名字，`sha256sum -c` 文件中的文件名也必须相应一致；不要盲目照抄。导入后检查：

```bash
docker image inspect lmsysorg/sglang:deepseek-v4-npu-910b \
  --format 'arch={{.Architecture}} id={{.Id}}'
```

必须显示 `arch=arm64`。

接着严格执行 §3.3：上传并解压 `dsv4-hwloc-arm64-debs.tar.gz`，在服务器本地构建：

```bash
cd /home/mem/dsv4/image
docker build -t dsv4-offload-env:cann85-910b2 .
```

再执行 §3.4 验收。你不需要额外下载一份完整的“派生镜像 tar”；派生镜像由现有基础镜像加 2.3 MiB 离线 deb 包现场生成。

---

## 7. 服务器放置代码

### 7.1 使用了自制代码包

如果上一节制作了 `dsv4-code-bundle.tar.gz`，将它放到
`/home/mem/dsv4/packages/` 后执行：

```bash
cd /home/mem/dsv4/packages
sha256sum -c dsv4-code-bundle.tar.gz.sha256
```

```bash
tar -xzf dsv4-code-bundle.tar.gz \
  -C /home/mem/dsv4/code \
  --strip-components=1
```

### 7.2 没有制作代码包

如果两个目录是分别上传的，只需把它们放成：

```text
/home/mem/dsv4/code/cann-recipes-infer/
/home/mem/dsv4/code/ktransformers-AK/
```

不需要创建 `dsv4-code-bundle.tar.gz`。

设置路径：

```bash
export DSV4_ROOT=/home/mem/dsv4
export REPO="$DSV4_ROOT/code/ktransformers-AK"
export RELEASE_DIR="$DSV4_ROOT/code/cann-recipes-infer/integration/sglang/dsv4-flash-single-npu-moe-offload"
```

检查关键文件：

```bash
test -f "$RELEASE_DIR/apply_all.sh"
test -f "$RELEASE_DIR/scripts/launch_dsv4_singleCard_cann8.5.0_910b.sh"
test -d "$RELEASE_DIR/scripts/tools"
test -d "$REPO/third_party/pybind11"
test -d "$REPO/third_party/custom_flashinfer"
test -d "$REPO/third_party/sglang"
test -d "$REPO/third_party/llama.cpp"
echo $?
```

应输出 `0`。

检查版本：

如果源码来自 `git clone`，执行：

```bash
git -C "$REPO" rev-parse HEAD
git -C "$REPO/third_party/pybind11" rev-parse HEAD
git -C "$REPO/third_party/custom_flashinfer" rev-parse HEAD
git -C "$REPO/third_party/sglang" rev-parse HEAD
git -C "$REPO/third_party/llama.cpp" rev-parse HEAD
```

如果源码来自 §4.0 的官网 ZIP，上述命令会因为没有 `.git` 目录而失败，不要据此重新下载。改为检查关键文件与目录，并保留六个原始 ZIP 文件名作为版本凭据：

```bash
test -f "$REPO/setup.py"
test -f "$REPO/third_party/pybind11/CMakeLists.txt"
test -n "$(find "$REPO/third_party/custom_flashinfer" -mindepth 1 -maxdepth 1 -print -quit)"
test -d "$REPO/third_party/sglang/python/sglang"
test -f "$REPO/third_party/llama.cpp/convert-hf-to-gguf.py"
echo $?
```

应输出 `0`。对应完整 SHA 必须是：

```text
ktransformers      d7b5b49a3ef214a822aba613423551dd56416557
pybind11           bb05e0810b87e74709d9f4c4545f1f57a1b386f5
custom_flashinfer  fd94393fb5b8ba8bae9c0bd6ab1c2a429d81ac76
sglang             298193eb34c9d87debbdb5957edead0a8b9ec988
llama.cpp          a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f
```

---

## 8. 放置和检查权重

权重最终路径：

```text
/home/mem/dsv4/models/DeepSeek-V4-Flash-W8A8
/home/mem/dsv4/models/DeepSeek-V4-Flash
```

检查大小：

```bash
du -sh \
  /home/mem/dsv4/models/DeepSeek-V4-Flash-W8A8 \
  /home/mem/dsv4/models/DeepSeek-V4-Flash
```

参考：

- W8A8 约 275 GiB
- 原版 MXFP4 约 150 GiB

检查索引引用的分片是否齐全：

```bash
python3 - <<'PY'
import json
from pathlib import Path

roots = [
    Path("/home/mem/dsv4/models/DeepSeek-V4-Flash-W8A8"),
    Path("/home/mem/dsv4/models/DeepSeek-V4-Flash"),
]

failed = False
for root in roots:
    print(f"\n检查 {root}")
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        print("失败：缺少 model.safetensors.index.json")
        failed = True
        continue
    data = json.loads(index.read_text())
    shards = sorted(set(data["weight_map"].values()))
    missing = [name for name in shards if not (root / name).is_file()]
    print("索引分片数：", len(shards))
    print("缺失分片数：", len(missing))
    for name in missing[:20]:
        print("缺失：", name)
    failed = failed or bool(missing)

raise SystemExit(1 if failed else 0)
PY
```

退出码必须是 0。

---

## 9. 检查物理 NPU 5

```bash
npu-smi info
```

```bash
ls -l \
  /dev/davinci5 \
  /dev/davinci_manager \
  /dev/devmm_svm \
  /dev/hisi_hdc
```

确认 NPU 5 没有被其他任务占用，再启动容器。

---

## 10. 启动物理 NPU 5 构建容器

```bash
export DSV4_ROOT=/home/mem/dsv4
export RELEASE_DIR="$DSV4_ROOT/code/cann-recipes-infer/integration/sglang/dsv4-flash-single-npu-moe-offload"
```

```bash
WORKSPACE="$DSV4_ROOT/code" \
MODEL_DIR="$DSV4_ROOT/models" \
IMAGE=dsv4-offload-env:cann85-910b2 \
NAME=dsv4-npu5 \
SERVICE_PORT=8020 \
SHM_SIZE=64g \
NPU_VISIBLE_DEVICES=5 \
bash "$RELEASE_DIR/scripts/launch_dsv4_singleCard_cann8.5.0_910b.sh"
```

进入容器：

```bash
docker exec -it dsv4-npu5 bash
```

容器内检查：

```bash
npu-smi info
```

```bash
python3 - <<'PY'
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("device_count:", torch.npu.device_count())
print("available:", torch.npu.is_available())

try:
    import custom_ops
    print("custom_ops: OK")
except Exception as exc:
    print("custom_ops FAILED:", repr(exc))
PY
```

通过标准：

```text
device_count: 1
available: True
custom_ops: OK
```

物理卡 5 被单独映射后，容器通常将其表示为逻辑卡 0。后续使用 `NPU_DEVICE_ID=0`。

---

## 11. 打补丁

以下命令都在容器内执行。

```bash
export REPO=/workspace/code/ktransformers-AK
export RELEASE_DIR=/workspace/code/cann-recipes-infer/integration/sglang/dsv4-flash-single-npu-moe-offload
```

如果源码由 Git clone 获得，确认版本：

```bash
git -C "$REPO" rev-parse --short HEAD
git -C "$REPO/third_party/sglang" rev-parse --short HEAD
git -C "$REPO/third_party/llama.cpp" rev-parse --short HEAD
```

预期：

```text
d7b5b49
298193eb3
a94e6ff
```

如果源码来自官网 ZIP，跳过这三条 `rev-parse`；ZIP 没有 `.git` 元数据。先执行 §7 的非 Git 结构检查，再继续。镜像里的 `git apply` 可以给普通目录应用补丁。

应用补丁：

```bash
bash "$RELEASE_DIR/apply_all.sh" "$REPO"
```

安装交付工具脚本：

```bash
mkdir -p "$REPO/tools"
cp -a "$RELEASE_DIR/scripts/tools/." "$REPO/tools/"
```

补丁失败时不要使用 `--reject` 或强制应用。先检查下载 URL 中的 SHA、目录是否多套了一层，以及补丁是否已经应用过。

---

## 12. 编译 kt-kernel

选择 Python：

```bash
export PYTHON_BIN=$(command -v python3.11 || command -v python3)
"$PYTHON_BIN" --version
```

后续预检、转换和启动必须使用同一个 `PYTHON_BIN`。

编译：

```bash
cd "$REPO/kt-kernel"
```

```bash
CPUINFER_USE_ASCEND_NPU=1 \
CPUINFER_ARM_SVE=OFF \
CPUINFER_ARM_BF16=OFF \
CPUINFER_ARM_I8MM=OFF \
"$PYTHON_BIN" setup.py build_ext --inplace \
  2>&1 | tee /workspace/code/logs/kt-kernel-build.log
```

构建结果必须满足：

```text
LLAMA_ARM_DOTPROD=ON
SVE=OFF
BF16=OFF
I8MM=OFF
Found Ascend CL library
```

检查扩展：

```bash
find "$REPO/kt-kernel" -name 'kt_kernel_ext*.so'
```

注册包：

```bash
ln -sfn python "$REPO/kt-kernel/kt_kernel"
export PYTHONPATH="$REPO/third_party/sglang/python:$REPO/kt-kernel${PYTHONPATH:+:$PYTHONPATH}"
```

验证：

```bash
"$PYTHON_BIN" -c 'import kt_kernel; print("kt_kernel import OK")'
```

---

## 13. 转换 43 个 CPU GGUF

```bash
export W8A8_DIR=/workspace/models/DeepSeek-V4-Flash-W8A8
export MXFP4_SRC=/workspace/models/DeepSeek-V4-Flash
export GGUF_CACHE=/workspace/models/cache
```

```bash
mkdir -p "$GGUF_CACHE"
cd "$REPO"
```

开始转换：

```bash
nohup "$PYTHON_BIN" tools/batch_convert_mxfp4_layers_mp.py \
  --input "$MXFP4_SRC" \
  --output-dir "$GGUF_CACHE" \
  --layer-start 0 \
  --layer-end 42 \
  --jobs 16 \
  --verify-sample 3 \
  > /workspace/code/logs/mxfp4-convert.log 2>&1 &
```

查看进度：

```bash
tail -f /workspace/code/logs/mxfp4-convert.log
```

完成后检查：

```bash
find "$GGUF_CACHE" -maxdepth 1 \
  -name 'dsv4_layer*_mxfp4.gguf' | wc -l
```

必须为 43。

完整校验：

```bash
"$PYTHON_BIN" tools/verify_mxfp4_gguf_set.py \
  --dir "$GGUF_CACHE" \
  --sha256-manifest tools/mxfp4_gguf_sha256.txt \
  2>&1 | tee /workspace/code/logs/gguf-verify.log
```

校验通过前不要删除原始 MXFP4 权重。

---

## 14. 启动前预检

每次新 shell 都重新设置：

```bash
export REPO=/workspace/code/ktransformers-AK
export PYTHON_BIN=$(command -v python3.11 || command -v python3)
export W8A8_DIR=/workspace/models/DeepSeek-V4-Flash-W8A8
export GGUF_CACHE=/workspace/models/cache
export PYTHONPATH="$REPO/third_party/sglang/python:$REPO/kt-kernel${PYTHONPATH:+:$PYTHONPATH}"
```

```bash
cd "$REPO"
GGUF_DIR="$GGUF_CACHE" \
GGUF_SUFFIX=_mxfp4 \
bash tools/e2e_preflight.sh
```

检查退出码：

```bash
echo $?
```

必须为 0。

---

## 15. 启动物理 NPU 5 的单卡服务

由于容器只映射物理卡 5，这里使用逻辑卡 0：

```bash
cd "$REPO"
```

```bash
NPU_DEVICE_ID=0 \
PORT=8020 \
PYTHON_BIN="$PYTHON_BIN" \
MODEL_PATH="$W8A8_DIR" \
KT_GGUF_TEMPLATE="$GGUF_CACHE/dsv4_layer{layer_idx}_mxfp4.gguf" \
KT_THREADPOOL_COUNT=8 \
KT_CPUINFER=128 \
KT_NUM_GPU_EXPERTS=32 \
CHUNKED_PREFILL_SIZE=32768 \
bash tools/launch_ds4flash_npu.sh \
  2>&1 | tee /workspace/code/logs/serve-npu5.log
```

注意：

- 服务保持前台运行。
- graph 模式保持默认，不加 `--disable-cuda-graph`。
- 第一轮不要并发发送请求。
- 如果容器没有将物理卡 5 重映射成逻辑卡 0，以容器内实际编号修改 `NPU_DEVICE_ID`。

---

## 16. 功能验收

另开宿主机终端：

```bash
until curl -sf http://127.0.0.1:8020/health >/dev/null; do
  sleep 5
done
```

```bash
curl -sS -X POST http://127.0.0.1:8020/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"中国的首都是","sampling_params":{"max_new_tokens":64,"temperature":0}}'
```

通过标准：

- 输出是连贯中文
- 能正确回答北京
- 没有乱码、无限重复、NaN
- 服务进程不退出

检查吞吐日志：

```bash
grep 'gen throughput' /home/mem/dsv4/code/logs/serve-npu5.log
```

---

## 17. 单请求性能

服务必须以 `CHUNKED_PREFILL_SIZE=32768` 启动。

在容器内执行：

```bash
cd "$REPO"
PORT=8020 bash tools/decode_throughput_test.sh
```

必须看预热后的稳态结果。冷启动和首次 page cache 未命中的结果不作为最终数据。

当前交付使用 `--max-running-requests 1`。在单请求稳定以前，不进行并发 100 测试。

---

## 18. 并发 5、10、80 的聚合 tok/s 压测

仓库提供一个只依赖 Python 3 标准库的脚本：

```text
scripts/bench_concurrency.py
```

它请求 OpenAI 兼容的 `/v1/chat/completions` 流式接口，统计：

- 请求成功率和 request/s
- 聚合输出吞吐 `completion_tokens / 墙钟时间`，单位 tok/s
- 聚合总吞吐 `(prompt_tokens + completion_tokens) / 墙钟时间`
- E2E、TTFT、TPOT 的 mean、p50、p95、p99

在宿主机上的 Ds910 仓库目录执行。一次只测一个档位更容易发现问题：

```bash
cd /home/mem/Ds910

python3 scripts/bench_concurrency.py \
  --base-url http://127.0.0.1:8020 \
  --model DeepSeek-V4-Flash \
  --concurrency 5 \
  --requests 20 \
  --max-tokens 128
```

依次改为：

```bash
--concurrency 10 --requests 20
--concurrency 80 --requests 80
```

也可以一次跑三个档位：

```bash
python3 scripts/bench_concurrency.py \
  --base-url http://127.0.0.1:8020 \
  --model DeepSeek-V4-Flash \
  --concurrency 5 10 80 \
  --max-tokens 128
```

未指定 `--requests` 时，每档使用 `max(20, concurrency)` 个请求。并发 80 在当前 `--max-running-requests 1` 的单发路径上主要测排队、成功率和聚合吞吐，可能运行十几分钟，并不代表 80 条序列同时 decode。

JSON 明细默认写入：

```text
results/concurrency/
```

服务必须返回 OpenAI `usage` 才能精确统计 tok/s。缺少 usage 时脚本显示 `N/A`，不会把 SSE 数据块数量冒充 token 数。

---

## 19. GPQA-Diamond

模型功能和吞吐通过后才安装：

```bash
pip install evalscope
```

内网环境应提前准备 evalscope 及其依赖 wheel，或者在能够访问内网 PyPI 的环境安装。

运行：

```bash
cd "$REPO"
MODEL_PATH="$W8A8_DIR" \
PORT=8020 \
bash tools/gpqa_accuracy_repeat.sh
```

应重复多轮比较均值，不用单轮分数下结论。

---

## 20. 停止服务、停止容器和删除容器

如果服务正在当前终端前台运行，先按 `Ctrl+C`，这只停止模型服务。然后回到宿主机执行：

```bash
docker ps -a --filter 'name=^/dsv4-npu5$'
docker stop --time 60 dsv4-npu5
```

`docker stop` 不删除容器。以后需要继续检查容器文件时可以启动并进入：

```bash
docker start dsv4-npu5
docker exec -it dsv4-npu5 bash
```

注意：`docker start` 只恢复容器，不保证模型服务自动重新拉起；需要进入容器重新执行 §15 的启动命令。

确定不再需要这个容器后，再删除：

```bash
docker stop --time 60 dsv4-npu5 2>/dev/null || true
docker rm dsv4-npu5
```

验收：

```bash
docker ps -a --filter 'name=^/dsv4-npu5$'
```

应没有结果。不要使用模糊名称、通配符或批量 `docker rm`。删除容器不会删除绑定挂载的 `/home/mem/dsv4/models`、`code`、`logs` 和 GGUF；也不会删除 `dsv4-offload-env:cann85-910b2` 镜像。

---

## 21. 当前不要下载或安装的东西

以下内容不属于这条固定复现路线：

- DeepSeek-V4-Flash-0731-W8A8
- DeepSeek-V4-Flash-w8a8-mtp
- vLLM-Ascend
- ModelSlim
- CUDA/NVIDIA 软件
- 最新版 SGLang
- 最新版 KTransformers
- A3 的 CANN 9.0.0 环境脚本
- A3 的 ops-transformer 源码构建链

910B/CANN 8.5.0 路径依赖指定镜像中已经集成的 torch、torch_npu、CANN 和 NPU 自定义算子。

---

## 22. 最短路线图

```text
补齐 cann-recipes-infer 完整交付目录
  ↓
确认 pybind11 和 custom_flashinfer 不为空且 SHA 正确
  ↓
上传基础镜像 tar（你已完成）+ 2.3 MiB hwloc 离线包
  ↓
服务器 docker load 基础镜像并离线构建派生镜像
  ↓
按固定 SHA 手工下载并整理完整代码包
  ↓
服务器统一放入 /home/mem/dsv4
  ↓
检查两份权重分片齐全
  ↓
映射物理 NPU 5 启动容器
  ↓
apply_all.sh
  ↓
编译 kt-kernel
  ↓
转换并校验 43 个 GGUF
  ↓
预检
  ↓
启动单卡服务
  ↓
连贯性请求
  ↓
单请求吞吐
  ↓
并发 5 / 10 / 80 聚合 tok/s
  ↓
GPQA-Diamond
```
