# 九頭龍閃 · Kuzuryūsen

把 ComfyUI 上的 MiniMax H3 视频生成拆到两张卡上：采样阶段切分 MLP 并让卡间归约与 GEMM 在两条流上重叠，出片阶段把 VAE 解码按模型自带的时间块分摊给两张卡。两项都不是近似算法，输出与单卡路径逐位一致。

名字取自『るろうに剣心』飞天御剑流奥义——九记简单的斩击同时命中九处，合成一击。这里对应的是：几项单独看着不起眼的改动，拼起来才形成可用的整体提速。

> 实测基线：2×Radeon RX 7900 XTX（ROCm / RCCL），832×480、124 帧、8 步。
> 端到端 152.0 → 132.2 s（−13.0%），其中出片阶段 34.0 → 27.1–28.5 s（−16.2% ~ −20.3%），采样 8.810 → 7.616 s/step（−13.6%），像素最大绝对差 0。

## 目录

```
comfyui-tp2/            ComfyUI 自定义节点（TP 分片 + VAE 跨卡解码 + TP2VAEDecodeSplit）
scripts/                run_tp.sh / run_case.sh / bench.py / p2p_patch.py 等启动与压测脚本
p2p/                    内核级 PCIe P2P 打开工具（源码 + 安装脚本 + 拓扑探针）
experiments/hip_attn/   int8 attention 实验（HIP kernel + harness，未并入主路径）
docs/acceptance.md      验收报告（改动思路、三场景对比、上限差距、开关与失效边界）
docs/dashboard/         验收看板（自包含 HTML + ECharts）
```

## 它解决什么

单卡出片时，采样结束后的 post 阶段约 34 s，其中视频 VAE 解码 18.56 s、CPU 帧转换 8.35 s、编码 5.16 s、音频 1.93 s、权重加载 2.20 s。VAE 解码在 24 GB 卡上既吃显存又吃时间，而这部分时间原本完全串行。

两条改动：

1. **采样侧（MLP 分块 + 双流重叠）**：把 MLP 按序列切 8 块，第 i 块的 `all_reduce` 与第 i+1 块的 GEMM 放在不同 stream 上，藏掉约 38% 的卡间通信时间。
2. **出片侧（VAE 跨卡）**：复用 VAE 内部 `tokens_chunk_size=5 / token_overlap=2 / frame_pre_padding=3` 的时间块边界，rank0 解前 `c+1` 块、rank1 解后 `n-c` 块，接缝块两侧各解一遍，再用 `batch_isend_irecv` 交换重叠帧。帧计数与上游 `decode_output_shape` 严格对账，漂移直接报错。

外加四项“单项收益≈0、但决定特性可用”的兜底：P2P 拓扑补丁、通信子预热、60 s 心跳、`IS_CHANGED→NaN` 强制每步执行。没有它们，拆分要么挂死 600 s，要么收益减半。

## 实测数据

| 配置 | per_step | post | e2e |
|---|---|---|---|
| 单卡 | 11.235 s | 39.2 s | 159.4 s |
| TP2 + MLP 分块 | 7.618 s | 34.0 s | 136.7 s |
| + VAE 跨卡 | 7.614 s | 27.1 s | 131.9 s |

出片分段（基线 → 拆分）：视频解码 18.56 → 11.97 s，帧交换 0 → 0.39 s，其余基本不变。

正确性：同一 latent 分别走单卡与拆分，`max_abs_diff = 0`（接缝帧亦为 0）；latent 长度 1–89 的控制流逐块对账全部通过。

## 环境要求

- 两张 GPU，且**每卡放得下 VAE + 一个时间块**（24 GB 级实测可用）
- PyTorch + `torch.distributed` 的 NCCL/RCCL 后端（ROCm 已验证；CUDA 未验证，见限制）
- ComfyUI，以及 MiniMax H3 的 UNet / CLIP / 视频 VAE / 音频 VAE 权重

## 使用

1. 把 `comfyui-tp2/` 放进 `ComfyUI/custom_nodes/`。
2. 启动双实例（脚本按 `COMFY_DIR`、`PYTHON`、`OUT_DIR` 环境变量取值）：

```bash
export COMFY_DIR=/path/to/ComfyUI
export PYTHON=/path/to/python
COMFY_TP_VAE_SPLIT=1 bash scripts/run_tp.sh 2 8191 myrun
```

3. 工作流里把 `VAEDecode` 换成 `TP2VAEDecodeSplit`（或 `bench.py --vae-split` 自动替换）。
4. 单卡 / 非 TP 场景不需要任何改动：`TP_WORLD=1` 时所有钩子首行早退，与原生 ComfyUI 行为一致。

## 开关

| 变量 | 默认 | 作用 |
|---|---|---|
| `COMFY_TP_WORLD` | 1 | 实例数；1 表示完全走原生路径 |
| `COMFY_TP_VAE_SPLIT` | 0 | 1 = 启用 VAE 跨卡拆分 |
| `COMFY_TP_MLP_CHUNKS` | 8 | MLP 分块数；0 回到逐位旧路径 |
| `COMFY_TP_AR_MODE` | bf16 | int8 定点归约更快但画质未通过，不建议 |
| `COMFY_TP_P2P` | 1 | 生成 `NCCL_TOPO_FILE` 的拓扑补丁；0 关闭 |
| `COMFY_TP_VAE_TIMEOUT` | 60 | 心跳超时秒数，超时快速失败而非挂到 NCCL 30 min |

## 限制

- **对大模型推理没有正面收益**。本机 A/B 微基准：LLM decode 每层 0.253 → 1.368 ms（+440.8%），prefill 128 token 0.503 → 1.445 ms（+187.2%），只有 prefill 512 token 是 −14.6%。因此全部改动默认关闭、不常驻。
- **NVIDIA 消费卡（4090/4080 等）不能直接套用**：驱动层禁用 PCIe P2P，脚本按 ROCm 编写，CUDA 路径未验证。
- 时长不足 3 个时间块（约 < 40 帧）时自动退回整片解码，无收益。
- 上游 VAE 常量变更会使帧计划对账失败并报错，这是刻意设计（宁可直接失败，不产错帧）。
- 本机环境存在既有音频流缺失问题（`avcodec_send_frame() returned 22`），基线与拆分一致，与本次改动无关。

以上细节、三场景对比、上限差距与失效矩阵见 [docs/acceptance.md](docs/acceptance.md)，可视化版本见 [docs/dashboard/vae-split-dashboard.html](docs/dashboard/vae-split-dashboard.html)。

## 许可

MIT，见 [LICENSE](LICENSE)。
