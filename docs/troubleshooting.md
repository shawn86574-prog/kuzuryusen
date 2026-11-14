# 故障排查 · TP=2 实例在跑而 TP=1 不起作用

> 记录日期：2026-11-07。以本机 ComfyUI-AMD 的 8188 单实例为样本分析，最终在同一安装上启动
> TP=2 双实例验收通过。占用权互斥机制的背景见
> [x99-7900xtx-p2p · dual-gpu-ownership](https://github.com/shawn86574-prog/x99-7900xtx-p2p/blob/master/docs/dual-gpu-ownership.md)。

## 现象

用户观察到「ComfyUI TP=2 的实例在跑，TP=1 的实例却不加速」。分析结论是：**那个所谓
TP=1 的实例从头到尾就没在 TP 模式下运行过**——它打的是 `world=1` 的原生单卡路径，加速
无从谈起；而当时 8191/8192 双实例并未运行。两边的差别不在配置项多少，而在**启动模型**。

## 根因：TP=2 是双进程模型，8188 是单进程

| 证据 | 内容 | 结论 |
|---|---|---|
| 启动命令 | `main.py --disable-dynamic-vram --lowvram --port 8188`，无任何 `COMFY_TP_*` 环境变量 | 没有 `COMFY_TP_WORLD` → 代码取默认值 1 |
| 运行日志 | 多次打印 `[tp2] world=1 rank=0 enrolled` | 节点加载了，但 `_init_pg()` 首行 `if TP_WORLD <= 1: return` 直接早退：通信子不建、TP/MLP/VAE 钩子全部不安装 |
| 启动脚本 | 自启动脚本仅 `export HIP_VISIBLE_DEVICES=0,1`，其余只有 `--lowvram --port 8188` | 双卡**可见** ≠ TP：可见只是必要条件之一 |
| 实测速度 | 采样 39.6 ~ 60 s/it（单卡 + lowvram），一次 prompt 11 分 17 秒 | 与 TP2 的 7.6 s/step 相差 **5.2 ~ 7.9 倍**，走的是纯单卡路径 |
| 对照 | `scripts/run_tp.sh 2` 才注入 `COMFY_TP_WORLD=2`、`COMFY_TP_RANK=0/1`、`MASTER_ADDR/MASTER_PORT`，每 rank 各起一个进程各绑一张卡 | 单进程无论怎么改环境变量都构不成 TP=2 |

一句话：**TP=1「不起作用」不是 bug，是设计——`world=1` 时全部钩子早退，走 100% 原生
路径，它本来就不该有任何加速；出问题的是把 TP=2 误当成了「一个进程开双卡」。**

## 排查口径（先看这三行）

1. **看日志**：`grep '\[tp2\]' r0.log r1.log` —— `world=1` 即没进 TP；`world=2` 且
   `rank=0`/`rank=1` 各一条才算成立。
2. **看进程**：`pgrep -af 'main.py'` —— 必须有 **2 个**进程，各自 `HIP_VISIBLE_DEVICES`
   绑不同卡；单进程即使 `HIP_VISIBLE_DEVICES=0,1` 也不是 TP。
3. **看占用权**：`acquire` 失败先读 `/sys/class/drm/card*/device/mem_info_vram_used`，
   再用 `ls -l /proc/*/fd 2>/dev/null | grep -E 'drm|kfd'` 找残留子进程。

## 解决过程与实测结果

| 步骤 | 动作 | 结果 |
|---|---|---|
| ① 检查节点版本 | 该安装的 `comfyui-tp2` 仅 377 行（旧版：只有采样 TP，无 VAE 跨卡/心跳/帧对账） | 备份后覆盖为本仓库 832 行最新版 |
| ② 参数化启动脚本 | `run_tp.sh` 的 `COMFY_DIR`/`PYTHON` 支持环境变量覆盖 | 一条命令即可指向任一 ComfyUI 安装 |
| ③ 停单实例 | 杀掉 8188（运行 41 分钟，用户批准打断出片） | 8188 端口下线 |
| ④ 清残留显存 | `gpu-mode acquire` 两次失败：`ComfyUI_Qwen_H3_Prompt` 节点的 `llama-serve` 子进程（pid 292464）残留，两卡各占 9.7 GB；杀掉后回落到 48 MiB | 第二坑：**节点 spawn 的子进程不随 ComfyUI 退出**，会挡住下一次双卡占用 |
| ⑤ 启动 TP=2 | `COMFY_DIR=…/ComfyUI-AMD/ComfyUI PYTHON=…/venv/bin/python COMFY_TP_VAE_SPLIT=1 bash scripts/run_tp.sh 2 8191 amd` | 自动挂起常驻服务、P2P 拓扑补丁生效（`NCCL_TOPO_FILE`）、rank0/1 双端口 ready |
| ⑥ 验收 | 日志 `[tp2] world=2 rank=0 enrolled` + `[tp2] world=2 rank=1 enrolled`；`curl 8191/8192/system_stats` 均正常返回 | **TP=2 生效**，仅剩已知的 `hipErrorInvalidImage` flash-attention 回退警告（与 TP 无关） |

## 教训

1. **`HIP_VISIBLE_DEVICES=0,1` 不是 TP**。它是必要条件，不是充分条件；判定只认
   `COMFY_TP_WORLD`，验收只认日志里的 `world=2`。
2. **关闭 ComfyUI 要关进程组**。自定义节点（推理后端类）会 `spawn` 长驻子进程，
   主进程退出后它们继续占显存；停实例用 `kill -- -<pgid>` 或先按 `/proc/*/fd` 找到
   子进程再清。
3. **acquire 失败先怀疑残留，再怀疑服务**。占用权检查的是
   `mem_info_vram_used < 1500 MiB`，9.7 GB 级的孤儿进程是最常见的挡路原因。

## 数据来源

本机 8188 单实例日志（`/tmp/comfy_launch3.log`）、TP=2 验收日志
（`tp2/logs/amd/r0.log`、`r1.log`）、`AutoLaunch_ComfyUI.sh`、`scripts/run_tp.sh`、
`comfyui-tp2/__init__.py`。s/it 与 s/step 为各自日志实测；倍数按 39.6~60 ÷ 7.614 计算。

---

# 附 · TP=2 输出纯噪声：插件被加载两次导致权重双重切分

> 记录日期：2026-11-14。同一套配置单卡正常、TP2 出片全噪声，误判方向是「LoRA 维度不匹配」。

## 现象

TP2 双实例跑 DMD + Combat，成片为纯噪声（抽帧全屏静态噪点），换 `ema_pruned` 甚至完全不带
LoRA 依然噪声；同一工作流、同一 seed 在 8188 单卡上画质正常。两个 rank 的输出 md5 一致，
说明两侧算的是同一份错误结果，而不是某一侧坏了。

## 根因：`custom_nodes` 里有两个插件副本

```
custom_nodes/
├── comfyui-tp2                            ← 正在用的 832 行版本
└── comfyui-tp2.bak-377line-11071942       ← 11-07 的备份，同样会被加载
```

ComfyUI 会把 `custom_nodes` 下每个子目录都当插件导入（只跳过 `.` 开头和 `__pycache__`），
于是同一套 monkeypatch 被执行两次：

```
[tp2] rank0 state_dict split: heads 56->28, keys 932
[tp2] rank0 state_dict split: heads 28->14, keys 932   ← 第二次，head 只剩 1/4
```

qkv 头数被切到 14（应为 28），注意力彻底错位，输出必然是噪声。原来的
`_split_ids = set()` 只按 `id(sd)` 去重，两个模块实例各有自己的 set，挡不住。

## 修复

1. 备份移出插件目录：`mv custom_nodes/comfyui-tp2.bak-* /home/jay/tp2/backups/`
2. 两处 patch 加幂等标记（`comfyui-tp2/__init__.py`）：

```python
if not getattr(comfy.sd, "_tp2_state_dict_patched", False):
    comfy.sd.load_diffusion_model_state_dict = _tp_load_diffusion_model_state_dict
    comfy.sd._tp2_state_dict_patched = True
```

LoRA 那处同理（`_tp2_lora_patched`）。标记挂在 `comfy.sd` 上，跨模块副本共享。

## 判定口径（三行）

```bash
grep 'state_dict split' r0.log        # 只能出现一次 heads 56->28；出现两次即中招
grep 'lora split keys' r0.log         # 208/416（单 LoRA）或 208/828（DMD），各一次
ls custom_nodes | grep -i tp2        # 只应有 comfyui-tp2 一个条目
```

## 复验矩阵（832×480 / 56 帧，seed 145）

| 配置 | 修复前 | 修复后 |
|---|---|---|
| 单卡 + ema_pruned | 正常 | 正常 |
| 单卡 + DMD + Combat | 正常 | 正常 |
| TP2 + ema_pruned | **纯噪声** | 正常 |
| TP2 + DMD + Combat | **纯噪声** | 正常 |
| TP2 无 LoRA（20 步） | **纯噪声** | — |

修复后 per_step 3.47 s（双重切分时为 3.98 s，切多了反而更慢），显存 12.0–13.7 GB/卡。

## 教训

1. **插件目录只放一个副本**。备份放到仓库或 `~/backups`，不要留在 `custom_nodes`。
2. **「两侧 md5 一致」不等于正确**。它只能证明两卡同步，不证明切分正确；画质验收必须抽帧。
3. **先排除 LoRA 再怀疑 TP**：本次不带 LoRA 仍噪声，一步就把范围收窄到核心切分。
4. **切分次数可直接量化**：`heads 56->28` 只应出现一次，日志里数一下即可判定。
