"""MiniMax-H3 TP2 for RDNA3 (gfx1100) — 按帖子 2643 的思路实现。

思路：
  1. 启动两个 ComfyUI 实例，每个实例只可见一张卡（HIP_VISIBLE_DEVICES）。
  2. 加载 DiT 前把 state_dict 按 rank 切半（qkv 列并行 / out_proj 行并行 / fc1 列 / fc2 行），
     ComfyUI 会从张量形状自动推出 heads=28、ffn=7168，模型源码无需改动。
  3. out_proj 与 fc2 的输出是部分和，需要在模型内部做 all_reduce（默认 bf16）。
  4. 其余算子（norm / adaln / 首尾投影 / 文本编码 / VAE）保持复制，不参与并行。
"""

import gc
import json
import os
import time

import torch

import comfy.ldm.minimax.model as h3
import comfy.sd

TP_WORLD = int(os.environ.get("COMFY_TP_WORLD", "1"))
TP_RANK = int(os.environ.get("COMFY_TP_RANK", "0"))
TIMING_OUT = os.environ.get("COMFY_TP_TIMING_OUT", "")

_split_ids = set()
_stat = {"events": [], "step_calls": [], "step_ar": [], "all_reduce_calls": 0, "all_reduce_bytes": 0}


def _init_pg():
    if TP_WORLD <= 1:
        return
    import torch.distributed as dist
    if dist.is_initialized():
        return
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=TP_RANK,
        world_size=TP_WORLD,
        timeout=__import__("datetime").timedelta(minutes=30),
    )
    torch.cuda.set_device(0)
    if TP_WORLD == 2 and os.environ.get("COMFY_TP_P2P_WARMUP", "1") != "0":
        # 预热默认组的 P2P 通信子：首次 send/recv 会现场新建一个 2-rank 通信子（实测 ~2.5 s），
        # 那是解码末段的关键路径；挪到启动阶段只多几毫秒。
        try:
            from torch.distributed import P2POp
            t = torch.zeros(64, device="cuda")
            r = torch.zeros_like(t)
            works = dist.batch_isend_irecv([P2POp(dist.irecv, r, 1 - TP_RANK),
                                            P2POp(dist.isend, t, 1 - TP_RANK)])
            for w in works:
                w.wait()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[tp2] p2p warmup skipped: {e}", flush=True)


_init_pg()


# ---------------------------------------------------------------- 权重切分

def _split_tensor(t, mode):
    if mode == "row":
        return torch.chunk(t, TP_WORLD, dim=0)[TP_RANK].contiguous()
    if mode == "col":
        # 行并行只切输入维 K：[N,1] 的 per-row scale、[16] 的 codebook 必须原样保留
        if t.dim() == 1 or t.shape[-1] == 1:
            return t
        return torch.chunk(t, TP_WORLD, dim=1)[TP_RANK].contiguous()
    if mode == "qkv":
        # [q | k | v] 三段各自按头切，不能整块对半切
        return torch.cat([torch.chunk(p, TP_WORLD, dim=0)[TP_RANK]
                          for p in torch.chunk(t, 3, dim=0)], dim=0).contiguous()
    if mode == "gateup":
        # fc1 是 [gate | up] 两段，每段再切
        return torch.cat([torch.chunk(p, TP_WORLD, dim=0)[TP_RANK]
                          for p in torch.chunk(t, 2, dim=0)], dim=0).contiguous()
    raise ValueError(mode)


_LAYOUT = (
    (".attn.qkv_proj.", "qkv"),
    (".attn.out_proj.", "col"),
    (".attn.to_gate_compress.", "row"),
    (".mlp.fc1.", "gateup"),
    (".mlp.fc2.", "col"),
)


def _mode_for(key):
    # comfy_quant 是量化元数据；codebook 是全局固定的 16 值码本，两者都不能切
    if "comfy_quant" in key or "weight_codebook" in key:
        return None
    for suffix, mode in _LAYOUT:
        if suffix in key:
            return mode
    return None


# 行并行层（out_proj / fc2）的输入被沿 K 切成两半，两半各自算部分和再加起来。
# 实测（真实 w4a8 权重 + bf16 反量化路径）单层误差 rel≈2.9e-3，与 bf16 本身的
# 存储舍入同量级，不是 18.9 dB 差距的主因；而且 comfy_kitchen 的 A8 激活量化
# 是尺度不变的，帖子 2643 的 all_reduce(MAX) 共享 scale 在本内核上是空操作。
# 打开 COMFY_TP_ROW_BF16 后，把这两类层的量化配置标成
# full_precision_matrix_mult：ComfyUI 会反量化权重 + 走 bf16 GEMM（画质 +0.19 dB，
# 但每步 +0.87 s，故默认关闭）。量化配置是读文件时由 metadata 里的
# _quantization_metadata 展开成每层的 <layer>.comfy_quant 张量的，所以要在进
# load_diffusion_model_state_dict 之前改 metadata。
ROW_BF16 = os.environ.get("COMFY_TP_ROW_BF16") == "1"
_ROW_BF16_SUFFIX = (".attn.out_proj", ".mlp.fc2")


def _row_bf16_metadata(metadata):
    if not ROW_BF16 or not isinstance(metadata, dict):
        return metadata
    raw = metadata.get("_quantization_metadata")
    if not raw:
        return metadata
    qm = json.loads(raw)
    n = 0
    for layer, conf in qm.get("layers", {}).items():
        if layer.endswith(_ROW_BF16_SUFFIX):
            conf["full_precision_matrix_mult"] = True
            n += 1
    if n == 0:
        return metadata
    metadata = dict(metadata)
    metadata["_quantization_metadata"] = json.dumps(qm)
    print(f"[tp2] rank{TP_RANK} row-parallel layers -> bf16: {n}", flush=True)
    return metadata


def split_state_dict(sd):
    """rank 化：只切 DiT 的并行层，其余原样保留。"""
    if TP_WORLD <= 1 or os.environ.get("COMFY_TP_DUP") == "1":
        return sd
    qkv = sd.get("blocks.0.attn.qkv_proj.weight")
    hd = sd.get("blocks.0.attn.q_norm.weight")
    if qkv is None or hd is None:
        return sd
    heads = qkv.shape[0] // (3 * hd.shape[0])
    if heads % TP_WORLD != 0 or heads < TP_WORLD * 2:
        # 已经被切过（heads=28），不再二次切分
        return sd

    out = {}
    for k in list(sd.keys()):
        v = sd.pop(k)
        mode = _mode_for(k)
        if mode is None or not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        out[k] = _split_tensor(v, mode)

    sd.clear()
    gc.collect()
    print(f"[tp2] rank{TP_RANK} state_dict split: heads {heads}->{heads // TP_WORLD}, "
          f"keys {len(out)}", flush=True)
    return out


_orig_load = comfy.sd.load_diffusion_model_state_dict


def _tp_load_diffusion_model_state_dict(sd, model_options={}, metadata=None, disable_dynamic=False):
    if TP_WORLD > 1 and id(sd) not in _split_ids:
        _split_ids.add(id(sd))
        sd = split_state_dict(sd)
    metadata = _row_bf16_metadata(metadata)
    return _orig_load(sd, model_options=model_options, metadata=metadata,
                      disable_dynamic=disable_dynamic)


comfy.sd.load_diffusion_model_state_dict = _tp_load_diffusion_model_state_dict


# ---------------------------------------------------------------- LoRA 切分
# lora_up(B) 跟着权重的输出维切，lora_down(A) 只在行并行层（out_proj / fc2）切 K

def _lora_mode_for(key):
    # 兼容两种命名：lora_up/down 与 diffusers 的 lora_B(A)
    up = ".lora_up.weight" in key or ".lora_B.weight" in key
    down = ".lora_down.weight" in key or ".lora_A.weight" in key
    if not (up or down):
        return None
    for suffix, mode in _LAYOUT:
        if suffix in key:
            if up:
                # up/B 是 [out, rank]：只有输出维被切的层（qkv / fc1）才切它
                return None if mode == "col" else mode
            # down/A 是 [rank, in]：只有输入维被切的层（out_proj / fc2）才切它
            return "col" if mode == "col" else None
    return None


_orig_load_lora = comfy.sd.load_lora_for_models


def _tp_load_lora_for_models(model, clip, lora, strength_model, strength_clip,
                             **kwargs):
    if TP_WORLD > 1 and isinstance(lora, dict):
        new = {}
        nsplit = 0
        for k, v in lora.items():
            mode = _lora_mode_for(k)
            if mode is None or not isinstance(v, torch.Tensor):
                new[k] = v
            else:
                new[k] = _split_tensor(v, mode)
                nsplit += 1
        lora = new
        print(f"[tp2] rank{TP_RANK} lora split keys={nsplit}/{len(new)}", flush=True)
    return _orig_load_lora(model, clip, lora, strength_model, strength_clip, **kwargs)


comfy.sd.load_lora_for_models = _tp_load_lora_for_models


# ---------------------------------------------------------------- all_reduce

# 本机 RCCL 走 P2P/IPC，实测 ~4.7 GB/s，通信仍是 TP2 的主要开销，
# 所以 all_reduce 默认用 16 位（流量减半）。实测 bf16 与 fp32 的最终画质
# 只差 0.05 dB（18.89 / 18.84），而 fp32 每步要多 4 s，故 bf16 为准。
AR_DTYPE = os.environ.get("COMFY_TP_AR_DTYPE", "bf16")
_AR_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
# 仅用于计时拆分：跳过 all_reduce（输出无意义），量出每步纯计算时间
AR_SKIP = os.environ.get("COMFY_TP_AR_SKIP") == "1"

# MLP 沿序列切块，让「第 i 块的 all_reduce」与「第 i+1 块的 fc1/fc2 GEMM」在两条
# stream 上重叠。MLP 是逐 token 运算，切块与整块在数学上等价（只是激活量化按块
# 取 amax，数值差在 bf16 噪声底以下）。0/1 = 关闭，与旧路径完全一致。
MLP_CHUNKS = int(os.environ.get("COMFY_TP_MLP_CHUNKS", "0"))

# int8 定点全归约：PCIe Gen3 x8 已经跑满（RCCL 5.06 GB/s ≈ copy-engine 5.15 GB/s），
# 提速只能减字节。每行 absmax 跨 rank 取 MAX 得到共享 scale，量化到 ±63，两 rank
# 求和后落在 ±126，int8 累加不回绕（实测 NCCL int8 SUM 是 8 位回绕）。
AR_MODE = os.environ.get("COMFY_TP_AR_MODE", "bf16")
INT8_MAX = 63


def _ar_int8(t):
    import torch.distributed as dist
    s = t.shape
    f = t.reshape(-1, s[-1])
    amax = f.abs().amax(dim=1, keepdim=True)
    dist.all_reduce(amax, op=dist.ReduceOp.MAX)
    scale = (amax / INT8_MAX).clamp(min=1e-8)
    q = torch.round(f / scale).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    dist.all_reduce(q, op=dist.ReduceOp.SUM)
    _stat["all_reduce_bytes"] += q.numel()
    return (q.to(t.dtype) * scale).reshape(s)


def _ar(t):
    """t: cuda tensor；返回同 dtype 的跨 rank 求和结果。"""
    import torch.distributed as dist
    _stat["all_reduce_calls"] += 1
    if AR_MODE == "int8":
        return _ar_int8(t)
    dt = _AR_DTYPES[AR_DTYPE]
    buf = t.to(dt).contiguous()
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    _stat["all_reduce_bytes"] += buf.numel() * buf.element_size()
    return buf.to(t.dtype)


def tp_all_reduce(t):
    if TP_WORLD <= 1 or not torch.is_tensor(t) or not t.is_cuda or AR_SKIP:
        return t
    return _ar(t)


_comm_stream = None


def _get_comm_stream():
    global _comm_stream
    if _comm_stream is None:
        _comm_stream = torch.cuda.Stream()
    return _comm_stream


def _mlp_chunked(self, x):
    """分块 MLP：GEMM 在默认流，all_reduce 在 comm 流，两者重叠。"""
    S = x.shape[0]
    n = min(MLP_CHUNKS, max(1, S // 512))
    if n <= 1:
        return tp_all_reduce(_orig_mlp_forward(self, x))
    cs = (S + n - 1) // n
    cur = torch.cuda.current_stream()
    comm = _get_comm_stream()
    parts = []
    for i in range(0, S, cs):
        oc = _orig_mlp_forward(self, x[i:i + cs])
        ev = torch.cuda.Event()
        ev.record(cur)
        with torch.cuda.stream(comm):
            comm.wait_event(ev)
            parts.append(_ar(oc))
    ev_ready = torch.cuda.Event()
    ev_ready.record(comm)
    cur.wait_event(ev_ready)
    out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    return out.to(x.dtype)


_orig_attn_forward = h3.Attention.forward
_orig_mlp_forward = h3.MLP.forward


DBG_DIR = os.environ.get("COMFY_TP_DBG_DIR", "")
DBG_N = int(os.environ.get("COMFY_TP_DBG_N", "8"))
SHAPE_LOG = os.environ.get("COMFY_TP_SHAPE") == "1"
_dbg = {"attn": 0, "mlp": 0, "attn_in": 0, "mlp_in": 0}


def _dbg_dump(kind, t):
    if SHAPE_LOG and _dbg[kind] < DBG_N:
        print(f"[tp2] {kind} {tuple(t.shape)}", flush=True)
    if not DBG_DIR:
        _dbg[kind] += 1
        return t
    i = _dbg[kind]
    _dbg[kind] = i + 1
    if i < DBG_N:
        d = os.path.join(DBG_DIR, kind)
        os.makedirs(d, exist_ok=True)
        torch.save(t.detach().float().cpu(), os.path.join(d, f"{i:03d}.pt"))
    return t


def _attn_forward(self, x, rope_freqs=None, transformer_options={}):
    _dbg_dump("attn_in", x)
    return _dbg_dump("attn", tp_all_reduce(_orig_attn_forward(self, x, rope_freqs, transformer_options)))


def _mlp_forward(self, x):
    _dbg_dump("mlp_in", x)
    if TP_WORLD > 1 and MLP_CHUNKS > 1:
        out = _mlp_chunked(self, x)
    else:
        out = tp_all_reduce(_orig_mlp_forward(self, x))
    return _dbg_dump("mlp", out)


if TP_WORLD > 1 or DBG_DIR:
    h3.Attention.forward = _attn_forward
    h3.MLP.forward = _mlp_forward


# ---------------------------------------------------------------- 计时

PROF_DIR = os.environ.get("COMFY_TP_PROF_DIR", "")
PROF_SKIP = int(os.environ.get("COMFY_TP_PROF_SKIP", "1"))
PROF_STEPS = int(os.environ.get("COMFY_TP_PROF_STEPS", "2"))
# prompt 级：整条工作流（文本编码/采样/VAE/编码）打一条 trace，用于量固定开销
PROF_PROMPT = os.environ.get("COMFY_TP_PROF_SCOPE", "") == "prompt"
_prof = {"n": 0, "prompt": 0}


def _prof_start():
    if not PROF_DIR or PROF_PROMPT:
        return None
    i = _prof["n"]
    if not (PROF_SKIP <= i < PROF_SKIP + PROF_STEPS):
        return None
    try:
        from torch.profiler import profile, ProfilerActivity
        p = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False, with_stack=False)
        p.__enter__()
        return p
    except Exception as e:
        print(f"[tp2] profiler start failed: {e}", flush=True)
        return None


def _prof_stop(p):
    i = _prof["n"]
    try:
        p.__exit__(None, None, None)
        os.makedirs(PROF_DIR, exist_ok=True)
        path = os.path.join(PROF_DIR, f"trace_r{TP_RANK}_{i}.json")
        p.export_chrome_trace(path)
        print(f"[tp2] trace -> {path}", flush=True)
    except Exception as e:
        print(f"[tp2] profiler export failed: {e}", flush=True)


def _install_timing():
    if not TIMING_OUT:
        return
    import comfy.model_base as mb

    orig_apply = mb.BaseModel.apply_model

    def timed_apply(self, *args, **kwargs):
        now = time.time()
        ar0 = _stat["all_reduce_calls"]
        prof = _prof_start()
        try:
            return orig_apply(self, *args, **kwargs)
        finally:
            if prof is not None:
                _prof_stop(prof)
            _prof["n"] += 1
            _stat["step_calls"].append(now)
            _stat["step_ar"].append(_stat["all_reduce_calls"] - ar0)
            _write_stat(now)

    mb.BaseModel.apply_model = timed_apply


def _write_stat(now=None):
    if not TIMING_OUT:
        return
    data = dict(_stat)
    if now is not None:
        data["last"] = now
    with open(TIMING_OUT, "w") as f:
        json.dump(data, f)


_install_timing()


# ---------------------------------------------------------------- 加载耗时拆分
# 目的：采样前那 58 s 里，DiT 与文本编码器的权重加载/CPU 转换各占多少，用实测而非猜测。
# COMFY_TP_LOADTIME=1 开启，结果写到 timing 同目录的 loadtime_r{rank}.json。

LOADTIME = os.environ.get("COMFY_TP_LOADTIME") == "1"
_load_log = []


def _install_loadtime():
    if not (LOADTIME and TIMING_OUT):
        return
    import comfy.model_management as mm

    orig = mm.load_models_gpu
    import comfy.sd as csd
    orig_lora = csd.load_lora_for_models
    orig_enc = csd.CLIP.encode_from_tokens_scheduled
    path = os.path.join(os.path.dirname(TIMING_OUT), f"loadtime_r{TP_RANK}.json")

    def _log(rec):
        try:
            _load_log.append(rec)
            with open(path, "w") as f:
                json.dump(_load_log, f)
            print(f"[tp2] load {rec['dur']}s {rec['models']}", flush=True)
        except Exception:
            pass

    def timed_load(models, *args, **kwargs):
        t0 = time.time()
        try:
            return orig(models, *args, **kwargs)
        finally:
            try:
                names = []
                for m in models:
                    mod = getattr(m, "model", None)
                    names.append(type(mod).__name__ if mod is not None else type(m).__name__)
                _log({"t0": round(t0, 3), "dur": round(time.time() - t0, 2), "models": names})
            except Exception:
                pass

    def timed_lora(*args, **kwargs):
        t0 = time.time()
        try:
            return orig_lora(*args, **kwargs)
        finally:
            _log({"t0": round(t0, 3), "dur": round(time.time() - t0, 2), "models": ["LoRA"]})

    def timed_enc(*args, **kwargs):
        t0 = time.time()
        try:
            return orig_enc(*args, **kwargs)
        finally:
            _log({"t0": round(t0, 3), "dur": round(time.time() - t0, 2), "models": ["TE_encode"]})

    mm.load_models_gpu = timed_load
    csd.load_lora_for_models = timed_lora
    csd.CLIP.encode_from_tokens_scheduled = timed_enc


_install_loadtime()


# ---------------------------------------------------------------- 整条工作流 trace
# 目的：采样循环里 GPU 空隙实测只占 0.2%，真正的大头在采样之外（文本编码、VAE 解码、
# 编码出片），用 PROMPT 级 trace 量整条工作流里 GPU 忙 vs 等主机的比例。

def _install_exec_hook():
    if not PROF_PROMPT:
        return
    try:
        import execution
    except Exception as e:
        print(f"[tp2] exec hook skipped: {e}", flush=True)
        return
    orig = execution.PromptExecutor.execute

    def execute(self, *a, **kw):
        p = None
        if _prof["prompt"] == 0:
            try:
                from torch.profiler import profile, ProfilerActivity
                p = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            record_shapes=False, with_stack=False)
                p.__enter__()
            except Exception as e:
                print(f"[tp2] prompt profiler failed: {e}", flush=True)
                p = None
        try:
            return orig(self, *a, **kw)
        finally:
            if p is not None:
                _prof["prompt"] += 1
                _prof_stop(p)

    execution.PromptExecutor.execute = execute


_install_exec_hook()
print(f"[tp2] world={TP_WORLD} rank={TP_RANK} enrolled", flush=True)


# ---------------------------------------------------------------- 出片跨卡拆分：VAE 解码

# 背景：采样后的 34 s 里 VAE 解码占 ~11.9 s。H3 视频 VAE 的 decode 本身就在时间轴上
# 分块（tokens_chunk_size=5 latent 帧 + token_overlap=2 帧上下文，块间用 frame_overlap
# 帧线性混合），所以按「块」把解码分给两张卡是与单卡路径逐位等价的，不需要自造 overlap：
#   rank0 解块 [0, c]，写帧 [0, offs[c+1])
#   rank1 解块 [c, n)，写帧 [offs[c+1], total)
# 接缝块 c 两边都解一遍：rank1 需要它的尾段分片（单卡里的 dec_overlap）来混合自己第一块
# 的前 frame_overlap 帧 —— 这正是单卡路径本来就存在的依赖，因此结果不变。代价是每卡
# 多算 1/n 的块，n=7 时每卡解 4/7 的块，解码时间约减半。
# 与现有 TP2 逻辑不冲突：collective 全在模型前向之外，只用默认组做一次点对点交换，
# 不与 all_reduce 的 comm stream 相交；WORLD=1 或开关关闭时直接走原生 VAEDecode。
VAE_SPLIT = os.environ.get("COMFY_TP_VAE_SPLIT") == "1"
VAE_LOG = os.environ.get("COMFY_TP_VAE_LOG") == "1"
# 交换前心跳的等待上限：宁可快速失败，也不要因为对端没来而挂到 NCCL 的 30 min 超时
VAE_SPLIT_TIMEOUT = float(os.environ.get("COMFY_TP_VAE_TIMEOUT", "60"))


def _vlog(msg):
    if VAE_LOG:
        print(f"[tp2] rank{TP_RANK} {msg}", flush=True)


def _chunk_offsets(fs, z_len, pad_tokens, num_chunks):
    """按 decode_temporal 的帧计划算每块的写帧偏移，以及输出总帧数。"""
    chunk_dec = fs.tokens_chunk_size * fs.vae_ratio_t
    split_count = int(fs.token_drop > 0) + 1
    z_len_padded = z_len + pad_tokens
    counts, tail = [], 0
    for i in range(num_chunks):
        t0 = i * fs.tokens_chunk_size
        t1 = t0 + fs.tokens_chunk_size + fs.token_overlap
        clip_tokens = max(0, min(t1, z_len_padded) - min(t0, z_len_padded))
        clip_frames = clip_tokens * fs.vae_ratio_t
        for j in range(split_count):
            f0 = j * chunk_dec
            f1 = min(f0 + chunk_dec, clip_frames)
            n = max(0, f1 - f0 - fs.frame_pre_padding)
            if j == 0:
                counts.append(n)
            else:
                tail = n
    total = sum(counts) + tail - fs._decode_temporal_pad_frames(z_len_padded, pad_tokens)
    offs = [0]
    for n in counts:
        offs.append(offs[-1] + n)
    return offs, total


def _copy_part(fs, part, dec, off, total):
    """同 decode_temporal.write_part：定稿到 [0,1] 后按帧偏移写进输出缓冲，末尾截断。"""
    if part.shape[2] <= 0:
        return
    part = fs._finalize_pixels(part)
    n = min(part.shape[2], max(0, total - off))
    if n > 0:
        dec[:, :, off:off + n].copy_(part[:, :, :n])


def _decode_chunk_range(fs, z, dec, offs, total, cs, ce, w_lo, w_hi, w_tail):
    """解块 [cs, ce)，只把 [w_lo, w_hi) 的块写进 dec；w_tail 时补写末尾 overlap 段。"""
    chunk_dec = fs.tokens_chunk_size * fs.vae_ratio_t
    split_count = int(fs.token_drop > 0) + 1
    dec_overlap = None
    for i in range(cs, ce):
        t0 = i * fs.tokens_chunk_size
        t1 = t0 + fs.tokens_chunk_size + fs.token_overlap
        clip_z = z[:, :, t0:t1]
        clip_dec = fs._adaptive_decode(clip_z)
        for j in range(split_count):
            f0 = j * chunk_dec
            f1 = min(f0 + chunk_dec, clip_dec.shape[2])
            part = clip_dec[:, :, f0:f1][:, :, fs.frame_pre_padding:]
            if j == 0:
                if dec_overlap is not None:
                    part = fs.blend(dec_overlap, part, fs.frame_overlap, dim=-3)
                    dec_overlap = None
                if w_lo <= i < w_hi:
                    _copy_part(fs, part, dec, offs[i], total)
            else:
                dec_overlap = part.contiguous()
        del clip_dec, clip_z
    if w_tail and dec_overlap is not None:
        _copy_part(fs, dec_overlap, dec, offs[len(offs) - 1], total)


def _exchange_frames(dec, split_f):
    """两 rank 互换帧区间：各自把自己那半搬到对端。

    必须用 batch_isend_irecv 把送/收合成一次 group call：普通 isend/irecv 会落到同一条
    NCCL stream 上，先排队的 recv kernel 会一直等数据，后面的 send 永远发不出去（卡死）。
    """
    import torch.distributed as dist
    from torch.distributed import P2POp
    b, ch, f, h, w = dec.shape
    peer = 1 - TP_RANK
    # gpu-only 时输出缓冲本来就在卡上，别再白拷一次
    dev = dec.device if dec.is_cuda else torch.device("cuda")
    if TP_RANK == 0:
        send = dec[:, :, :split_f].to(dev).contiguous()
        recv = torch.empty((b, ch, f - split_f, h, w), dtype=dec.dtype, device=dev)
    else:
        send = dec[:, :, split_f:].to(dev).contiguous()
        recv = torch.empty((b, ch, split_f, h, w), dtype=dec.dtype, device=dev)
    works = dist.batch_isend_irecv([P2POp(dist.irecv, recv, peer), P2POp(dist.isend, send, peer)])
    for wk in works:
        wk.wait()
    dst = dec[:, :, split_f:] if TP_RANK == 0 else dec[:, :, :split_f]
    dst.copy_(recv.to(dec.device))


def _split_plan(fs, shape):
    """本地判定能否拆分，并给出分块计划。返回 (pad, n, offs, total, seam, split_f) 或 None。

    两侧必须由同一份 latent 形状判成同一结论（见 _heartbeat），否则会一侧进 collective、
    另一侧不进。这里顺手和上游 decode_output_shape 对账：上游改常量时立刻报错，而不是静默错帧。
    """
    if not (getattr(fs, "comfy_has_chunked_io", False) and hasattr(fs, "tokens_chunk_size")):
        return None
    t_lat = shape[2]
    pad_tokens, num_chunks = fs._decode_temporal_chunks(t_lat)
    offs, total = _chunk_offsets(fs, t_lat, pad_tokens, num_chunks)
    c = num_chunks // 2
    if num_chunks < 3 or c + 1 >= num_chunks or not (0 < offs[c + 1] < total):
        return None
    ref_total = fs.decode_output_shape(shape)[2]
    if total != ref_total:
        raise RuntimeError(f"tp2 vae split plan drifted from upstream: {total} != {ref_total} "
                           f"(t_lat={t_lat}, chunks={num_chunks}, pad={pad_tokens})")
    return pad_tokens, num_chunks, offs, total, c, offs[c + 1]


def _heartbeat(ok):
    """交换前心跳：两 rank 用同一份判据决定「拆分」还是「整片解码」。

    用短超时的 async all_reduce，把「对端没来 / 对端判定不可拆」变成快速失败或一致回退，
    而不是挂到 NCCL 的 30 min 超时。返回 True 表示两侧都同意拆分。
    """
    import torch.distributed as dist
    t = torch.tensor([1 if ok else 0], dtype=torch.int32, device="cuda")
    work = dist.all_reduce(t, op=dist.ReduceOp.MIN, async_op=True)
    t_end = time.time() + VAE_SPLIT_TIMEOUT
    while not work.is_completed():
        if time.time() > t_end:
            raise RuntimeError(
                f"tp2 vae split heartbeat timeout ({VAE_SPLIT_TIMEOUT:.0f}s): "
                f"peer rank did not reach the decode node with the same plan")
        time.sleep(0.05)
    work.wait()
    return bool(t.item())


def _decode_split(vae, fs, latent, plan):
    import comfy.model_management as mm
    import comfy.model_prefetch

    pad_tokens, num_chunks, offs, total, c, split_f = plan
    t_all = time.time()
    with mm.cuda_device_context(vae.device):
        memory_used = vae.memory_used_decode(latent.shape, vae.vae_dtype)
        with comfy.model_prefetch.pause_malloc_graph():
            mm.load_models_gpu([vae.patcher], memory_required=memory_used,
                               force_full_load=vae.disable_offload)
        t_load = time.time()

        z = latent.to(device=vae.device, dtype=vae.vae_dtype)
        lm = fs.latents_mean.view(1, -1, 1, 1, 1).to(z)
        ls = fs.latents_std.view(1, -1, 1, 1, 1).to(z)
        z = z * ls + lm
        with comfy.model_prefetch.pause_malloc_graph():
            out_buf = torch.empty(fs.decode_output_shape(z.shape), device=vae.output_device,
                                  dtype=vae.vae_output_dtype())
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        # 接缝块 c：rank0 写 [0, c]，rank1 从 c 开始解但只写 [c+1, n)，外加末尾 overlap 段。
        t0 = time.time()
        if TP_RANK == 0:
            _decode_chunk_range(fs, z, out_buf, offs, total, 0, c + 1, 0, c + 1, False)
        else:
            _decode_chunk_range(fs, z, out_buf, offs, total, c, num_chunks, c + 1, num_chunks, True)
        t1 = time.time()
        _exchange_frames(out_buf, split_f)
        t2 = time.time()
        _vlog(f"vae split chunks={num_chunks} seam={c} "
              f"own={split_f if TP_RANK == 0 else total - split_f}/{total} "
              f"load={t_load - t_all:.2f}s decode={t1 - t0:.2f}s "
              f"exchange={t2 - t1:.2f}s total={t2 - t_all:.2f}s")

    return out_buf.to(vae.output_device).movedim(1, -1)


def _vae_decode_maybe_split(vae, latent):
    fs = getattr(vae, "first_stage_model", None)
    if TP_WORLD <= 1 or not VAE_SPLIT:
        return vae.decode(latent)
    usable = latent.ndim == 5 and latent.shape[0] == 1
    plan = _split_plan(fs, latent.shape) if usable else None
    if not _heartbeat(plan is not None):
        _vlog("vae split -> full decode (local plan unusable or peer disagreed)")
        return vae.decode(latent)
    return _decode_split(vae, fs, latent, plan)


class TP2VAEDecodeSplit:
    """VAE 解码跨卡拆分。COMFY_TP_VAE_SPLIT=1 且 WORLD>1 时按时间块分摊，否则等同 VAEDecode。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "tp2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 节点内含跨 rank 的点对点交换，任何一侧命中缓存都会让另一侧卡死，强制每次都执行
        return float("NaN")

    def decode(self, vae, samples):
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        images = _vae_decode_maybe_split(vae, latent)
        if len(images.shape) == 5:  # Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images,)


# ---------------------------------------------------------------- 调试节点：导出 latent

class TP2DumpLatent:
    """把采样结果的原生 latent 存成 .pt，用于比对单卡 / TP2 两个 rank 的数值一致性。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "path": ("STRING", {"default": "/tmp/tp2_latent.pt"})}}

    RETURN_TYPES = ()
    FUNCTION = "dump"
    CATEGORY = "tp2"
    OUTPUT_NODE = True

    def dump(self, samples, path):
        s = samples["samples"]
        streams = s.unbind() if getattr(s, "is_nested", False) else [s]
        data = {f"stream{i}": t.detach().float().cpu() for i, t in enumerate(streams)}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(data, path)
        print(f"[tp2] dumped latent {[tuple(v.shape) for v in data.values()]} -> {path}", flush=True)
        return {}


NODE_CLASS_MAPPINGS = {"TP2DumpLatent": TP2DumpLatent, "TP2VAEDecodeSplit": TP2VAEDecodeSplit}
NODE_DISPLAY_NAME_MAPPINGS = {"TP2DumpLatent": "TP2 Dump Latent",
                              "TP2VAEDecodeSplit": "TP2 VAE Decode (split across cards)"}


# ---------------------------------------------------------------- post 阶段节点计时
# COMFY_TP_NODELOG=1 时把 post 拆成「视频解码 / 音频解码 / 封装编码」三段，用来判断下一步
# 该优化谁。挂在 VAE.decode 与 Video.save_to 上（普通方法，比 V3 节点的 execute 稳）。
NODELOG = os.environ.get("COMFY_TP_NODELOG") == "1"


def _install_node_log():
    if not (NODELOG and TP_WORLD > 1):
        return
    try:
        import comfy.sd as csd
        from comfy_api.latest._input_impl import video_types as vt

        orig_decode = csd.VAE.decode

        def timed_decode(self, *a, **kw):
            t0 = time.time()
            try:
                return orig_decode(self, *a, **kw)
            finally:
                print(f"[tp2] rank{TP_RANK} vae.decode {type(self.first_stage_model).__name__} "
                      f"{time.time() - t0:.2f}s", flush=True)

        csd.VAE.decode = timed_decode

        # 不同节点走不同实现：SaveVideo 用 VideoFromComponents，带图列表的用 VideoFromList
        for cls in (vt.VideoFromComponents, vt.VideoFromList, vt.VideoFromFile):
            orig_save = cls.save_to

            def timed_save(self, *a, __o=orig_save, __n=cls.__name__, **kw):
                t0 = time.time()
                try:
                    return __o(self, *a, **kw)
                finally:
                    print(f"[tp2] rank{TP_RANK} {__n}.save_to {time.time() - t0:.2f}s", flush=True)

            cls.save_to = timed_save
    except Exception as e:
        print(f"[tp2] node log install skipped: {e}", flush=True)


_install_node_log()
