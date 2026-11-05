"""行并行层在 bf16 路径（LoRA 已把权重反量化成 bf16）下的切分误差。

LoRA 生效时 ComfyUI 会 dequant 成 bf16 做 GEMM，A8 激活量化不在执行路径里。
本脚本用真实 w4a8 权重反量化成 bf16，比较：
  full  = 整 K 一次 bf16 GEMM（单卡做法）
  tp    = 两半各算 bf16 部分和，再 bf16 all_reduce（TP2 做法）
  tp_f32= 两半各算，fp32 累加（诊断：部分和 bf16 舍入占多少）
"""
import json, os, struct
import torch

P = os.path.join(os.environ.get("COMFY_DIR", "/opt/ComfyUI"),
                "models/diffusion_models/minimax_h3_fl2va_pruned-w4a8_convrot_pruned.safetensors")

def load(prefix):
    with open(P, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__" or not k.startswith(prefix):
            continue
        s, e = v["data_offsets"]
        dt = {"I8": torch.int8, "F32": torch.float32,
              "F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16}[v["dtype"]]
        with open(P, "rb") as f:
            f.seek(base + s)
            out[k.split(".")[-1]] = torch.frombuffer(
                bytearray(f.read(e - s)), dtype=dt).reshape(v["shape"]).clone()
    return out

from comfy_kitchen.tensor.w4a8_int8 import dequantize_w4a8_int8_weight

def deq(t):
    return dequantize_w4a8_int8_weight(
        t["weight"].cuda(), t["weight_s_rel"].cuda(), t["weight_s_channel"].cuda(),
        codebook=t["weight_codebook"].cuda(), group_size=16, convrot_groupsize=256,
        output_dtype=torch.bfloat16)

def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()

torch.manual_seed(0)
for name, prefix, S in (("out_proj", "blocks.0.attn.out_proj", 519),
                        ("fc2", "blocks.0.mlp.fc2", 519)):
    W = deq(load(prefix))          # [N, K]
    N, K = W.shape
    for xscale in (1.0, 3.0):
        x = (torch.randn(S, K, device="cuda") * xscale).to(torch.bfloat16)
        full = (x @ W.T).to(torch.bfloat16)
        half = K // 2
        pa = (x[:, :half] @ W[:, :half].T).to(torch.bfloat16)
        pb = (x[:, half:] @ W[:, half:].T).to(torch.bfloat16)
        tp = (pa + pb).to(torch.bfloat16)
        tp32 = (pa.float() + pb.float()).to(torch.bfloat16)
        # 理想的 fp32 部分和（不受 bf16 输出舍入影响）
        full32 = (x.float() @ W.float().T)
        tp_parts32 = (x[:, :half].float() @ W[:, :half].float().T) + \
                     (x[:, half:].float() @ W[:, half:].float().T)
        print(f"{name:9s} K={K:5d} S={S} x*{xscale}: "
              f"tp(bf16 parts)={rel(tp, full):.3e} "
              f"tp(fp32 parts)={rel(tp32, full):.3e} "
              f"math(fp32 both)={rel(tp_parts32, full32):.3e} "
              f"|out|={full.float().abs().mean().item():.3f}")
