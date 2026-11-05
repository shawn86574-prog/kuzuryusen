#!/usr/bin/env python3
"""实测 MiniMax-H3 video VAE 的 latent->pixel 时间映射，并验证按 chunk 边界
切片解码能否与整段解码逐帧一致（用于 TP2 VAE 拆分节点）。"""
import json
import os
import sys
import time
import types

import torch

sys.path.insert(0, os.environ.get("COMFY_DIR", "/opt/ComfyUI"))
import comfy.sd
import comfy.utils

VAE_PATH = os.path.join(os.environ.get("COMFY_DIR", "/opt/ComfyUI"),
                            "models/vae/Minimax_h3_video_vae_fp16.safetensors")
OUT = "vae_probe_result.json"

vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE_PATH))
fs = vae.first_stage_model


def _row_one_tile(self, z_row, x_idx, x_len):
    # 探测时强制每行单 tile 解码，避免空闲显存大时 _decode_tile_row 把多个 tile 拼成
    # 一次超大序列调用（23 GB 峰值 OOM）
    for j_pos, j_len in zip(x_idx, x_len):
        yield self._decode_pixels(z_row[..., j_pos // self.vae_ratio:(j_pos + j_len) // self.vae_ratio])


fs._decode_tile_row = types.MethodType(_row_one_tile, fs)
params = dict(vae_ratio=fs.vae_ratio, vae_ratio_t=fs.vae_ratio_t, clip_length=fs.clip_length,
              token_drop=fs.token_drop, tokens_chunk_size=fs.tokens_chunk_size,
              token_overlap=fs.token_overlap, frame_overlap=fs.frame_overlap,
              frame_pre_padding=fs.frame_pre_padding, tile_size=fs.tile_size,
              temporal_compression_decode=vae.temporal_compression_decode())
print("PARAMS", json.dumps(params), flush=True)

g = torch.Generator().manual_seed(1234)
Z = torch.randn(1, 24, 37, 30, 52, generator=g)


def dec(z):
    t0 = time.time()
    out = vae.decode(z)
    torch.cuda.synchronize()
    return out.float(), round(time.time() - t0, 2)


def cmp(a, b):
    d = (a - b).abs()
    return dict(max=round(d.max().item(), 6), mean=round(d.mean().item(), 6))


res = {"params": params, "map": [], "windows": [], "prefix": [], "blend": []}

# 1) latent 长度 -> 像素帧数（时间映射）
for T in [2, 3, 7, 12, 17, 22, 27, 32, 37]:
    out, dt = dec(Z[:, :, :T])
    res["map"].append({"T_lat": T, "frames": out.shape[2], "s": dt})
    print("MAP", res["map"][-1], flush=True)
    del out

full, t_full = dec(Z)
F = full.shape[2]
res["full"] = {"frames": F, "s": t_full, "shape": list(full.shape)}
print("FULL", res["full"], flush=True)

# 2) 从 chunk 边界（5 的倍数）起解码：帧数 + 与整段解码的对齐/数值差
for a in [10, 15, 20, 25]:
    out, dt = dec(Z[:, :, a:])
    g0 = (F - out.shape[2])
    rec = {"a": a, "frames": out.shape[2], "s": dt,
           "expect_g0_17c": 17 * (a // fs.tokens_chunk_size), "derived_g0": g0}
    if 0 <= g0 and g0 + out.shape[2] <= F:
        rec["diff_vs_full"] = cmp(out, full[g0:g0 + out.shape[2]])
    rec["diff_vs_17c"] = cmp(out[:min(out.shape[2], F - 17 * (a // 5))],
                             full[17 * (a // 5):17 * (a // 5) + min(out.shape[2], F - 17 * (a // 5))])
    res["windows"].append(rec)
    print("WIN", rec, flush=True)
    del out

# 3) 前缀窗口 [0, 5m+2)：产出 17m+5 帧，前 17m 帧应与整段一致；尾部 5 帧是未混合的 overlap
for m in [2, 3, 4]:
    L = 5 * m + 2
    pre, dt = dec(Z[:, :, :L])
    keep = 17 * m
    rec = {"m": m, "L": L, "frames": pre.shape[2], "s": dt, "keep": keep,
           "head_exact": cmp(pre[:keep], full[:keep]),
           "tail5_vs_full": cmp(pre[keep:keep + 5], full[keep:keep + 5])}
    # 4) 用 rank1 前 5 帧与 rank0 的 overlap 帧混合，应等于整段解码的对应 5 帧
    a = 5 * m
    win, _ = dec(Z[:, :, a:])
    g0 = F - win.shape[2]
    if g0 == keep and win.shape[2] >= 5:
        blend = fs.blend(pre[keep:keep + 5], win[:5], fs.frame_overlap, dim=0)
        res["blend"].append({"m": m, "g0": g0, "diff_vs_full": cmp(blend, full[keep:keep + 5]),
                             "unblended_vs_full": cmp(win[:5], full[keep:keep + 5])})
        print("BLEND", res["blend"][-1], flush=True)
    res["prefix"].append(rec)
    print("PRE", rec, flush=True)
    del pre, win

json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False)
print("WROTE", OUT, flush=True)
