#!/usr/bin/env python
"""Correctness + benchmark harness for the int8 flash-attention prototype (gfx1100).

usage:
  python harness.py --check              # S=1024 correctness vs torch fp32 reference
  python harness.py --bench              # S=14869 benchmark (warmup + 3 timed iters)
  python harness.py --check --bench      # both
"""
import argparse
import ctypes
import os
import sys

import torch
import torch.nn.functional as F

D = 128
LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libattn_int8.so")

lib = ctypes.CDLL(LIB)
lib.hipattn_quantize.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.hipattn_forward.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib.hipattn_last_error.restype = ctypes.c_int


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


class Bufs:
    def __init__(self, H, S):
        self.H, self.S = H, S
        self.Spad = ((S + 127) // 128) * 128
        dev = "cuda"
        self.qq = torch.empty(H * self.Spad * D, dtype=torch.int8, device=dev)
        self.kq = torch.empty_like(self.qq)
        self.vq = torch.empty_like(self.qq)
        self.qsc = torch.empty(H * self.Spad, dtype=torch.float32, device=dev)
        self.ksc = torch.empty_like(self.qsc)
        self.vsc = torch.empty(H * D, dtype=torch.float32, device=dev)
        self.vcolmax = torch.zeros(H * D, dtype=torch.float32, device=dev)
        self.out = torch.empty(H, S, D, dtype=torch.bfloat16, device=dev)

    def quantize(self, q, k, v, stream):
        lib.hipattn_quantize(ptr(q), ptr(k), ptr(v), self.H, self.S, self.Spad,
                             ptr(self.qq), ptr(self.qsc), ptr(self.kq), ptr(self.ksc),
                             ptr(self.vq), ptr(self.vsc), ptr(self.vcolmax), stream)

    def forward(self, stream):
        lib.hipattn_forward(ptr(self.qq), ptr(self.qsc), ptr(self.kq), ptr(self.ksc),
                            ptr(self.vq), ptr(self.vsc), ptr(self.out),
                            self.H, self.S, self.Spad, stream)


def cur_stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def run(q, k, v, bufs, with_quant=True):
    st = cur_stream()
    if with_quant:
        bufs.quantize(q, k, v, st)
    bufs.forward(st)
    err = lib.hipattn_last_error()
    assert err == 0, "hip error %d" % err
    return bufs.out


def check(H=28, S=1024, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(H, S, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(H, S, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(H, S, D, dtype=torch.bfloat16, device="cuda")
    bufs = Bufs(H, S)
    out = run(q, k, v, bufs)
    torch.cuda.synchronize()
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    o = out.float()
    diff = (o - ref).abs()
    denom = ref.abs().clamp_min(1e-3)
    rel = (diff / denom)
    print("=== correctness  H=%d S=%d D=%d ===" % (H, S, D))
    print("  max |err|          : %.5f" % diff.max().item())
    print("  mean |err|         : %.5f" % diff.mean().item())
    print("  max rel err        : %.4f (%.2f%%)" % (rel.max().item(), 100 * rel.max().item()))
    print("  mean rel err       : %.4f (%.2f%%)" % (rel.mean().item(), 100 * rel.mean().item()))
    print("  rel l2 (||e||/||r||): %.4f (%.2f%%)" % (
        (diff.norm() / ref.norm()).item(), 100 * (diff.norm() / ref.norm()).item()))
    print("  ref abs mean       : %.5f" % ref.abs().mean().item())
    return rel.mean().item()


def bench(H=28, S=14869, iters=3, warmup=2):
    torch.manual_seed(0)
    q = torch.randn(H, S, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(H, S, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(H, S, D, dtype=torch.bfloat16, device="cuda")
    bufs = Bufs(H, S)
    st = cur_stream()
    flops = 4.0 * H * S * S * D

    def timed(fn, n):
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        for _ in range(n):
            fn()
        ev1.record()
        torch.cuda.synchronize()
        return ev0.elapsed_time(ev1) / n

    # warmup
    for _ in range(warmup):
        run(q, k, v, bufs)
    torch.cuda.synchronize()

    t_quant = timed(lambda: bufs.quantize(q, k, v, st), iters)
    t_fwd = timed(lambda: bufs.forward(st), iters)
    t_all = timed(lambda: run(q, k, v, bufs), iters)

    print("=== benchmark  H=%d S=%d D=%d  (warmup=%d, timed=%d) ===" % (H, S, D, warmup, iters))
    print("  quantize          : %8.3f ms" % t_quant)
    print("  attention kernel  : %8.3f ms  -> %7.1f TFLOPS" % (t_fwd, flops / (t_fwd * 1e-3) / 1e12))
    print("  quantize + kernel : %8.3f ms  -> %7.1f TFLOPS" % (t_all, flops / (t_all * 1e-3) / 1e12))
    print("  gate < 25 ms      : %s (kernel-only %s)" % (
        "PASS" if t_all < 25 else "FAIL", "PASS" if t_fwd < 25 else "FAIL"))
    print("  flops             : %.4f TFLOP" % (flops / 1e12))
    return t_fwd, t_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--S", type=int, default=1024)
    a = ap.parse_args()
    if not a.check and not a.bench:
        a.check = a.bench = True
    print("device:", torch.cuda.get_device_name(0))
    if a.check:
        check(S=a.S)
    if a.bench:
        bench()
