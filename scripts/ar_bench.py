#!/usr/bin/env python3
"""测量 all_reduce 的真实带宽与「每次调用固定开销」：
   - 逐层 114 次小 all_reduce（与模型内一致）
   - 合并成 6 次大 all_reduce（同样的字节总量）
   对比两者，可区分「带宽不足」与「调用次数太多」。
"""
import os, time, sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HID = 5376
CASES = [int(x) for x in os.environ.get("AR_S", "14430,37296,74880").split(",")]   # 默认 832×480 / 1344×768 / 1920×1088 的 S


def worker(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, init_method=f"env://")
    dev = torch.device(f"cuda:{rank}")
    res = []
    for S in CASES:
        t = torch.randn(S, HID, device=dev, dtype=torch.bfloat16)
        n_bytes = t.numel() * t.element_size()

        # 1) 114 次逐层 AR
        for _ in range(1):
            dist.all_reduce(t)
        torch.cuda.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        for _ in range(114):
            dist.all_reduce(t)
        torch.cuda.synchronize(); dist.barrier()
        t_small = time.perf_counter() - t0

        # 2) 合并成 6 次大 AR（每块 19 层）
        big = torch.randn(S * 19, HID, device=dev, dtype=torch.bfloat16)
        dist.all_reduce(big)
        torch.cuda.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        for _ in range(6):
            dist.all_reduce(big)
        torch.cuda.synchronize(); dist.barrier()
        t_big = time.perf_counter() - t0
        del big

        total = 114 * n_bytes
        res.append((S, total / 2**30, t_small, total / t_small / 1e9, t_big, total / t_big / 1e9))
    if rank == 0:
        print(f"{'S':>7} {'payload/step':>12} {'114 次分段':>11} {'带宽':>9} {'6 次合并':>10} {'带宽':>9}")
        for S, gb, ts, bs, tb, bb in res:
            print(f"{S:>7} {gb:>11.2f}GiB {ts:>10.2f}s {bs:>8.2f}GB/s {tb:>9.2f}s {bb:>8.2f}GB/s")
    dist.destroy_process_group()


if __name__ == "__main__":
    port = 29531
    mp.spawn(worker, args=(2, port), nprocs=2, join=True)
