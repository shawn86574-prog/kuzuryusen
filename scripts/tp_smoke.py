import os, time
import torch, torch.distributed as dist

rank = int(os.environ["COMFY_TP_RANK"]); world = int(os.environ["COMFY_TP_WORLD"])
dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world)
t = torch.ones(4096, 5376, dtype=torch.float32, device="cuda")
if rank == 1:
    t.fill_(2)
dist.all_reduce(t, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
print(f"rank{rank} ok sum={t.mean().item():.1f} device={t.device} "
      f"visible={torch.cuda.device_count()}", flush=True)
# 带宽测试
x = torch.empty(8192, 5376, dtype=torch.float32, device="cuda")
for _ in range(5):
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
dt = (time.time() - t0) / 20
n = x.numel() * 4
print(f"rank{rank} all_reduce {n/2**20:.1f} MiB in {dt*1000:.1f} ms -> {n/dt/1e9:.2f} GB/s", flush=True)
dist.destroy_process_group()
