import torch, time
print("peer 0->1:", torch.cuda.can_device_access_peer(0,1), "1->0:", torch.cuda.can_device_access_peer(1,0))
torch.cuda.set_device(0)
a = torch.empty(256*1024*1024//4, dtype=torch.float32, device="cuda:0")  # 256MB
b = torch.empty_like(a, device="cuda:1")
a.fill_(1.0)
for _ in range(3): b.copy_(a, non_blocking=True)
torch.cuda.synchronize()
t0=time.time()
for _ in range(20): b.copy_(a, non_blocking=True)
torch.cuda.synchronize()
dt=(time.time()-t0)/20
print(f"copy 256MB 0->1: {dt*1000:.1f} ms -> {256/1024/dt:.2f} GB/s")
