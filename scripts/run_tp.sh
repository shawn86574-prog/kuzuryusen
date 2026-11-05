#!/usr/bin/env bash
# 启动 TP 实例：run_tp.sh <world> <base_port> <tag>
set -u
WORLD=${1:-1}
BASE_PORT=${2:-8191}
TAG=${3:-run}
COMFY=${COMFY_DIR:-/home/jay/ComfyUI}
PY=${PYTHON:-python3}
OUT=${OUT_DIR:-$HOME/tp2-logs}/$TAG
# --mmap-torch-files 让权重按需映射，省掉启动时整份 CPU 侧转换：采样前 91.7 -> 58.0 s，
# 端到端 179.3 -> 145.5 s（数值不变，差异等同 run-to-run 噪声底）。
EXTRA_ARGS=${EXTRA_ARGS---use-ck-attention --mmap-torch-files}

mkdir -p "$OUT"
GM=${GM:-/home/jay/strata/scripts/gpu-mode.sh}
# 双卡占用权：自动挂起 Strata；进程结束后由看门狗自动恢复日常态
if [ -x "$GM" ]; then
  bash "$GM" acquire "tp${WORLD}-${TAG}" --min-free-mib "${MIN_FREE_MIB:-16384}" || {
    echo "[FAIL] 无法获得双卡占用权（Strata 工作中或显存/内存不足）"; exit 1; }
fi
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((BASE_PORT + 300))

# A 轨（默认开）：MLP 按序列切 8 块，让第 i 块的 all_reduce 与第 i+1 块的 GEMM 在两条
# stream 上重叠。8 步实测 8.81 -> 7.62 s/step。COMFY_TP_MLP_CHUNKS=0 可回到逐位旧路径。
export COMFY_TP_MLP_CHUNKS=${COMFY_TP_MLP_CHUNKS:-8}
# B 轨（默认关 bf16）：int8 定点全归约，字节减半（7.62 -> 7.04 s/step），但 4 步
# latent 均值偏差 5%、整片 PSNR 24.5 dB，画质未验证通过，需要时显式开 COMFY_TP_AR_MODE=int8。
export COMFY_TP_AR_MODE=${COMFY_TP_AR_MODE:-bf16}

# 出片跨卡拆分（默认关）：COMFY_TP_VAE_SPLIT=1 时把工作流的 VAEDecode 换成
# TP2VAEDecodeSplit（bench.py --vae-split），两张卡按 VAE 内部时间块各解一半。
# 8 步 832x480x124 实测 post 34.2 -> 27.1 s，像素与单卡路径逐位一致（diff = 0）。
export COMFY_TP_VAE_SPLIT=${COMFY_TP_VAE_SPLIT:-0}

# P2P 拓扑修正补丁：RCCL 会因两卡分属不同 switch 退回 SHM，这里按真实 sysfs
# 设备树生成 NCCL_TOPO_FILE，让 collective 走 P2P/IPC。COMFY_TP_P2P=0 可关掉。
if [ "$WORLD" -gt 1 ] && [ "${COMFY_TP_P2P:-1}" != "0" ]; then
  TOPO=$($PY "$(dirname "$0")/p2p_patch.py" --print 2>/dev/null)
  if [ -n "${TOPO:-}" ] && [ -f "$TOPO" ]; then
    export NCCL_TOPO_FILE="$TOPO"
    echo "p2p patch: NCCL_TOPO_FILE=$TOPO"
  fi
fi

for r in $(seq 0 $((WORLD - 1))); do
  PORT=$((BASE_PORT + r))
  HIP_VISIBLE_DEVICES=$r \
  COMFY_TP_WORLD=$WORLD \
  COMFY_TP_RANK=$r \
  COMFY_TP_TIMING_OUT=$OUT/timing_r$r.json \
  setsid "$PY" "$COMFY/main.py" --listen 127.0.0.1 --port "$PORT" \
      --output-directory "$OUT/out_r$r" $EXTRA_ARGS \
      > "$OUT/r$r.log" 2>&1 < /dev/null &
  pid=$!
  echo "started rank$r port=$PORT pid=$pid"
  [ "$r" = 0 ] && [ -x "$GM" ] && bash "$GM" holder "$pid" >/dev/null 2>&1
done

for r in $(seq 0 $((WORLD - 1))); do
  PORT=$((BASE_PORT + r))
  for i in $(seq 1 120); do
    if curl -sf -m 1 "http://127.0.0.1:$PORT/system_stats" > /dev/null; then
      echo "rank$r ready on $PORT"; break
    fi
    sleep 1
  done
done
