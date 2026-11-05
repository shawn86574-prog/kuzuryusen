#!/usr/bin/env bash
# 全档 P2P 复测：每个分辨率跑两组 —— 仅打通 P2P、P2P+MLP4 块异步重叠。
set -u
cd "$(dirname "$0")" || exit 1
for spec in "640 352" "832 480" "1024 576" "1280 704" "1344 768" "1536 864"; do
  set -- $spec
  W=$1; H=$2
  echo "=== ${W}x${H} P2P $(date +%H:%M:%S)"
  COMFY_TP_MLP_CHUNKS=0 EXTRA_ARGS="--use-ck-attention" \
    bash run_case.sh "p2p_${W}" 2 "$W" "$H" 124 4
  echo "=== ${W}x${H} P2P+OV4 $(date +%H:%M:%S)"
  COMFY_TP_MLP_CHUNKS=4 EXTRA_ARGS="--use-ck-attention" \
    bash run_case.sh "ov_${W}" 2 "$W" "$H" 124 4
done
echo "ALL DONE $(date +%H:%M:%S)"
