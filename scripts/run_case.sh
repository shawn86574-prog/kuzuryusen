#!/usr/bin/env bash
# run_case.sh <tag> <world> <width> <height> <frames> <steps> [extra bench args...]
# 启动 world 个实例 -> 投同一工作流 -> 收 latent/耗时，结束时不杀实例（下一轮自己清）。
set -u
TAG=$1; WORLD=$2; W=$3; H=$4; F=$5; S=$6
shift 6
D=${OUT_DIR:-$HOME/tp2-logs}/$TAG
PORTS=$(seq 8191 $((8191 + WORLD - 1)))
OUTS=""; TIMS=""
for p in $PORTS; do
  OUTS="$OUTS $D/out_r$((p - 8191))"
  TIMS="$TIMS $D/timing_r$((p - 8191)).json"
done

cd "$(dirname "$0")" || exit 1
# 先清掉占用这些端口的旧实例，否则新实例起不来、bench 会打到旧实例上
for p in 8191 8192 8193 8194; do
  pid=$(ss -ltnpH "sport = :$p" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null && echo "cleared old instance on $p (pid $pid)"
done
sleep 5
EXTRA_ARGS="${EXTRA_ARGS---use-ck-attention --mmap-torch-files}" bash ./run_tp.sh "$WORLD" 8191 "$TAG" > /dev/null 2>&1
${PYTHON:-python3} bench.py --ports $PORTS --outdirs $OUTS --timings $TIMS \
  --unet minimax_h3_fl2va_pruned-w4a8_convrot_pruned.safetensors \
  --clip qwen3vl_32b_minimax_h3-w4a8_convrot.safetensors \
  --vae Minimax_h3_video_vae_fp16.safetensors --audio-vae Minimax_H3_fp32_audio_vae.safetensors \
  --lora minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors \
  --width "$W" --height "$H" --frames "$F" --steps "$S" \
  --dump-dir "$D" --tag "$TAG" --result "$D/result.json" "$@" 2>&1 | tail -3
