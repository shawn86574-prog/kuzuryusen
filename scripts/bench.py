#!/usr/bin/env python3
"""TP2 基准驱动：同时向 N 个 ComfyUI 实例投同一张工作流，采集耗时与显存。"""
import argparse
import hashlib
import json
import os
import threading
import time

import requests

GPU = {"0": "0000:05:00.0", "1": "0000:08:00.0"}
VRAM = "/sys/bus/pci/devices/{}/mem_info_vram_used"
VRAM_T = "/sys/bus/pci/devices/{}/mem_info_vram_total"


def read_vram():
    out = {}
    for k, dev in GPU.items():
        try:
            out[k] = int(open(VRAM.format(dev)).read().strip()) / 2**30
        except Exception:
            out[k] = None
    return out


class VramMon(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            self.samples.append({"t": time.time(), **read_vram()})
            time.sleep(0.5)

    def peak(self, gpu):
        vals = [s[gpu] for s in self.samples if s.get(gpu)]
        return max(vals) if vals else None

    def mean_after(self, gpu, t0, skip=20.0):
        vals = [s[gpu] for s in self.samples if s.get(gpu) and s["t"] > t0 + skip]
        return sum(vals) / len(vals) if vals else None


def build_wf(a, port=None):
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": a.unet, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": a.clip, "type": "minimax", "device": os.environ.get("BENCH_CLIP_DEVICE", "default")}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": a.vae}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": a.audio_vae}},
        "6": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": a.prompt,
                         "width": a.width, "height": a.height, "length": a.frames}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": a.sampler}},
        "8": {"class_type": "BasicScheduler",
              "inputs": {"model": ["2", 0], "scheduler": "simple", "steps": a.steps, "denoise": 1.0}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed}},
        "10": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["6", 0]}},
        "11": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["9", 0], "guider": ["10", 0], "sampler": ["7", 0],
                          "sigmas": ["8", 0], "latent_image": ["6", 1]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "13": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["5", 0]}},
        "14": {"class_type": "CreateVideo", "inputs": {"images": ["12", 0], "audio": ["13", 0], "fps": 24}},
        "15": {"class_type": "SaveVideo",
               "inputs": {"video": ["14", 0], "filename_prefix": "tp2/out", "format": "auto"}},
    }
    ffn_chunks = int(os.environ.get("BENCH_FFN_CHUNKS", "0") or 0)
    head_chunks = int(os.environ.get("BENCH_ATTN_HEAD_CHUNKS", "0") or 0)
    model_src = ["1", 0]
    if a.lora and a.lora.lower() != "none":
        wf["2"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": model_src, "lora_name": a.lora,
                              "strength_model": a.lora_strength}}
        model_src = ["2", 0]
    if ffn_chunks > 1:
        wf["17"] = {"class_type": "MiniMaxChunkFeedForward",
                    "inputs": {"model": model_src, "chunks": ffn_chunks, "seq_threshold": 4096}}
        model_src = ["17", 0]
    if head_chunks > 1:
        wf["18"] = {"class_type": "MiniMaxLowVRAMAttention",
                    "inputs": {"model": model_src, "head_chunks": head_chunks}}
        model_src = ["18", 0]
    if a.lora2:
        wf["19"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": model_src, "lora_name": a.lora2,
                               "strength_model": a.lora2_strength}}
        model_src = ["19", 0]
    wf["8"]["inputs"]["model"] = model_src
    wf["10"]["inputs"]["model"] = model_src

    if a.dump_dir:
        os.makedirs(a.dump_dir, exist_ok=True)
        wf["16"] = {"class_type": "TP2DumpLatent",
                    "inputs": {"samples": ["11", 0],
                               "path": os.path.join(a.dump_dir, f"latent_{port}.pt")}}
    return wf


def submit(port, wf):
    r = requests.post(f"http://127.0.0.1:{port}/prompt",
                      json={"prompt": wf, "client_id": f"tp2-{port}"}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"port {port} submit failed {r.status_code}: {r.text[:800]}")
    return r.json()["prompt_id"]


def wait_done(port, pid, timeout=7200):
    poll_timeout = float(os.environ.get("BENCH_POLL_TIMEOUT", "10"))
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = requests.get(f"http://127.0.0.1:{port}/history/{pid}", timeout=poll_timeout).json()
        except requests.exceptions.RequestException:
            # TP2 下 rank1 可能被 collective 阻塞，单次轮询超时不代表任务失败
            time.sleep(2.0)
            continue
        if pid in h:
            entry = h[pid]
            st = entry.get("status", {})
            if st.get("status_str") == "error" or st.get("completed") is False and st.get("status_str") == "error":
                return False, entry
            if st.get("completed"):
                return True, entry
        time.sleep(1.0)
    return False, None


def steps_from_timing(path, steps):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    calls = d.get("step_calls") or []
    if len(calls) < 2:
        return None
    deltas = [round(calls[i + 1] - calls[i], 3) for i in range(len(calls) - 1)]
    # 首步含模型加载/预热，末步可能含收尾，按帖子口径取第 2..steps-1
    core = deltas[1:-1] if len(deltas) >= 3 else deltas
    return {"deltas": deltas, "calls": len(calls),
            "per_step": round(sum(core) / len(core), 3),
            "all_reduce": d.get("all_reduce_calls"),
            "all_reduce_gb": round((d.get("all_reduce_bytes") or 0) / 2**30, 2),
            "first_call": calls[0], "last_call": calls[-1]}


def md5_of_outputs(outdir):
    res = {}
    for root, _, files in os.walk(outdir):
        for f in files:
            if f.endswith((".mp4", ".webm", ".mkv")):
                p = os.path.join(root, f)
                res[f] = hashlib.md5(open(p, "rb").read()).hexdigest()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", nargs="+", type=int, required=True)
    ap.add_argument("--outdirs", nargs="+", required=True)
    ap.add_argument("--timings", nargs="+", required=True)
    ap.add_argument("--unet", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--audio-vae", required=True)
    ap.add_argument("--lora", default="Minimax_H3_fp32_audio_vae.safetensors.tmp")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--lora2", default="")
    ap.add_argument("--lora2-strength", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=124)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=145)
    ap.add_argument("--sampler", default="res_multistep")
    ap.add_argument("--prompt", default="A calm lakeside at sunrise, slow dolly-in, mist over the water, gentle ripples, cinematic soft light.")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--dump-dir", default="")
    ap.add_argument("--vae-split", action="store_true",
                    help="节点 12 换成 TP2VAEDecodeSplit（需实例 COMFY_TP_VAE_SPLIT=1）")
    ap.add_argument("--result", default="/home/jay/tp2/logs/result.json")
    a = ap.parse_args()

    wfs = [build_wf(a, p) for p in a.ports]
    if a.vae_split:
        for wf in wfs:
            wf["12"]["class_type"] = "TP2VAEDecodeSplit"
    if a.no_lora:
        for wf in wfs:
            wf.pop("2", None)
            wf["8"]["inputs"]["model"] = ["1", 0]
            wf["10"]["inputs"]["model"] = ["1", 0]
    mon = VramMon()
    mon.start()

    t0 = time.time()
    pids = [submit(p, wf) for p, wf in zip(a.ports, wfs)]
    print("submitted", pids, flush=True)

    ok_all, results = True, []
    threads = []

    def watch(i):
        ok, entry = wait_done(a.ports[i], pids[i])
        results.append((i, ok, entry, time.time() - t0))

    for i in range(len(a.ports)):
        t = threading.Thread(target=watch, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    total = time.time() - t0
    mon.stop_flag = True

    rep = {"tag": a.tag, "res": f"{a.width}x{a.height}", "frames": a.frames,
           "steps": a.steps, "world": len(a.ports), "unet": a.unet,
           "end_to_end_s": round(max(r[3] for r in results), 1), "ranks": []}

    for i, ok, entry, w in sorted(results):
        st = steps_from_timing(a.timings[i], a.steps)
        get = lambda g: (mon.peak(g), mon.mean_after(g, t0))
        # 采样前（模型加载 + 文本编码）与采样后（VAE 解码 + 出片）的墙钟时间
        pre = round(st["first_call"] - t0, 1) if st else None
        post = round(w - (st["last_call"] - t0), 1) if st else None
        rep["ranks"].append({
            "rank": i, "ok": ok, "wall_s": round(w, 1), "steps": st,
            "pre_sampling_s": pre, "post_sampling_s": post,
            "vram_peak_gb": round(mon.peak(str(i)), 2) if mon.peak(str(i)) else None,
            "vram_avg_gb": round(get(str(i))[1], 2) if get(str(i))[1] else None,
            "md5": md5_of_outputs(a.outdirs[i]),
            "err": None if ok else json.dumps(entry)[:1500] if entry else "timeout",
        })

    os.makedirs(os.path.dirname(a.result), exist_ok=True)
    json.dump(rep, open(a.result, "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
