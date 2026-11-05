#!/usr/bin/env python3
"""RCCL 拓扑修正补丁：按本机真实 PCIe 设备树生成 NCCL_TOPO_FILE。

背景：本机两块 gfx1100 各挂在一个 AMD "Navi 10 XL" PCIe switch 的 x16 下游口上，
两个 switch 的上游口又分别接在 CPU 根端口（8.0 GT/s ×8）下，没有任何共享 switch。
RCCL 的拓扑判定因此给出 isAllDirectP2p 0，collective 退回 SHM（主机内存中转）。
但 HIP 层 peer 访问是通的（can_device_access_peer=1 / NativeAtomicSupported=1），
强制声明两卡为同一 switch 下的 peer 后可走 P2P/IPC，带宽 4.1 -> 5.03 GB/s。

这个补丁做两件事：
  1. 从 sysfs 读真实 busid / vendor / device / 链路速率，生成拓扑声明文件；
  2. 打印检测到的链路链条，便于报告里说明“补丁改的是判定，不是硬件”。

用法：
  p2p_patch.py                 # 生成 p2p_topology.xml 并输出路径
  p2p_patch.py --print         # 只输出路径（给 shell 用）
  p2p_patch.py --info          # 只打印链路检查结果，不写文件
"""
import argparse
import glob
import os
import sys
import tempfile

SYS = "/sys/bus/pci/devices"
DEFAULT_OUT = os.path.join(tempfile.gettempdir(), "p2p_topology.xml")
GPU_CLASS = "0x030000"
AMD = "0x1002"


def rd(path, name):
    try:
        with open(os.path.join(path, name)) as f:
            return f.read().strip()
    except OSError:
        return ""


def amd_gpus():
    out = []
    for d in sorted(glob.glob(os.path.join(SYS, "*"))):
        if rd(d, "class") == GPU_CLASS and rd(d, "vendor") == AMD:
            out.append(d)
    return out


def parent(dev_path):
    """sysfs 里设备的父目录就是上游 bridge（PCI 设备树）。"""
    p = os.path.dirname(os.path.realpath(dev_path))
    if rd(p, "class").startswith("0x0604"):
        return p
    return None


def link(dev_path):
    """返回 (speed_str, width_int)；speed_str 形如 '16.0 GT/s PCIe'。"""
    spd = rd(dev_path, "current_link_speed") or rd(dev_path, "max_link_speed")
    wid = rd(dev_path, "max_link_width") or rd(dev_path, "current_link_width")
    try:
        w = int(wid)
    except ValueError:
        w = 16
    return (spd or "16.0 GT/s PCIe"), w


def cur_link(dev_path):
    """返回当前实际协商速率/宽度（报告里讲链路瓶颈要用这个，不是 max）。"""
    spd = rd(dev_path, "current_link_speed") or rd(dev_path, "max_link_speed")
    try:
        w = int(rd(dev_path, "current_link_width") or 0)
    except ValueError:
        w = 0
    return spd or "?", w


def gfx_info():
    """从 KFD 读 gfx_target_version / simd_count，拼出 gcn/arch 字段。"""
    for n in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*")):
        props = {}
        try:
            with open(os.path.join(n, "properties")) as f:
                for line in f:
                    k, _, v = line.partition(" ")
                    props[k] = v.strip()
        except OSError:
            continue
        ver = props.get("gfx_target_version", "0")
        if props.get("simd_count", "0") != "0" and ver != "0":
            major, minor = int(ver) // 10000, int(ver) % 10000 // 100
            # gfx1100 -> arch 38911 是 RCCL 对 gfx11 的编码（与官方拓扑示例一致）
            arch = 38911 if major == 11 else 0
            # RCCL 的 sm 字段按 CU 数算：simd_count / simd_per_cu（7900 XTX = 96）
            try:
                cu = int(props.get("simd_count", "0")) // max(1, int(props.get("simd_per_cu", "2")))
            except ValueError:
                cu = 48
            return f"gfx{major}{minor:02d}", str(cu), arch
    return "gfx1100", "96", 38911


def detect():
    gpus = amd_gpus()
    if len(gpus) < 2:
        return None, f"只找到 {len(gpus)} 块 AMD GPU，TP 至少需要 2 块"
    gcn, simd, arch = gfx_info()
    chain = []
    for i, g in enumerate(gpus):
        up = parent(g)
        root = os.path.dirname(os.path.realpath(up)) if up else None
        chain.append({
            "index": i,
            "busid": os.path.basename(g),
            "device": rd(g, "device"),
            "link": link(g),
            "switch": os.path.basename(up) if up else "",
            "switch_device": rd(up, "device") if up else "",
            "switch_link": link(up) if up else ("", 0),
            "root": os.path.basename(root) if root else "",
            "root_link": cur_link(root) if root else ("?", 0),
            "switch_cur": cur_link(up) if up else ("?", 0),
        })
    return {
        "gcn": gcn, "simd": simd, "arch": arch,
        "gpus": chain,
        # 用第一块卡的上游 switch 作为“共享 switch”的声明载体
        "switch_busid": chain[0]["switch"] or chain[0]["busid"],
        "switch_device": chain[0]["switch_device"] or chain[0]["device"],
        "switch_link": chain[0]["switch_link"],
    }, None


def render(info):
    spd, wid = info["switch_link"]
    if not spd:
        spd, wid = "16.0 GT/s PCIe", 16
    lines = [
        '<system version="2">',
        '  <cpu host_hash="0xdeadbeef" numaid="0" affinity="3fffff" arch="x86_64"'
        ' vendor="GenuineIntel" familyid="6" modelid="79">',
        f'    <pci busid="{info["switch_busid"]}" class="0x060400" vendor="{AMD}"'
        f' device="{info["switch_device"]}" link_speed="{spd}" link_width="{wid}">',
    ]
    for g in info["gpus"]:
        gspd, gwid = g["link"]
        lines += [
            f'      <pci busid="{g["busid"]}" class="{GPU_CLASS}" vendor="{AMD}"'
            f' device="{g["device"]}" link_speed="{gspd}" link_width="{gwid}">',
            f'        <gpu dev="{g["index"]}" sm="{info["simd"]}" gcn="{info["gcn"]}"'
            f' arch="{info["arch"]}" rank="{g["index"]}" gdr="0"/>',
            '      </pci>',
        ]
    lines += ['    </pci>', '  </cpu>', '</system>', '']
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--print", dest="only_print", action="store_true")
    ap.add_argument("--info", action="store_true")
    a = ap.parse_args()

    info, err = detect()
    if err:
        print(err, file=sys.stderr)
        return 1

    if a.info:
        for g in info["gpus"]:
            print(f'GPU{g["index"]} {g["busid"]} dev={g["device"]} '
                  f'{g["link"][0]} x{g["link"][1]} <- switch {g["switch"]} '
                  f'{g["switch_cur"][0]} x{g["switch_cur"][1]} <- root {g["root"]} '
                  f'{g["root_link"][0]} x{g["root_link"][1]}')
        return 0

    xml = render(info)
    with open(a.out, "w") as f:
        f.write(xml)
    if a.only_print:
        print(a.out)
    else:
        print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
