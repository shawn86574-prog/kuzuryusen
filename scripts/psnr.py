"""比较两个 mp4 的画面质量：逐帧 PSNR / 全局 PSNR。"""
import sys
import numpy as np


def frames(path, w=None, h=None):
    import av
    with av.open(path) as c:
        arr = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
    return arr.astype(np.float32)


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


if __name__ == "__main__":
    a_path, b_path, w, h = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    A, B = frames(a_path, w, h), frames(b_path, w, h)
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    per = [psnr(A[i], B[i]) for i in range(n)]
    print(f"frames {len(A)} vs {len(B)}")
    print(f"global PSNR = {psnr(A, B):.2f} dB")
    print(f"per-frame min/median/max = {min(per):.2f} / {np.median(per):.2f} / {max(per):.2f} dB")
