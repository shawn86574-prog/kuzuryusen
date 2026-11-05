// attn_int8.hip : single-layer int8 flash attention prototype for gfx1100 (RDNA3)
//
//  * inputs: q,k,v bf16 (H,S,D=128) on device; quantized to int8 with per-row absmax
//    (V is quantized per *column* (per head dim) AND stored transposed, so that the second
//    GEMM's dequant factor ps[m]*vs[n] is separable / constant over the k reduction, and
//    so that its B fragment is 8 *contiguous* bytes in shared memory).
//  * flash-style online softmax, f32 running max/sum, int32 mma accumulators.
//  * P requantized to int8 per row per key-block (scale = row absmax of that block / 127,
//    derived from the monotonicity of exp2 on the block row max), and dequantized with
//    its scale when accumulating P*V.
//  * operands are rocwmma int8 16x16x16 fragments, but they are *loaded by hand* with a
//    single 8-byte LDS per lane (probes 1..3 measured the hardware layouts on this GPU).
//    rocWMMA's own load_matrix_sync for int8 goes through byte-granular ds_load_u8 +
//    ds_swizzle/ds_bpermute, which dominated the runtime.
//
// Measured layouts (gfx1100, wave32, 16x16x16 int8 WMMA, see probe.hip / probe2.hip / probe3.hip):
//   matrix_a   row_major : lane l -> row = l&15, k = (l>>4)*8 + j   (8 contiguous int8)
//   matrix_b   col_major : lane l -> n   = l&15, k = (l>>4)*8 + j   (8 contiguous int8)
//   accumulator          : lane l, elem i -> row = 2*i + (l>>4), col = l&15
//
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocwmma/rocwmma.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>

using namespace rocwmma;

#define DK     128              // head dim
#define MW     16               // query rows per warp
#define NWARP  8                // warps per CTA
#define MCTA   (MW * NWARP)     // 128 query rows per CTA
#define NBLK   64               // keys per kv block
#define KST    136              // K tile row stride in smem bytes (128 + 8 pad: kills bank conflicts)
#define VST    72               // V^T tile row stride in smem bytes ( 64 + 8 pad)
#define PST    72               // P tile row stride in smem bytes   ( 64 + 8 pad)

static constexpr float ATTN_SCALE = 1.4426950408889634f * 0.08838834764831845f; // log2(e)/sqrt(128)

typedef fragment<matrix_a, 16, 16, 16, int8_t, row_major>  FragA;
typedef fragment<matrix_b, 16, 16, 16, int8_t, col_major>  FragBCol;
typedef fragment<accumulator, 16, 16, 16, float>           FragAccF;
typedef fragment<accumulator, 16, 16, 16, int32_t>         FragAccI;

typedef char i8x8 __attribute__((ext_vector_type(8)));

// one 8-byte LDS per lane: the whole operand fragment of a 16x16x16 int8 WMMA
template <typename F>
__device__ __forceinline__ void ld_frag8(F& f, const int8_t* p) {
    *(i8x8*)&f.x = *(const i8x8*)p;
}

// ---------------------------------------------------------------- quantize (per row absmax)
// one warp per row; each lane handles 4 consecutive elements (coalesced 8B loads / 4B stores)
__global__ void k_quant_rows(const __hip_bfloat16* __restrict__ x, int8_t* __restrict__ xq,
                             float* __restrict__ sc, int S, int Spad, long nrows)
{
    long row = (long)(blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= nrows) return;
    int head = (int)(row / Spad);
    int rr   = (int)(row % Spad);
    int lane = threadIdx.x & 31;
    int d0   = lane * 4;
    int8_t* dst = xq + row * DK + d0;
    if (rr >= S) {                                  // zero-padded tail row
        *(int*)dst = 0;
        if (lane == 0) sc[row] = 0.f;
        return;
    }
    const __hip_bfloat16* src = x + (size_t)head * S * DK + (size_t)rr * DK + d0;
    __hip_bfloat162 h01 = *(const __hip_bfloat162*)(src);
    __hip_bfloat162 h23 = *(const __hip_bfloat162*)(src + 2);
    float v0 = __bfloat162float(__low2bfloat16(h01)),  v1 = __bfloat162float(__high2bfloat16(h01));
    float v2 = __bfloat162float(__low2bfloat16(h23)),  v3 = __bfloat162float(__high2bfloat16(h23));
    float m = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor(m, off));
    float s = m * (1.f / 127.f);
    if (!(s > 0.f)) s = 1.f;
    float inv = 1.f / s;
    int i0 = (int)rintf(v0 * inv), i1 = (int)rintf(v1 * inv);
    int i2 = (int)rintf(v2 * inv), i3 = (int)rintf(v3 * inv);
    i0 = min(127, max(-127, i0)); i1 = min(127, max(-127, i1));
    i2 = min(127, max(-127, i2)); i3 = min(127, max(-127, i3));
    *(int*)dst = (i0 & 0xff) | ((i1 & 0xff) << 8) | ((i2 & 0xff) << 16) | ((i3 & 0xff) << 24);
    if (lane == 0) sc[row] = s;
}

// ------------------------------------------------------- V per-column absmax (over keys)
// coalesced: block of 128 threads, thread t owns dim t, loops over the key chunk
__global__ void k_vcol_absmax(const __hip_bfloat16* __restrict__ v, float* __restrict__ cm,
                              int H, int S, int nchunk)
{
    int h = blockIdx.x, c = blockIdx.y, d = threadIdx.x;
    int k0 = (int)((long)c * S / nchunk), k1 = (int)((long)(c + 1) * S / nchunk);
    const __hip_bfloat16* p = v + (size_t)h * S * DK + d;
    float m = 0.f;
    for (int k = k0; k < k1; k++) m = fmaxf(m, fabsf(__bfloat162float(p[(size_t)k * DK])));
    atomicMax((int*)(cm + h * DK + d), __float_as_int(m));
}

// ------------------------------------------- V quantize, transposed: vqT[head][dim][key]
// block = 128 threads, thread t owns dim t; one block does 16 keys x 128 dims.
__global__ void k_quant_vt(const __hip_bfloat16* __restrict__ v, int8_t* __restrict__ vqT,
                           float* __restrict__ vsc, const float* __restrict__ cm,
                           int H, int S, int Spad)
{
    int head = blockIdx.y;
    int k0   = blockIdx.x * 16;
    int d    = threadIdx.x;
    if (k0 >= Spad) return;
    const float* cms = cm + head * DK;
    float s = cms[d] * (1.f / 127.f);
    if (!(s > 0.f)) s = 1.f;
    vsc[head * DK + d] = s;
    int8_t* dst = vqT + (size_t)head * Spad * DK + (size_t)d * Spad + k0;
    int wv[4] = {0, 0, 0, 0};
    const __hip_bfloat16* src = v + (size_t)head * S * DK + d;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        int iv = 0;
        if (k0 + j < S) {
            float x = __bfloat162float(src[(size_t)(k0 + j) * DK]) / s;
            iv = (int)rintf(x);
            iv = min(127, max(-127, iv));
        }
        wv[j >> 2] |= (iv & 0xff) << (8 * (j & 3));
    }
    *(int*)dst       = wv[0];
    *(int*)(dst + 4) = wv[1];
    *(int*)(dst + 8) = wv[2];
    *(int*)(dst + 12) = wv[3];
}

// ---------------------------------------------------------------- int8 flash attention
__global__ __launch_bounds__(NWARP * 32, 1) void attn_fwd(
    const int8_t* __restrict__ Qg, const float* __restrict__ Qs,
    const int8_t* __restrict__ Kg, const float* __restrict__ Ks_,
    const int8_t* __restrict__ Vg, const float* __restrict__ Vs_,
    __hip_bfloat16* __restrict__ Og,
    int H, int S, int Spad)
{
    // K/V/P tiles and the epilogue staging buffer do not overlap in time -> alias them
    __shared__ union {
        struct {
            int8_t K[NBLK * KST];       //  8.5 KB
            int8_t V[DK * VST];         //  9.0 KB  (transposed: [dim][key])
            int8_t P[MCTA * PST];       //  9.0 KB
            float  Ks[NBLK];
        } a;
        float epi[NWARP][MW * 32];      // 16 KB
    } sm;
    auto& sK   = sm.a.K;
    auto& sV   = sm.a.V;
    auto& sP   = sm.a.P;
    auto& sKs  = sm.a.Ks;
    auto& sEpi = sm.epi;
    // per-row state lives in shared memory (16 identical copies per lane group would be
    // pure register waste); these are broadcast reads/writes
    __shared__ float sQsc[MCTA];              // query-row dequant scale (ATTN_SCALE folded)
    __shared__ float sVsc[DK];                // output-column V dequant scale
    __shared__ float sMrow[NWARP][MW];        // running row max
    __shared__ float sLrow[NWARP][MW];        // running row sum

    const int w     = threadIdx.x >> 5;
    const int lane  = threadIdx.x & 31;
    const int lb    = lane & 15;   // row of an A fragment / n of a B fragment
    const int hb    = lane >> 4;   // which 8-byte half of the k range
    const int head  = blockIdx.y;
    const int qb    = blockIdx.x;
    const int rowW0 = qb * MCTA + w * MW;

    // stage the scale arrays into shared memory
    {
        const int tid = threadIdx.x;
        if (tid < MCTA) {
            sQsc[tid] = Qs[(size_t)head * Spad + qb * MCTA + tid] * ATTN_SCALE;
        }
        if (tid < DK) sVsc[tid] = Vs_[(size_t)head * DK + tid];
        if (tid < NWARP * MW) {
            sMrow[tid / MW][tid % MW] = -INFINITY;
            sLrow[tid / MW][tid % MW] = 0.f;
        }
    }

    FragA qa[8];
    {
        const int8_t* qp = Qg + (size_t)head * Spad * DK + (size_t)rowW0 * DK;
        #pragma unroll
        for (int kb = 0; kb < 8; kb++) ld_frag8(qa[kb], qp + lb * DK + kb * 16 + hb * 8);
    }

    FragAccF acc[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) fill_fragment(acc[t], 0.f);

    const int kvEnd = (S + NBLK - 1) / NBLK;

    for (int kb0 = 0; kb0 < kvEnd; kb0++) {
        const int kstart = kb0 * NBLK;
        const int nvalid = (S - kstart < NBLK) ? (S - kstart) : NBLK;
        __syncthreads();                       // previous GEMM2 done reading sK/sV

        const int8_t* kp = Kg + (size_t)head * Spad * DK + (size_t)kstart * DK;
        const int8_t* vp = Vg + (size_t)head * Spad * DK + (size_t)kstart;   // [dim][Spad]
        #pragma unroll
        for (int t = threadIdx.x; t < NBLK * DK / 16; t += NWARP * 32) {
            const int rr = t >> 3, cc = (t & 7) * 16;                       // K : 64 keys x 8 chunks
            *(uint4*)(sK + rr * KST + cc) = *(const uint4*)(kp + rr * DK + cc);
            const int dd = t >> 2, cd = (t & 3) * 16;                       // V^T : 128 dims x 4 chunks
            *(uint4*)(sV + dd * VST + cd) = *(const uint4*)(vp + (size_t)dd * Spad + cd);
        }
        if (threadIdx.x < nvalid) sKs[threadIdx.x] = Ks_[(size_t)head * Spad + kstart + threadIdx.x];
        __syncthreads();                       // sK/sV/sKs ready

        // ---- GEMM1 : S(16 x 64) = Q(16 x 128) * K^T(128 x 64)
        FragAccI sf[4];
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) fill_fragment(sf[nb], 0);
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            const int8_t* kb_ = sK + (nb * 16 + lb) * KST + hb * 8;
            #pragma unroll
            for (int kg = 0; kg < 2; kg++) {          // 4 k-blocks at a time keeps regs low
                FragBCol kf[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) ld_frag8(kf[j], kb_ + (kg * 4 + j) * 16);
                #pragma unroll
                for (int j = 0; j < 4; j++) mma_sync(sf[nb], qa[kg * 4 + j], kf[j], sf[nb]);
            }
        }

        // ---- dequant + online softmax
        // pass 1: row max over this key block
        float bmax[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) bmax[i] = -INFINITY;
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            if (nb * 16 + lb < nvalid) {
                const float ksv = sKs[nb * 16 + lb];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float v = (float)sf[nb].x[i] * (sQsc[w * MW + 2 * i + hb] * ksv);
                    bmax[i] = fmaxf(bmax[i], v);
                }
            }
        }
        #pragma unroll
        for (int off = 1; off < 16; off <<= 1) {
            #pragma unroll
            for (int i = 0; i < 8; i++) bmax[i] = fmaxf(bmax[i], __shfl_xor(bmax[i], off));
        }
        float alpha[8], pinv[8], psca[8];
        {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int r = 2 * i + hb;
                float old = sMrow[w][r];
                float mn = fmaxf(old, bmax[i]);
                sMrow[w][r] = mn;                      // updated running max
                alpha[i] = exp2f(old - mn);
                // row max of P = exp2(row max of S - m): exp2 is monotonic
                float pm = exp2f(bmax[i] - mn);
                pinv[i]  = 127.f / pm;
                psca[i]  = pm * (1.f / 127.f);
            }
        }
        // pass 2: p = exp2(s - m), requantize per row, accumulate row sum
        float bsum[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) bsum[i] = 0.f;
        {
            // 8 bytes per lane would be strided across rows, so P is written byte-wise
            #pragma unroll
            for (int nb = 0; nb < 4; nb++) {
                const bool ok = (nb * 16 + lb < nvalid);
                const float ksv = sKs[nb * 16 + lb];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int r = 2 * i + hb;
                    float p = ok ? exp2f((float)sf[nb].x[i] * (sQsc[w * MW + r] * ksv)
                                        - sMrow[w][r]) : 0.f;
                    bsum[i] += p;
                    int iv = (int)(p * pinv[i] + 0.5f);
                    sP[(w * MW + r) * PST + nb * 16 + lb] = (int8_t)min(127, iv);
                }
            }
        }
        #pragma unroll
        for (int off = 1; off < 16; off <<= 1) {
            #pragma unroll
            for (int i = 0; i < 8; i++) bsum[i] += __shfl_xor(bsum[i], off);
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) sLrow[w][2 * i + hb] = sLrow[w][2 * i + hb] * alpha[i] + bsum[i];
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[t].x[i] *= alpha[i];
        }
        __syncthreads();                       // sP ready

        // ---- GEMM2 : acc += P(16 x 64) * V(64 x 128); two column groups of 4 tiles
        const int8_t* pa_ = sP + (w * MW + lb) * PST + hb * 8;
        #pragma unroll
        for (int tg = 0; tg < 2; tg++) {
            FragAccI cf[4];
            #pragma unroll
            for (int t = 0; t < 4; t++) fill_fragment(cf[t], 0);
            #pragma unroll
            for (int kb2 = 0; kb2 < 4; kb2++) {
                FragA pa;
                ld_frag8(pa, pa_ + kb2 * 16);
                FragBCol vf[4];
                #pragma unroll
                for (int t = 0; t < 4; t++)
                    ld_frag8(vf[t], sV + ((tg * 4 + t) * 16 + lb) * VST + kb2 * 16 + hb * 8);
                #pragma unroll
                for (int t = 0; t < 4; t++) mma_sync(cf[t], pa, vf[t], cf[t]);
            }
            #pragma unroll
            for (int t = 0; t < 4; t++) {
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    acc[tg * 4 + t].x[i] += (float)cf[t].x[i] * (psca[i] * sVsc[(tg * 4 + t) * 16 + lb]);
            }
        }
    }

    // ---------------------------------------------------------------- epilogue
    __syncthreads();
    #pragma unroll
    for (int t = 0; t < 8; t++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[t].x[i] /= sLrow[w][2 * i + hb];
    }
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        store_matrix_sync(&sEpi[w][0],  acc[2 * c],     32, mem_row_major);
        store_matrix_sync(&sEpi[w][16], acc[2 * c + 1], 32, mem_row_major);
        __syncthreads();
        const int r    = lane >> 1;
        const int half = lane & 1;
        const int grow = rowW0 + r;
        if (grow < S) {
            float* srcf = &sEpi[w][r * 32 + half * 16];
            __hip_bfloat162 o[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float2 f; f.x = srcf[2 * j]; f.y = srcf[2 * j + 1];
                o[j] = __float22bfloat162_rn(f);
            }
            __hip_bfloat16* dst = Og + ((size_t)head * S + grow) * DK + c * 32 + half * 16;
            *(uint4*)dst       = *(uint4*)&o[0];
            *(uint4*)(dst + 8) = *(uint4*)&o[4];
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------- host launchers
extern "C" {

void hipattn_quantize(const void* q, const void* k, const void* v, int H, int S, int Spad,
                      int8_t* qq, float* qsc, int8_t* kq, float* ksc,
                      int8_t* vq, float* vsc, float* vcolmax, void* stream_)
{
    hipStream_t st = (hipStream_t)stream_;
    const long nrows = (long)H * Spad;
    const int nchunk = 64;
    const __hip_bfloat16* vv = (const __hip_bfloat16*)v;
    hipMemsetAsync(vcolmax, 0, (size_t)H * DK * sizeof(float), st);
    dim3 g1(H, nchunk);
    k_vcol_absmax<<<g1, 128, 0, st>>>(vv, vcolmax, H, S, nchunk);
    const int nb = (int)((nrows + 7) / 8);          // 8 rows per block (8 warps)
    k_quant_rows<<<nb, 256, 0, st>>>((const __hip_bfloat16*)q, qq, qsc, S, Spad, nrows);
    k_quant_rows<<<nb, 256, 0, st>>>((const __hip_bfloat16*)k, kq, ksc, S, Spad, nrows);
    dim3 g2((Spad + 15) / 16, H);
    k_quant_vt<<<g2, 128, 0, st>>>(vv, vq, vsc, vcolmax, H, S, Spad);
}

void hipattn_forward(const int8_t* qq, const float* qsc, const int8_t* kq, const float* ksc,
                     const int8_t* vq, const float* vsc, void* out, int H, int S, int Spad,
                     void* stream_)
{
    hipStream_t st = (hipStream_t)stream_;
    dim3 grid((S + MCTA - 1) / MCTA, H);
    attn_fwd<<<grid, NWARP * 32, 0, st>>>(qq, qsc, kq, ksc, vq, vsc,
                                          (__hip_bfloat16*)out, H, S, Spad);
}

int hipattn_last_error() { return (int)hipGetLastError(); }

}
