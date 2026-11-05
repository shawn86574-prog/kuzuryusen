// probe2.hip : dump the int8 A/B fragment register layouts used by the gfx11 WMMA builtin,
//              and verify that a hand-built register set reproduces the rocwmma result.
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace rocwmma;
typedef int int32x4_t __attribute__((ext_vector_type(4)));
typedef int int32x8_t __attribute__((ext_vector_type(8)));
#define W 16

__global__ void k_dump_a(const int8_t* A, int32_t* out) {
    fragment<matrix_a, W, W, W, int8_t, row_major> a;
    load_matrix_sync(a, A, W);
    int lane = threadIdx.x;
    const unsigned char* p = (const unsigned char*)&a;
    for (int i = 0; i < 16; i++) out[lane * 16 + i] = p[i];
    if (lane == 0) out[512] = (int32_t)(sizeof(a));
}

__global__ void k_dump_b(const int8_t* Bt, int32_t* out) {
    fragment<matrix_b, W, W, W, int8_t, col_major> b;
    load_matrix_sync(b, Bt, W);
    int lane = threadIdx.x;
    const unsigned char* p = (const unsigned char*)&b;
    for (int i = 0; i < 16; i++) out[lane * 16 + i] = p[i];
}

// hand-built registers + builtin, vs reference
__global__ void k_builtin(const int8_t* A, const int8_t* Bt, int32_t* C,
                          const int32_t* aRegs, const int32_t* bRegs) {
    int lane = threadIdx.x;
    int32x4_t av, bv;
    int32_t* avp = (int32_t*)&av; int32_t* bvp = (int32_t*)&bv;
    for (int i = 0; i < 4; i++) { avp[i] = aRegs[lane * 4 + i]; bvp[i] = bRegs[lane * 4 + i]; }
    int32x8_t cv;
    cv = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, av, true, bv, cv, false);
    int32_t* cp = (int32_t*)&cv;
    for (int i = 0; i < 8; i++) C[lane * 8 + i] = cp[i];
}

// extract raw dwords from the rocwmma fragments (to build host-side tables)
__global__ void k_regs_a(const int8_t* A, int32_t* out) {
    fragment<matrix_a, W, W, W, int8_t, row_major> a;
    load_matrix_sync(a, A, W);
    const int32_t* p = (const int32_t*)&a;
    for (int i = 0; i < 4; i++) out[threadIdx.x * 4 + i] = p[i];
}
__global__ void k_regs_b(const int8_t* Bt, int32_t* out) {
    fragment<matrix_b, W, W, W, int8_t, col_major> b;
    load_matrix_sync(b, Bt, W);
    const int32_t* p = (const int32_t*)&b;
    for (int i = 0; i < 4; i++) out[threadIdx.x * 4 + i] = p[i];
}
__global__ void k_mma_ref(const int8_t* A, const int8_t* Bt, int32_t* C) {
    fragment<matrix_a, W, W, W, int8_t, row_major> a;
    fragment<matrix_b, W, W, W, int8_t, col_major> b;
    fragment<accumulator, W, W, W, int32_t> c;
    fill_fragment(c, 0);
    load_matrix_sync(a, A, W);
    load_matrix_sync(b, Bt, W);
    mma_sync(c, a, b, c);
    store_matrix_sync(C, c, W, mem_row_major);
}

static int32_t as_s8(unsigned char b) { return (int32_t)((b > 127) ? (int)b - 256 : (int)b); }

int main() {
    std::vector<int8_t> hA(W * W), hBt(W * W);
    std::vector<int32_t> refC(W * W, 0);
    for (int r = 0; r < W; r++)
        for (int k = 0; k < W; k++) hA[r * W + k] = (int8_t)(r * W + k);      // index pattern
    for (int n = 0; n < W; n++)
        for (int k = 0; k < W; k++) hBt[n * W + k] = (int8_t)(n * W + k);     // index pattern
    for (int m = 0; m < W; m++)
        for (int n = 0; n < W; n++) {
            int32_t s = 0;
            for (int k = 0; k < W; k++) s += (int32_t)as_s8((unsigned char)hA[m * W + k]) * (int32_t)as_s8((unsigned char)hBt[n * W + k]);
            refC[m * W + n] = s;
        }
    int8_t *dA, *dBt; int32_t *dOut, *dC, *dRegs;
    hipMalloc(&dA, W * W); hipMalloc(&dBt, W * W);
    hipMalloc(&dOut, (512 + 16) * 4); hipMalloc(&dC, W * W * 4); hipMalloc(&dRegs, 32 * 16 * 4);
    hipMemcpy(dA, hA.data(), W * W, hipMemcpyHostToDevice);
    hipMemcpy(dBt, hBt.data(), W * W, hipMemcpyHostToDevice);

    std::vector<int32_t> hOut(528), hRegs(512), hC(W * W);
    hipMemset(dOut, 0, (512 + 16) * 4);
    k_dump_a<<<1, 32>>>(dA, dOut);
    hipDeviceSynchronize();
    hipMemcpy(hOut.data(), dOut, (512 + 16) * 4, hipMemcpyDeviceToHost);
    printf("sizeof(A frag) = %d bytes\n", hOut[512]);
    printf("A row_major layout (lane: 8/16 elements as (r,k)):\n");
    for (int l = 0; l < 32; l++) {
        printf("  %2d:", l);
        for (int i = 0; i < 16; i++) {
            int v = hOut[l * 16 + i];
            printf(" %2d=(%d,%d)", v, v / 16, v % 16);
        }
        printf("\n");
    }

    hipMemset(dOut, 0, (512 + 16) * 4);
    k_dump_b<<<1, 32>>>(dBt, dOut);
    hipDeviceSynchronize();
    hipMemcpy(hOut.data(), dOut, (512 + 16) * 4, hipMemcpyDeviceToHost);
    printf("B col_major layout (lane: elements as (k,n)):\n");
    for (int l = 0; l < 32; l++) {
        printf("  %2d:", l);
        for (int i = 0; i < 16; i++) {
            int v = hOut[l * 16 + i];
            printf(" %2d=(%d,%d)", v, v / 16, v % 16);
        }
        printf("\n");
    }

    // verify direct builtin with the rocwmma-derived registers
    k_regs_a<<<1, 32>>>(dA, dRegs);
    std::vector<int32_t> hARegs(128);
    hipMemcpy(hARegs.data(), dRegs, 128 * 4, hipMemcpyDeviceToHost);
    k_regs_b<<<1, 32>>>(dBt, dRegs + 128);
    std::vector<int32_t> hBRegs(128);
    hipMemcpy(hBRegs.data(), dRegs + 128, 128 * 4, hipMemcpyDeviceToHost);
    int32_t *dAR, *dBR;
    hipMalloc(&dAR, 128 * 4); hipMalloc(&dBR, 128 * 4);
    hipMemcpy(dAR, hARegs.data(), 128 * 4, hipMemcpyHostToDevice);
    hipMemcpy(dBR, hBRegs.data(), 128 * 4, hipMemcpyHostToDevice);
    k_builtin<<<1, 32>>>(dA, dBt, dC, dAR, dBR);
    hipDeviceSynchronize();
    auto e = hipGetLastError();
    if (e != hipSuccess) { printf("builtin FAILED: %s\n", hipGetErrorString(e)); return 1; }
    hipMemcpy(hC.data(), dC, W * W * 4, hipMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < W * W; i++)
        if (hC[i] != refC[i]) { if (bad < 4) printf("  mismatch %d: %d vs %d\n", i, hC[i], refC[i]); bad++; }

    // the builtin writes C in lane-interleaved raw order; compare through the known C layout
    // row = 2*i + lane/16, col = lane%16
    int bad2 = 0;
    for (int l = 0; l < 32; l++)
        for (int i = 0; i < 8; i++) {
            int r = 2 * i + (l >> 4), c = l & 15;
            if (hC[l * 8 + i] != refC[r * W + c]) { if (bad2 < 4) printf("  clayout mismatch l=%d i=%d got %d want %d\n", l, i, hC[l * 8 + i], refC[r * W + c]); bad2++; }
        }
    printf("builtin direct (lane-interleaved order): bad=%d ; via C layout: bad=%d\n", bad, bad2);

    // Also check the A-layout hypothesis printed above against the actual bytes.
    (void)as_s8;
    return 0;
}
