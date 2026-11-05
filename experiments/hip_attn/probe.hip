// probe.hip : (1) verify rocwmma int8 16x16x16 path on gfx1100,
//             (2) empirically dump the accumulator fragment register layout.
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#define W 16
using namespace rocwmma;

__global__ void k_matmul_i8(const int8_t* A, const int8_t* Bt, int32_t* C) {
    fragment<matrix_a, W, W, W, int8_t, row_major> a;
    fragment<matrix_b, W, W, W, int8_t, col_major> b;
    fragment<accumulator, W, W, W, int32_t> c;
    fill_fragment(c, 0);
    load_matrix_sync(a, A, W);
    load_matrix_sync(b, Bt, W);
    mma_sync(c, a, b, c);
    store_matrix_sync(C, c, W, mem_row_major);
}

__global__ void k_layout(float* out /* [32*8] */, int32_t* idx /* [32*8] */) {
    __shared__ float x[W * W];
    int lane = threadIdx.x;
    // value = r*16 + c
    for (int i = lane; i < W * W; i += 32) x[i] = (float)(i);
    __syncthreads();
    fragment<accumulator, W, W, W, float> c;
    load_matrix_sync(c, x, W, mem_row_major);
    float const* p = reinterpret_cast<float const*>(&c);
    for (int i = 0; i < 8; i++) out[lane * 8 + i] = p[i];
    // also dump raw storage size
    if (lane == 0) idx[0] = (int32_t)(sizeof(c) / sizeof(float));
}

__global__ void k_scaled_store(const int8_t* A, const int8_t* Bt, float* C) {
    // acc = A*B (int32 frag); scale each element by (row+1) using layout table
    fragment<matrix_a, W, W, W, int8_t, row_major> a;
    fragment<matrix_b, W, W, W, int8_t, col_major> b;
    fragment<accumulator, W, W, W, int32_t> c;
    fill_fragment(c, 0);
    load_matrix_sync(a, A, W);
    load_matrix_sync(b, Bt, W);
    mma_sync(c, a, b, c);
    int32_t* p = reinterpret_cast<int32_t*>(&c);
    int lane = threadIdx.x;
    for (int i = 0; i < 8; i++) {
        // row = ((lane%32) >> 2) * 4 + (i >> 1)   -- hypothesis, printed mapping decides
        // We instead use the empirically dumped mapping via %d: not possible here; use formula below.
        int row = ((lane & 31) / 4) + (i / 2) * 8; // placeholder formula A
        float v = (float)p[i] * (float)(row + 1);
        // store into an float fragment so store_matrix_sync writes it out
        p[i] = (int32_t)v; // not used: we store float separately below
    }
    fragment<accumulator, W, W, W, float> cf;
    float* q = reinterpret_cast<float*>(&cf);
    for (int i = 0; i < 8; i++) {
        int row = ((lane & 31) / 4) + (i / 2) * 8;
        q[i] = (float)reinterpret_cast<int32_t*>(&c)[i] * (float)(row + 1);
    }
    store_matrix_sync(C, cf, W, mem_row_major);
}

int main() {
    int dev = 0;
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, dev);
    printf("Device: %s arch=%s\n", prop.name, prop.gcnArchName);

    std::vector<int8_t> hA(W * W), hBt(W * W);
    std::vector<int32_t> hC(W * W, -1);
    std::vector<int32_t> refC(W * W, 0);
    srand(7);
    for (int r = 0; r < W; r++)
        for (int k = 0; k < W; k++) hA[r * W + k] = (int8_t)((rand() % 9) - 4);
    // B stored transposed: bt[n][k] = B[k][n]
    for (int n = 0; n < W; n++)
        for (int k = 0; k < W; k++) hBt[n * W + k] = (int8_t)((rand() % 9) - 4);
    for (int m = 0; m < W; m++)
        for (int n = 0; n < W; n++) {
            int32_t s = 0;
            for (int k = 0; k < W; k++) s += (int32_t)hA[m * W + k] * (int32_t)hBt[n * W + k];
            refC[m * W + n] = s;
        }

    int8_t *dA, *dBt; int32_t* dC;
    hipMalloc(&dA, W * W); hipMalloc(&dBt, W * W); hipMalloc(&dC, W * W * 4);
    hipMemcpy(dA, hA.data(), W * W, hipMemcpyHostToDevice);
    hipMemcpy(dBt, hBt.data(), W * W, hipMemcpyHostToDevice);
    hipMemset(dC, 0, W * W * 4);
    k_matmul_i8<<<1, 32>>>(dA, dBt, dC);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("matmul kernel FAILED: %s\n", hipGetErrorString(e)); return 1; }
    hipMemcpy(hC.data(), dC, W * W * 4, hipMemcpyDeviceToHost);

    int bad = 0;
    for (int i = 0; i < W * W; i++)
        if (hC[i] != refC[i]) { if (bad < 5) printf("  mismatch idx %d got %d want %d\n", i, hC[i], refC[i]); bad++; }
    printf("int8 rocwmma 16x16x16: %s (bad=%d/%d)\n", bad == 0 ? "OK" : "MISMATCH", bad, W * W);

    // layout dump
    float* dOut; int32_t* dIdx;
    hipMalloc(&dOut, 32 * 8 * 4); hipMalloc(&dIdx, 4);
    k_layout<<<1, 32>>>(dOut, dIdx);
    hipDeviceSynchronize();
    std::vector<float> hOut(32 * 8);
    std::vector<int32_t> hIdx(1);
    hipMemcpy(hOut.data(), dOut, 32 * 8 * 4, hipMemcpyDeviceToHost);
    hipMemcpy(hIdx.data(), dIdx, 4, hipMemcpyDeviceToHost);
    printf("acc fragment float storage elements per lane = %d\n", hIdx[0]);
    printf("lane : elements (row,col)\n");
    for (int l = 0; l < 32; l++) {
        printf("  %2d :", l);
        for (int i = 0; i < 8; i++) {
            int v = (int)hOut[l * 8 + i];
            printf(" (%2d,%2d)", v / 16, v % 16);
        }
        printf("\n");
    }
    return 0;
}
