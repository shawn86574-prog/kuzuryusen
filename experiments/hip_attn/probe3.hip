// probe3.hip : dump the byte layout of the B row_major fragment (the one used by GEMM2 P*V),
//              so the load can be replaced by a single 8-byte LDS per lane.
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace rocwmma;
#define W 16

// B row_major stored as B[k][n] with ldm = W
__global__ void k_dump_brow(const int8_t* Bm, int32_t* out) {
    fragment<matrix_b, W, W, W, int8_t, row_major> b;
    load_matrix_sync(b, Bm, W);
    int lane = threadIdx.x;
    const unsigned char* p = (const unsigned char*)&b;
    for (int i = 0; i < 8; i++) out[lane * 8 + i] = p[i];
}

int main() {
    std::vector<int8_t> hBm(W * W);
    for (int r = 0; r < W; r++)
        for (int c = 0; c < W; c++) hBm[r * W + c] = (int8_t)(r * W + c);
    int8_t* dBm; int32_t* dOut; int32_t* dSz;
    hipMalloc(&dBm, W * W);
    hipMalloc(&dOut, 32 * 8 * 4);
    hipMalloc(&dSz, 4);
    hipMemcpy(dBm, hBm.data(), W * W, hipMemcpyHostToDevice);
    {
        // also report the fragment size via a tiny kernel
    }
    k_dump_brow<<<1, 32>>>(dBm, dOut);
    hipDeviceSynchronize();
    auto e = hipGetLastError();
    if (e != hipSuccess) { printf("FAILED: %s\n", hipGetErrorString(e)); return 1; }
    std::vector<int32_t> hOut(32 * 8);
    hipMemcpy(hOut.data(), dOut, 32 * 8 * 4, hipMemcpyDeviceToHost);
    printf("B row_major layout (lane: 8 bytes as (idx/16, idx%%16)):\n");
    for (int l = 0; l < 32; l++) {
        printf("  %2d:", l);
        for (int i = 0; i < 8; i++) {
            int v = hOut[l * 8 + i];
            printf(" %3d=(%d,%d)", v, v / 16, v % 16);
        }
        printf("\n");
    }
    return 0;
}
