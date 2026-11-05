// probe4.hip : measure the achievable int8 16x16x16 WMMA throughput on this gfx1100
//              with operands held in registers (no memory traffic, no conversions).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));

__global__ __launch_bounds__(256, 1) void k_peak(int iters, int nacc, int* out) {
    i32x4 av = {1, 2, 3, 4};
    i32x4 bv = {5, 6, 7, 8};
    i32x8 c[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) c[j] = (i32x8){0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            if (j >= nacc) break;
            c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, av, false, bv, c[j], false);
        }
    }
    int s = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) s += c[j][0] + c[j][7];
    if (s == 12345) out[threadIdx.x] = s;
    if (threadIdx.x == 0) out[0] = s;
}

static void run(int grid, int iters, int nacc, const char* tag) {
    int* out; hipMalloc(&out, 1024 * 4);
    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    k_peak<<<grid, 256>>>(iters, nacc, out);      // warmup
    hipDeviceSynchronize();
    hipEventRecord(e0);
    k_peak<<<grid, 256>>>(iters, nacc, out);
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float ms; hipEventElapsedTime(&ms, e0, e1);
    double mma = (double)grid * 8 * iters * nacc;          // warps = 8 per CTA
    double mac = mma * 16.0 * 16 * 16;
    printf("%-28s grid=%3d nacc=%d : %7.3f ms  %7.1f TOPS  (%5.2f cycles/wmma/SIMD @2.5GHz)\n",
           tag, grid, nacc, ms, mac * 2 / (ms * 1e-3) / 1e12,
           2.5e9 * ms * 1e-3 * 4.0 / (double)(iters * nacc));
    hipFree(out); hipEventDestroy(e0); hipEventDestroy(e1);
}

int main() {
    int dev; hipGetDevice(&dev);
    hipDeviceProp_t p; hipGetDeviceProperties(&p, dev);
    printf("%s  CUs=%d\n", p.name, p.multiProcessorCount);
    run(96,    20000, 8, "1 CTA/CU, 8 acc");
    run(96 * 2, 20000, 8, "2 CTA/CU, 8 acc");
    run(96 * 4, 20000, 8, "4 CTA/CU, 8 acc");
    run(96 * 2, 20000, 4, "2 CTA/CU, 4 acc");
    run(96 * 2, 20000, 2, "2 CTA/CU, 2 acc");
    run(96 * 2, 20000, 1, "2 CTA/CU, 1 acc");
    return 0;
}
