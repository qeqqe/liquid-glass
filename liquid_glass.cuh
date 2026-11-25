#pragma once

#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>

enum Equations { CONCAVE, CONVEX_SQUIRCLE, CONVEX_CIRCLE, LIP };

#define CUDA_CHECK(ans)                                                        \
    {                                                                          \
        gpuAssert((ans), __FILE__, __LINE__);                                  \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "GPUassert: %s %s %d\n",
                     cudaGetErrorString(code), file, line);
    }
}

void launchLiquidGlass(const float *d_src, float *d_dst, float radius,
                       int pos_x, int pos_y, size_t h, size_t w, int channels,
                       Equations eq, float ref_idx, float k = 0.2f,
                       cudaStream_t stream = 0);

// double alpha = std::atan2(1.0, std::abs(f(x)));
// double alpha = std::atan2(1.0, std::abs(2.0 * x));
