#define CUDA_CHECK(ans)                                                        \
    {                                                                          \
        gpuAssert((ans), __FILE__, __LINE__);                                  \
    }
CUTE_HOST_DEVICE void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}

// double alpha = std::atan2(1.0, std::abs(f(x)));
// double alpha = std::atan2(1.0, std::abs(2.0 * x));
