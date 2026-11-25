
#define CUDA_CHECK(ans)                                                        \
    {                                                                          \
        gpuAssert((ans), __FILE__, __LINE__);                                  \
    }
CUTE_HOST_DEVICE void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}
