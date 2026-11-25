#include "./liquid_glass.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

struct SurfaceSample {
    float height;
    float slope;
};

__host__ __device__ inline float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__host__ __device__ inline float smootherstep(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

__host__ __device__ inline float smootherstepDeriv(float t) {
    float tm1 = t - 1.0f;
    return 30.0f * t * t * tm1 * tm1;
}

__host__ __device__ inline SurfaceSample unitCircle(float t) {
    float u = 1.0f - t;
    float val = 1.0f - u * u;
    if (val <= 0.0f) {
        return {0.0f, INFINITY};
    }
    float h = sqrtf(val);
    return {h, u / h};
}

__host__ __device__ inline SurfaceSample unitTrough(float t) {
    SurfaceSample c = unitCircle(t);
    return {1.0f - c.height, -c.slope};
}

__host__ __device__ inline SurfaceSample unitSquircle4(float t) {
    float u = 1.0f - t;
    float u2 = u * u;
    float val = 1.0f - u2 * u2;
    if (val <= 0.0f) {
        return {0.0f, INFINITY};
    }
    float h = sqrtf(sqrtf(val));
    float ratio = u / h;
    return {h, ratio * ratio * ratio};
}

__host__ __device__ inline SurfaceSample unitLip(float t) {
    SurfaceSample c = unitCircle(t);
    float w = smootherstep(t);
    float dw = smootherstepDeriv(t);
    float h = (1.0f - w) * c.height + w * (1.0f - c.height);
    float dh = (1.0f - 2.0f * w) * c.slope + dw * (1.0f - 2.0f * c.height);
    return {h, dh};
}

__device__ inline void sampleBilinearWrite(const float *__restrict__ d_src,
                                           float *__restrict__ d_dst,
                                           size_t out_idx, float sample_x,
                                           float sample_y, size_t w, size_t h,
                                           int channels) {
    int wi = static_cast<int>(w);
    int hi = static_cast<int>(h);

    float sx = clampf(sample_x, 0.0f, static_cast<float>(wi - 1));
    float sy = clampf(sample_y, 0.0f, static_cast<float>(hi - 1));

    int x0 = static_cast<int>(floorf(sx));
    int y0 = static_cast<int>(floorf(sy));
    int x1 = (x0 + 1 < wi) ? (x0 + 1) : x0;
    int y1 = (y0 + 1 < hi) ? (y0 + 1) : y0;

    float fx = sx - static_cast<float>(x0);
    float fy = sy - static_cast<float>(y0);

    size_t base00 =
        (static_cast<size_t>(y0) * w + static_cast<size_t>(x0)) * channels;
    size_t base10 =
        (static_cast<size_t>(y0) * w + static_cast<size_t>(x1)) * channels;
    size_t base01 =
        (static_cast<size_t>(y1) * w + static_cast<size_t>(x0)) * channels;
    size_t base11 =
        (static_cast<size_t>(y1) * w + static_cast<size_t>(x1)) * channels;

    for (int c = 0; c < channels; ++c) {
        float v00 = d_src[base00 + c];
        float v10 = d_src[base10 + c];
        float v01 = d_src[base01 + c];
        float v11 = d_src[base11 + c];

        float top = fmaf(fx, v10 - v00, v00);
        float bot = fmaf(fx, v11 - v01, v01);
        d_dst[out_idx + c] = fmaf(fy, bot - top, top);
    }
}

} // namespace

__host__ __device__ SurfaceSample sampleProfile(float radius, float k,
                                                float dist_from_edge,
                                                Equations eq) {
    float r = radius * (1.0f - k);
    if (r <= 0.0f) {
        return {0.0f, 0.0f};
    }
    float t = clampf(dist_from_edge / r, 0.0f, 1.0f);

    SurfaceSample s = {0.0f, 0.0f};
    switch (eq) {
    case CONCAVE:
        s = unitTrough(t);
        break;
    case CONVEX_SQUIRCLE:
        s = unitSquircle4(t);
        break;
    case CONVEX_CIRCLE:
        s = unitCircle(t);
        break;
    case LIP:
        s = unitLip(t);
        break;
    }

    return {r * s.height, s.slope};
}

__global__ void applyLiquidGlass(const float *__restrict__ d_src,
                                 float *__restrict__ d_dst, float radius,
                                 int pos_x, int pos_y, size_t h, size_t w,
                                 int channels, Equations eq, float ref_idx,
                                 float k = 0.2f) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= static_cast<int>(w) || y >= static_cast<int>(h)) {
        return;
    }

    size_t out_idx =
        (static_cast<size_t>(y) * w + static_cast<size_t>(x)) * channels;

    float mid_x = 0.5f * static_cast<float>(w) + static_cast<float>(pos_x);
    float mid_y = 0.5f * static_cast<float>(h) + static_cast<float>(pos_y);

    float dx = static_cast<float>(x) - mid_x;
    float dy = static_cast<float>(y) - mid_y;
    float dist = hypotf(dx, dy); // distance from the mid of the circle, sdf

    if (dist > radius) {
        for (int c = 0; c < channels; ++c) {
            d_dst[out_idx + c] = d_src[out_idx + c];
        }
        return;
    }

    // for a rectangular (rounded) glass (like macos/ios) we can  swap
    // `dist_from_edge` for the negative of a 2D rounded box SDF, and replace
    // (-dx/dist, -dy/dist) with the normalized negative gradient of that SDF
    float dist_from_edge = radius - dist;
    SurfaceSample s = sampleProfile(radius, k, dist_from_edge, eq);

    // float alpha = atan2f(1.0f, fabsf(s.slope)); // angle to the tangent

    // angle of incidence from the surface normal (signed: >0 convex, <0
    // concave)
    float theta_i = atanf(s.slope);

    float inv_ref = (ref_idx > 0.0f) ? (1.0f / ref_idx) : 1.0f;
    float sin_t = clampf(sinf(theta_i) * inv_ref, -1.0f, 1.0f);
    float theta_t = asinf(sin_t);
    float displacement = s.height * tanf(theta_i - theta_t);

    float sample_x = static_cast<float>(x);
    float sample_y = static_cast<float>(y);
    if (dist > 0.0f) {
        float inv_dist = 1.0f / dist;
        sample_x += (-dx * inv_dist) * displacement;
        sample_y += (-dy * inv_dist) * displacement;
    }

    // okay so now we take the d_src at (sample_x, sample_y) with bilinear
    // filtering and write to d_dst[y * w + x]
    sampleBilinearWrite(d_src, d_dst, out_idx, sample_x, sample_y, w, h,
                        channels);
}

void launchLiquidGlass(const float *d_src, float *d_dst, float radius,
                       int pos_x, int pos_y, size_t h, size_t w, int channels,
                       Equations eq, float ref_idx, float k,
                       cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((static_cast<unsigned int>(w) + block.x - 1) / block.x,
              (static_cast<unsigned int>(h) + block.y - 1) / block.y);
    applyLiquidGlass<<<grid, block, 0, stream>>>(
        d_src, d_dst, radius, pos_x, pos_y, h, w, channels, eq, ref_idx, k);
}
