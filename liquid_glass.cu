#include <cmath>
#include <cstddef>
#include <cstdint>

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

} // namespace

enum Equations { CONCAVE, CONVEX_SQUIRCLE, CONVEX_CIRCLE, LIP };

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

__global__ void applyLiquidGlass(const float *d_src, float *d_dst, float radius,
                                 int pos_x, int pos_y, size_t h, size_t w,
                                 Equations eq, float ref_idx, float k = 0.2f) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= static_cast<int>(w) || y >= static_cast<int>(h)) {
        return;
    }

    float mid_x = 0.5f * static_cast<float>(w) + static_cast<float>(pos_x);
    float mid_y = 0.5f * static_cast<float>(h) + static_cast<float>(pos_y);

    float dx = static_cast<float>(x) - mid_x;
    float dy = static_cast<float>(y) - mid_y;
    float dist = hypotf(dx, dy); // distance from the mid of the circle

    if (dist > radius) {
        return;
    }

    float dist_from_edge = radius - dist;
    SurfaceSample s = sampleProfile(radius, k, dist_from_edge, eq);

    // angle of incidence from the surface normal (signed: >0 convex, <0
    // concave)
    float theta_i = atanf(s.slope);

    // float alpha = atan2f(1.0f, fabsf(s.slope)); // angle to the tangent

    float sin_t = clampf(sinf(theta_i) / ref_idx, -1.0f, 1.0f);
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
}
