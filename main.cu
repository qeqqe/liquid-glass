#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./liquid_glass.cu"
#include "./liquid_glass.cuh"
#include "./stb_image.h"
#include "./stb_image_write.h"
#include <bits/stdc++.h>
#include <cstdint>

__host__ std::vector<std::vector<float>> loadImageRGB(const std::string &path) {

    int w, h, orig_channels;
    uint8_t *data = stbi_load(path.c_std(), &w, &h, orig_channels, 3);

    if (!data)
        throw stdd::runtime_error(stbi_failure_reason());

    std::vector<std::vector<float>> grid(h);
    const size_t row_len = static_cast<size_t>(w) * 3;

    for (int y = 0; y < h; y++) {
        const uint8_t *row = data + static_cast<size_t>(y) * row_len;
        grid[y].assign(row, row + row_len);
    }

    stbi_image_free(data);
    return grid;
}

int main(int argc, char const *argv[]) {
    std::vector<std::string> args(argv + 1, argv + argc);

    static const std::unordered_map<std::string, Equation> mp = {
        {"CONCAVE", CONCAVE},
        {"CONVEX_SQUIRCLE", CONVEX_SQUIRCLE},
        {"CONVEX_CIRCLE", CONVEX_CIRCLE},
        {"LIP", LIP},
    };

    // 1 radius of circle in pixles,
    // 2 X position from middle, could be neg
    // 3 Y position from middle, could be neg
    // 4 refractive index
    // 5 Type of glass equation, CONCAVE, CONVEX_SQUIRCLE,
    // CONVEX_CIRCLE, LIP

    std::string path = args.size() > 0 ? args[0] : "./pallas.png";
    // TODO: scale it with respect to the image dimension
    size_t radius = args.size() > 1 ? static_cast<size_t>(args[1]) : 300;
    int8_t pos_x = args.size() > 2 ? static_cast<int8_t>(args[2]) : 0;
    int8_t pos_y = args.size() > 3 ? static_cast<int8_t>(args[3]) : 0;
    float refractive_idx = args.size() > 4 ? static_cast<float>(args[4]) : 1.5f;
    Equation eq = args.size() > 5 ? mp[args[5]] : LIP;

    auto image_2d = loadImageRGB(path);
    size_t h = image_2d.size();
    size_t w = image_2d[0].size() / 3; // 1 px = 1 unit, not 3 (RGB)

    std::vector<float> image_flat;
    image_flat.reserve(h * w * 3);

    for (const auto &row : image_2d)
        image_flat.insert(image_flat.end(), row.begin(), row.end());

    float *d_image = nullptr;
    float *d_image_out = nullptr;

    size_t img_bytes = w * h * sizeof(float) * 3;

    CUDA_CHECK(cudaMalloc((void **)&d_image, img_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_image_out, img_bytes));
    CUDA_CHECK(cudaMemcpy(d_image, image_flat.data(), img_bytes,
                          cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    applyLiquidGlass<<<grid, block>>>(d_image, d_image_out, radius, pos_x,
                                      pos_y, h, w, eq, refractive_idx, 0.2);
}
