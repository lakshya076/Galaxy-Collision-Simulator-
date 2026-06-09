#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

// Spreads 21 consecutive bits out into every 3rd bit of a 64-bit integer
inline CUDA_CALLABLE_MEMBER uint64_t expand_bits(uint32_t v) {
    uint64_t x = v & 0x1FFFFF; // 21 bits
    x = (x | x << 32) & 0x1F00000000FFFF;
    x = (x | x << 16) & 0x1F0000FF0000FF;
    x = (x | x << 8)  & 0x100F00F00F00F00F;
    x = (x | x << 4)  & 0x10c30c30c30c30c3;
    x = (x | x << 2)  & 0x1249249249249249;
    return x;
}

// Maps 3D coordinates into a single 64-bit Z-order curve Morton Code
inline CUDA_CALLABLE_MEMBER uint64_t get_morton_code(float x, float y, float z, float min_val, float max_val) {
    float scale = 2097151.0f / (max_val - min_val); // Map to 21-bit space (0 to 2^21 - 1)
    uint32_t ix = static_cast<uint32_t>((x - min_val) * scale);
    uint32_t iy = static_cast<uint32_t>((y - min_val) * scale);
    uint32_t iz = static_cast<uint32_t>((z - min_val) * scale);
    return (expand_bits(ix) << 2) | (expand_bits(iy) << 1) | expand_bits(iz);
}
