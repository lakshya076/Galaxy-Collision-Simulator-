#pragma once
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "types.h"
#include "morton.h"

struct alignas(32) BvhNodeTraverse {
    float com_x, com_y, com_z;
    float mass;
    float width;
    int child_left;
    int child_right;
    int padding;
};

struct BvhNodeBuild {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    int flag;
    int parent;
};

__device__ inline int longest_common_prefix(int i, int j, int num_leaves, const uint64_t* keys) {
    if (j < 0 || j >= num_leaves) return -1;
    uint64_t key_i = keys[i];
    uint64_t key_j = keys[j];
    if (key_i == key_j) {
        return 64 + __clz(i ^ j);
    }
    return __clzll(key_i ^ key_j);
}

// Generate Morton Keys & Initial Indices
__global__ void compute_morton_and_indices_kernel(const float* x, const float* y, const float* z, 
                                                  uint64_t* keys, int* indices, 
                                                  int num_stars, float min_val, float max_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_stars) return;
    keys[i] = get_morton_code(x[i], y[i], z[i], min_val, max_val);
    indices[i] = i;
}

// Transpose Sorted Indices back to SoA (In-place sort)
__global__ void apply_sorted_indices_kernel(
    const int* sorted_indices,
    const float* in_x, const float* in_y, const float* in_z,
    const float* in_vx, const float* in_vy, const float* in_vz,
    const float* in_mass, const uint8_t* in_r, const uint8_t* in_g, const uint8_t* in_b, const uint8_t* in_type,
    float* out_x, float* out_y, float* out_z,
    float* out_vx, float* out_vy, float* out_vz,
    float* out_mass, uint8_t* out_r, uint8_t* out_g, uint8_t* out_b, uint8_t* out_type,
    int num_stars
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_stars) return;

    int sorted_idx = sorted_indices[i];
    
    out_x[i] = in_x[sorted_idx];
    out_y[i] = in_y[sorted_idx];
    out_z[i] = in_z[sorted_idx];
    out_vx[i] = in_vx[sorted_idx];
    out_vy[i] = in_vy[sorted_idx];
    out_vz[i] = in_vz[sorted_idx];
    out_mass[i] = in_mass[sorted_idx];
    out_r[i] = in_r[sorted_idx];
    out_g[i] = in_g[sorted_idx];
    out_b[i] = in_b[sorted_idx];
    out_type[i] = in_type[sorted_idx];
}

// Karras Tree Builder
__global__ void build_radix_tree_kernel(int num_leaves, const uint64_t* sorted_keys, BvhNodeTraverse* nodes_tr, BvhNodeBuild* nodes_bd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_leaves - 1) return;

    // Determine direction of range (+1 or -1)
    int d = longest_common_prefix(i, i + 1, num_leaves, sorted_keys) - 
            longest_common_prefix(i, i - 1, num_leaves, sorted_keys);
    d = (d > 0) ? 1 : -1;

    // Compute upper bound for the length of the range
    int delta_min = longest_common_prefix(i, i - d, num_leaves, sorted_keys);
    int l_max = 2;
    while (longest_common_prefix(i, i + l_max * d, num_leaves, sorted_keys) > delta_min) {
        l_max *= 2;
    }

    // Find the other end using binary search
    int l = 0;
    for (int t = l_max / 2; t >= 1; t /= 2) {
        if (longest_common_prefix(i, i + (l + t) * d, num_leaves, sorted_keys) > delta_min) {
            l += t;
        }
    }
    int j = i + l * d;

    // Find the split position using binary search
    int delta_node = longest_common_prefix(i, j, num_leaves, sorted_keys);
    int s = 0;
    int step = l;
    do {
        step = (step + 1) >> 1;
        if (longest_common_prefix(i, i + (s + step) * d, num_leaves, sorted_keys) > delta_node) {
            s += step;
        }
    } while (step > 1);
    
    int split = i + s * d + ((d > 0) ? 0 : -1); 

    // Output child pointers
    int child_left;
    if (min(i, j) == split) {
        child_left = (num_leaves - 1) + split;
    } else {
        child_left = split;
    }

    int child_right;
    if (max(i, j) == split + 1) {
        child_right = (num_leaves - 1) + split + 1;
    } else {
        child_right = split + 1;
    }

    nodes_tr[i].child_left = child_left;
    nodes_tr[i].child_right = child_right;

    nodes_bd[child_left].parent = i;
    nodes_bd[child_right].parent = i;
}

// Initialize Leaves
__global__ void init_leaves_kernel(int num_leaves, BvhNodeTraverse* nodes_tr, BvhNodeBuild* nodes_bd, 
                                   const float* x, const float* y, const float* z, 
                                   const float* mass) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_leaves) return;

    int leaf_idx = (num_leaves - 1) + i;
    
    float mx = x[i];
    float my = y[i];
    float mz = z[i];
    float m = mass[i];

    nodes_bd[leaf_idx].min_x = mx;
    nodes_bd[leaf_idx].min_y = my;
    nodes_bd[leaf_idx].min_z = mz;
    nodes_bd[leaf_idx].max_x = mx;
    nodes_bd[leaf_idx].max_y = my;
    nodes_bd[leaf_idx].max_z = mz;
    
    nodes_tr[leaf_idx].com_x = mx;
    nodes_tr[leaf_idx].com_y = my;
    nodes_tr[leaf_idx].com_z = mz;
    nodes_tr[leaf_idx].mass = m;
    nodes_tr[leaf_idx].width = 0.0f;
    
    nodes_tr[leaf_idx].child_left = -1;
    nodes_tr[leaf_idx].child_right = -1;
    nodes_bd[leaf_idx].flag = 0;
    nodes_bd[leaf_idx].parent = -1;
}

// Initialize internal node flags to 0
__global__ void init_internal_nodes_kernel(int num_leaves, BvhNodeTraverse* nodes_tr, BvhNodeBuild* nodes_bd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_leaves - 1) return;
    nodes_bd[i].flag = 0;
    nodes_bd[i].parent = -1;
}

// Bottom-Up Aggregation
__global__ void aggregate_bvh_kernel(int num_leaves, BvhNodeTraverse* nodes_tr, BvhNodeBuild* nodes_bd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_leaves) return;

    int current = (num_leaves - 1) + i; // start at leaf i
    current = nodes_bd[current].parent;

    while (current != -1) {
        int old_flag = atomicAdd(&nodes_bd[current].flag, 1);
        if (old_flag == 0) {
            // First thread to arrive, die
            return;
        }

        int left = nodes_tr[current].child_left;
        int right = nodes_tr[current].child_right;

        float min_x = min(nodes_bd[left].min_x, nodes_bd[right].min_x);
        float min_y = min(nodes_bd[left].min_y, nodes_bd[right].min_y);
        float min_z = min(nodes_bd[left].min_z, nodes_bd[right].min_z);
        float max_x = max(nodes_bd[left].max_x, nodes_bd[right].max_x);
        float max_y = max(nodes_bd[left].max_y, nodes_bd[right].max_y);
        float max_z = max(nodes_bd[left].max_z, nodes_bd[right].max_z);
        
        nodes_bd[current].min_x = min_x;
        nodes_bd[current].min_y = min_y;
        nodes_bd[current].min_z = min_z;
        nodes_bd[current].max_x = max_x;
        nodes_bd[current].max_y = max_y;
        nodes_bd[current].max_z = max_z;
        
        nodes_tr[current].width = max(max_x - min_x, max(max_y - min_y, max_z - min_z));

        float m_left = nodes_tr[left].mass;
        float m_right = nodes_tr[right].mass;
        float m_total = m_left + m_right;
        
        nodes_tr[current].mass = m_total;
        if (m_total > 0.0f) {
            nodes_tr[current].com_x = (nodes_tr[left].com_x * m_left + nodes_tr[right].com_x * m_right) / m_total;
            nodes_tr[current].com_y = (nodes_tr[left].com_y * m_left + nodes_tr[right].com_y * m_right) / m_total;
            nodes_tr[current].com_z = (nodes_tr[left].com_z * m_left + nodes_tr[right].com_z * m_right) / m_total;
        } else {
            nodes_tr[current].com_x = (min_x + max_x) * 0.5f;
            nodes_tr[current].com_y = (min_y + max_y) * 0.5f;
            nodes_tr[current].com_z = (min_z + max_z) * 0.5f;
        }

        current = nodes_bd[current].parent;
    }
}
