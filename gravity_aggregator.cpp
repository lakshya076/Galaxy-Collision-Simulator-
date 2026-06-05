#include "gravity_aggregator.h"

void query_gravity_recursive(
    const OctreeNode* node, 
    const OctreeNode* root, 
    const Star& star, 
    uint32_t target_idx,
    const Star* all_stars, 
    const float* node_masses, 
    const float* node_com_x, 
    const float* node_com_y, 
    const float* node_com_z,
    float G, 
    float epsilon_sq,
    float& ax, float& ay, float& az
) {    
    uint32_t node_idx = get_node_index(root, node);

    if (node_masses[node_idx] == 0.0f) return;

    float s = node->max_x - node->min_x;
    
    float dx = node_com_x[node_idx] - star.x;
    float dy = node_com_y[node_idx] - star.y;
    float dz = node_com_z[node_idx] - star.z;
    float dist_sq = dx * dx + dy * dy + dz * dz;
    
    float theta = 0.8f;
    if (node->is_leaf || (s * s < (theta * theta) * dist_sq)) {
        if (node->is_leaf) {
            // Particle-to-Particle (Direct Gravity) inside leaf node
            for (int i = 0; i < node->star_count; i++) {
                uint32_t s_idx = node->star_indices[i];
                if (s_idx == target_idx) continue;
                float sdx = all_stars[s_idx].x - star.x;
                float sdy = all_stars[s_idx].y - star.y;
                float sdz = all_stars[s_idx].z - star.z;
                
                float r_sq = sdx * sdx + sdy * sdy + sdz * sdz + epsilon_sq;
                float inv_r = 1.0f / std::sqrt(r_sq); 
                float inv_r3 = inv_r * inv_r * inv_r;
                float acc_mag = G * all_stars[s_idx].mass * inv_r3;
                ax += acc_mag * sdx;
                ay += acc_mag * sdy;
                az += acc_mag * sdz;
            }
        } else {
            // Approximation: Treat the whole node as a single macro-particle
            float r_sq = dist_sq + epsilon_sq;
            float r = std::sqrt(r_sq);
            float inv_r3 = 1.0f / (r_sq * r);
            float acc_mag = G * node_masses[node_idx] * inv_r3;
            ax += acc_mag * dx;
            ay += acc_mag * dy;
            az += acc_mag * dz;
        }
    } else {
        // Node is too close; recurse into the 8 children
        for (int i = 0; i < 8; i++) {
            query_gravity_recursive(
                &node->first_child[i], root, star, target_idx,
                all_stars, node_masses, node_com_x, node_com_y, node_com_z,
                G, epsilon_sq, ax, ay, az
            );
        }
    }
}

void query_gravity(
    const OctreeNode* root,
    const Star& star,
    uint32_t target_idx,
    const Star* all_stars,
    const float* node_masses,
    const float* node_com_x,
    const float* node_com_y,
    const float* node_com_z,
    float G,
    float epsilon_sq,
    float& ax, float& ay, float& az
) {
    ax = 0.0f; ay = 0.0f; az = 0.0f;
    
    query_gravity_recursive(
        root, root, star, target_idx,
        all_stars, node_masses, node_com_x, node_com_y, node_com_z,
        G, epsilon_sq, ax, ay, az
    );
}