#include "engine.h"

using namespace std;

void set_child_bounds(OctreeNode* parent, OctreeNode* child, int index) {
    float mid_x = (parent->min_x + parent->max_x) / 2.0f;
    float mid_y = (parent->min_y + parent->max_y) / 2.0f;
    float mid_z = (parent->min_z + parent->max_z) / 2.0f;

    child->min_x = (index & 1) ? mid_x : parent->min_x;
    child->max_x = (index & 1) ? parent->max_x : mid_x;

    child->min_y = (index & 2) ? mid_y : parent->min_y;
    child->max_y = (index & 2) ? parent->max_y : mid_y;

    child->min_z = (index & 4) ? mid_z : parent->min_z;
    child->max_z = (index & 4) ? parent->max_z : mid_z;

    child->is_leaf = true;
    child->star_count = 0;
}

int get_child_index(OctreeNode* node, float x, float y, float z) {
    float mid_x = (node->min_x + node->max_x) / 2.0f;
    float mid_y = (node->min_y + node->max_y) / 2.0f;
    float mid_z = (node->min_z + node->max_z) / 2.0f;

    int idx = 0;
    idx |= (x >= mid_x) ? 1 : 0;
    idx |= (y >= mid_y) ? 2 : 0;
    idx |= (z >= mid_z) ? 4 : 0;
    return idx;
}

uint32_t get_node_index(const OctreeNode* root, const OctreeNode* node) {
    return static_cast<uint32_t>(node - root);
}

void insert_star(OctreeNode* node, uint32_t star_idx, const Star* all_stars, ArenaAllocator& arena) {
    if (node->is_leaf) {
        if (node->star_count < 8) {
            node->star_indices[node->star_count++] = star_idx;
        } else {
            node->is_leaf = false;
            
            uint32_t old_stars[8];
            for (int i = 0; i < 8; i++) {
                old_stars[i] = node->star_indices[i];
            }

            // Allocate 512 bytes for 8 contiguous children
            OctreeNode* children = arena.alloc_array<OctreeNode>(8);
            node->first_child = children;

            for (int i = 0; i < 8; i++) {
                set_child_bounds(node, &children[i], i);
            }

            // Route the original 8 stars into new children
            for (int i = 0; i < 8; i++) {
                uint32_t s_idx = old_stars[i];
                int c_idx = get_child_index(node, all_stars[s_idx].x, all_stars[s_idx].y, all_stars[s_idx].z);
                insert_star(&children[c_idx], s_idx, all_stars, arena);
            }

            // Route the new star
            int c_idx = get_child_index(node, all_stars[star_idx].x, all_stars[star_idx].y, all_stars[star_idx].z);
            insert_star(&children[c_idx], star_idx, all_stars, arena);
        }
    } else {
        int c_idx = get_child_index(node, all_stars[star_idx].x, all_stars[star_idx].y, all_stars[star_idx].z);
        insert_star(&node->first_child[c_idx], star_idx, all_stars, arena);
    }
}


void node_physics(OctreeNode* node, OctreeNode* root, const Star* stars, float* node_masses, 
    float* node_com_x, float* node_com_y, float* node_com_z) {
        uint32_t idx = get_node_index(root, node);

        if (node->is_leaf) {
            float total_mass = 0.0f;
            float weighted_x = 0.0f, weighted_y = 0.0f, weighted_z = 0.0f;

            for (int i = 0; i < node->star_count; i++) {
                uint32_t star_idx = node->star_indices[i];
                float curr_mass = stars[star_idx].mass;
                total_mass += curr_mass;
                weighted_x += stars[star_idx].x * curr_mass;
                weighted_y += stars[star_idx].y * curr_mass;
                weighted_z += stars[star_idx].z * curr_mass;
            }

            node_masses[idx] = total_mass;

            if(total_mass > 0.0f) {
                node_com_x[idx] = weighted_x / total_mass;
                node_com_y[idx] = weighted_y / total_mass;
                node_com_z[idx] = weighted_z / total_mass;
            } else {
                node_com_x[idx] = (node->min_x + node->max_x) * 0.5f;
                node_com_y[idx] = (node->min_y + node->max_y) * 0.5f;
                node_com_z[idx] = (node->min_z + node->max_z) * 0.5f;
            }
        } else {
            float total_mass = 0.0f;
            float weighted_x = 0.0f, weighted_y = 0.0f, weighted_z = 0.0f;
            
            for (int i = 0; i < 8; ++i) {
                OctreeNode* child = &node->first_child[i];
                node_physics(child, root, stars, node_masses, node_com_x, node_com_y, node_com_z);

                uint32_t child_idx = get_node_index(root, child);
                float cm = node_masses[child_idx];
                total_mass += cm;
                weighted_x += node_com_x[child_idx] * cm;
                weighted_y += node_com_y[child_idx] * cm;
                weighted_z += node_com_z[child_idx] * cm;
            }

            node_masses[idx] = total_mass;
            if (total_mass > 0.0f) {
                node_com_x[idx] = weighted_x / total_mass;
                node_com_y[idx] = weighted_y / total_mass;
                node_com_z[idx] = weighted_z / total_mass;
            } else {
                node_com_x[idx] = (node->min_x + node->max_x) * 0.5f;
                node_com_y[idx] = (node->min_y + node->max_y) * 0.5f;
                node_com_z[idx] = (node->min_z + node->max_z) * 0.5f;
            }
        }
}