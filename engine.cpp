#include "engine.h"

using namespace std;


float box_distance_squared(OctreeNode* node, float tx, float ty, float tz) {
    float closest_x = max(node->min_x, min(tx, node->max_x));
    float closest_y = max(node->min_y, min(ty, node->max_y));
    float closest_z = max(node->min_z, min(tz, node->max_z));

    float dx = closest_x - tx;
    float dy = closest_y - ty;
    float dz = closest_z - tz;
    return (dx * dx) + (dy * dy) + (dz * dz);
}

void query_radius_recursive(OctreeNode* node, float tx, float ty, float tz, float r_sq, const Star* all_stars, vector<QueryResult>& results) {
    if (box_distance_squared(node, tx, ty, tz) > r_sq) return;

    if (node->is_leaf) {
        for (int i = 0; i < node->star_count; i++) {
            uint32_t s_idx = node->star_indices[i];
            const Star& s = all_stars[s_idx];
            float dx = s.x - tx; float dy = s.y - ty; float dz = s.z - tz;
            float dist_sq = (dx*dx) + (dy*dy) + (dz*dz);
            if (dist_sq <= r_sq) {
                results.push_back({s_idx, dist_sq});
            }
        }
    } else {
        for (int i = 0; i < 8; i++) {
            query_radius_recursive(&node->first_child[i], tx, ty, tz, r_sq, all_stars, results);
        }
    }
}

void query_knn_recursive(OctreeNode* node, float tx, float ty, float tz, int k, const Star* all_stars, priority_queue<QueryResult>& heap) {
    float current_r_sq = (heap.size() == k) ? heap.top().dist_sq : numeric_limits<float>::max();
    
    if (box_distance_squared(node, tx, ty, tz) > current_r_sq) return;

    if (node->is_leaf) {
        for (int i = 0; i < node->star_count; i++) {
            uint32_t s_idx = node->star_indices[i];
            const Star& s = all_stars[s_idx];
            float dx = s.x - tx; float dy = s.y - ty; float dz = s.z - tz;
            float dist_sq = (dx*dx) + (dy*dy) + (dz*dz);
            
            if (heap.size() < k) {
                heap.push({s_idx, dist_sq});
            } else if (dist_sq < heap.top().dist_sq) {
                heap.pop();
                heap.push({s_idx, dist_sq});
            }
        }
    } else {
        for (int i = 0; i < 8; i++) {
            query_knn_recursive(&node->first_child[i], tx, ty, tz, k, all_stars, heap);
        }
    }
}

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
