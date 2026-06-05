#pragma once

#include <cmath>
#include "types.h"
#include "engine.h"

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
);

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
);