#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include "types.h"
#include "arena_allocator.h"

void set_child_bounds(OctreeNode* parent, OctreeNode* child, int index);

int get_child_index(OctreeNode* node, float x, float y, float z);

uint32_t get_node_index(const OctreeNode* root, const OctreeNode* node);

void insert_star(OctreeNode* node, uint32_t star_idx, const Star* all_stars, ArenaAllocator& arena);

void node_physics(OctreeNode* node, OctreeNode* root, const Star* stars, float* node_masses, 
    float* node_com_x, float* node_com_y, float* node_com_z);