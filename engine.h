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

using namespace std;

struct QueryResult {
    uint32_t star_idx;
    float dist_sq;
    bool operator<(const QueryResult& other) const {
        return dist_sq < other.dist_sq;
    }
};

float box_distance_squared(OctreeNode* node, float tx, float ty, float tz);

void query_radius_recursive(OctreeNode* node, float tx, float ty, float tz, float r_sq, const Star* all_stars, vector<QueryResult>& results);

void query_knn_recursive(OctreeNode* node, float tx, float ty, float tz, int k, const Star* all_stars, priority_queue<QueryResult>& heap);

void set_child_bounds(OctreeNode* parent, OctreeNode* child, int index);

int get_child_index(OctreeNode* node, float x, float y, float z);

void insert_star(OctreeNode* node, uint32_t star_idx, const Star* all_stars, ArenaAllocator& arena);