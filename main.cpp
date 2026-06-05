#include <iostream>
#include <random>
#include <chrono>
#include "generator.h"
#include "engine.h"
#include "arena_allocator.h"

using namespace std;

int main() {
    int num_dm = 50000;
    int num_disc = 90000;
    int num_bulge = 10000;
    
    int particles_per_galaxy = num_dm + num_disc + num_bulge;
    int total_particles = particles_per_galaxy * 2;
    
    Star* stars = new Star[total_particles];
    int num_stars = total_particles;

    mt19937 global_rng(std::random_device{}());

    auto start_time = chrono::high_resolution_clock::now();
    
    cout << "Generating galaxies..." << endl;
    initial_setup(stars, num_dm, num_disc, num_bulge, global_rng);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end_time - start_time;

    cout << "Collision scenario generated successfully!" << std::endl;
    cout << "Total stars spawned: " << total_particles << std::endl;
    cout << "Processing Time: " << diff.count() << " seconds" << endl;

    
    start_time = chrono::high_resolution_clock::now();
    float min_x = numeric_limits<float>::max(), min_y = numeric_limits<float>::max(), min_z = numeric_limits<float>::max();
    float max_x = numeric_limits<float>::lowest(), max_y = numeric_limits<float>::lowest(), max_z = numeric_limits<float>::lowest();

    for (size_t i = 0; i < num_stars; i++) {
        if (stars[i].x < min_x) min_x = stars[i].x;
        if (stars[i].y < min_y) min_y = stars[i].y;
        if (stars[i].z < min_z) min_z = stars[i].z;
        if (stars[i].x > max_x) max_x = stars[i].x;
        if (stars[i].y > max_y) max_y = stars[i].y;
        if (stars[i].z > max_z) max_z = stars[i].z;
    }

    ArenaAllocator arena(1024 * 1024 * 64);
    OctreeNode* root = arena.alloc<OctreeNode>();

    root->min_x = min_x; root->min_y = min_y; root->min_z = min_z;
    root->max_x = max_x; root->max_y = max_y; root->max_z = max_z;
    root->is_leaf = true;
    root->star_count = 0;

    cout << "\nBuilding Basic Octree:" << endl;
    for (uint32_t i = 0; i < num_stars; i++) {
        insert_star(root, i, stars, arena);
    }

    end_time = chrono::high_resolution_clock::now();
    diff = end_time - start_time;

    cout << "\nTotal Initialization Time: " << diff.count() << " seconds" << endl;
    cout << "Arena Memory Consumed: " << (arena.get_used_memory() / 1024.0 / 1024.0) << " MB" << endl;

    const size_t max_nodes = (1024 * 1024 * 64) / sizeof(OctreeNode);
    float* node_masses = new float[max_nodes];
    float* node_com_x = new float[max_nodes];
    float* node_com_y = new float[max_nodes];
    float* node_com_z = new float[max_nodes];

    cout << "\nPopulating Node Metadata" << endl;
    node_physics(root, root, stars, node_masses, node_com_x, node_com_y, node_com_z);

    delete[] node_masses;
    delete[] node_com_x;
    delete[] node_com_y;
    delete[] node_com_z;
    delete[] stars;

    return 0;
}