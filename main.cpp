#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <omp.h>
#include "generator.h"
#include "engine.h"
#include "arena_allocator.h"
#include "gravity_aggregator.h"
#include <algorithm>
#include <parallel/algorithm>
#include "morton.h"
#include "renderer.h"
#include <cstdio>
#include <string>

using namespace std;

struct Accel {
    float ax, ay, az;
};

int main(int argc, char** argv) {
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
    
#ifndef MODE_BAKE
    // Initialize OpenGL for Live or Playback mode
    GLFWwindow* window = init_opengl(1920, 1080);
    if (!window) {
        return -1;
    }
    #if defined(MODE_PLAYBACK)
        int num_visible = 2 * (num_disc + num_bulge);
        setup_buffers(stars, num_visible);
    #else
        setup_buffers(stars, num_stars);
    #endif
#else
    GLFWwindow* window = nullptr;
#endif

    // Hoist large allocations
    ArenaAllocator arena(1024 * 1024 * 64);
    const size_t max_nodes = (1024 * 1024 * 64) / sizeof(OctreeNode);
    uint32_t* leaf_star_indices = new uint32_t[max_nodes * 32];
    float* node_masses = new float[max_nodes];
    float* node_com_x = new float[max_nodes];
    float* node_com_y = new float[max_nodes];
    float* node_com_z = new float[max_nodes];
    std::vector<Accel> accels(num_stars);

    float G = 1.0f;
    float epsilon_sq = 1e-3f;
    float dt = 0.01f;

    cout << "\nStarting Simulation..." << endl;
    
    int frame_count = 0;

#if defined(MODE_BAKE)
    cout << "Writing raw simulation data to simulation.bin..." << endl;
    FILE* out_file = fopen("simulation.bin", "wb");
    if (!out_file) {
        cerr << "Failed to open simulation.bin for writing" << endl;
        return -1;
    }

    int total_frames = 1000;
    if (argc > 1) {
        total_frames = std::stoi(argv[1]);
        cout << "Baking " << total_frames << " frames." << endl;
    }

    double total_time = 0;
    for (frame_count = 0; frame_count < total_frames; ++frame_count) {
        start_time = chrono::high_resolution_clock::now();

        // Calculate Bounds
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

        float global_min = std::min({min_x, min_y, min_z});
        float global_max = std::max({max_x, max_y, max_z});

        // Morton Sort
        __gnu_parallel::sort(stars, stars + num_stars, [global_min, global_max](const Star& a, const Star& b) {
            return get_morton_code(a.x, a.y, a.z, global_min, global_max) < 
                   get_morton_code(b.x, b.y, b.z, global_min, global_max);
        });

        // Reset Arena and Build Octree
        arena.reset();
        OctreeNode* root = arena.alloc<OctreeNode>();

        float max_span = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
        root->half_width = max_span * 0.5f;
        root->center_x = (min_x + max_x) * 0.5f;
        root->center_y = (min_y + max_y) * 0.5f;
        root->center_z = (min_z + max_z) * 0.5f;
        root->is_leaf = true;
        root->star_count = 0;

        uint32_t next_free_idx = 0;
        root->start_star_index = next_free_idx;
        next_free_idx += 32;

        for (uint32_t i = 0; i < num_stars; i++) {
            insert_star(root, i, stars, arena, leaf_star_indices, next_free_idx);
        }

        // Metadata Population
        node_physics(root, root, stars, node_masses, node_com_x, node_com_y, node_com_z, leaf_star_indices);

        // Gravity Queries
        #pragma omp parallel for schedule(dynamic, 256)
        for (int i = 0; i < num_stars; ++i) {
            query_gravity(
                root, stars[i], i, stars, 
                node_masses, node_com_x, node_com_y, node_com_z, leaf_star_indices,
                G, epsilon_sq, accels[i].ax, accels[i].ay, accels[i].az
            );
        }

        // Velocity / Position Integration
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_stars; ++i) {
            stars[i].vx += accels[i].ax * dt;
            stars[i].vy += accels[i].ay * dt;
            stars[i].vz += accels[i].az * dt;

            stars[i].x += stars[i].vx * dt;
            stars[i].y += stars[i].vy * dt;
            stars[i].z += stars[i].vz * dt;
        }

        // Dump to file (visible stars only in PlaybackStar format)
        std::vector<PlaybackStar> write_buf;
        write_buf.reserve(num_stars);
        for (int i = 0; i < num_stars; ++i) {
            if (!stars[i].is_dm) {
                write_buf.push_back({stars[i].x, stars[i].y, stars[i].z, stars[i].r, stars[i].g, stars[i].b, 0});
            }
        }
        fwrite(write_buf.data(), sizeof(PlaybackStar), write_buf.size(), out_file);

        end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> diff = end_time - start_time;
        total_time += diff.count();

        if (frame_count % 10 == 0) {
            cout << "Baked " << frame_count << "/" << total_frames << " frames (" << total_time << "s)" << endl;
        }
    }
    fclose(out_file);
    cout << "Baking Complete in " << total_time << "s" << endl;

#elif defined(MODE_PLAYBACK)
    cout << "Reading from simulation.bin..." << endl;
    FILE* in_file = fopen("simulation.bin", "rb");
    if (!in_file) {
        cerr << "Failed to open simulation.bin! Run BAKE mode first." << endl;
        return -1;
    }

    int num_visible = 2 * (num_disc + num_bulge);
    std::vector<PlaybackStar> read_buffer(num_visible);

    while (!glfwWindowShouldClose(window)) {
        process_input(window);

        size_t read_elements = fread(read_buffer.data(), sizeof(PlaybackStar), num_visible, in_file);
        if (read_elements != num_visible) {
            // Loop playback if EOF reached
            fseek(in_file, 0, SEEK_SET);
            fread(read_buffer.data(), sizeof(PlaybackStar), num_visible, in_file);
        }

        // Copy positions and colors to stars array for rendering
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_visible; ++i) {
            stars[i].x = read_buffer[i].x;
            stars[i].y = read_buffer[i].y;
            stars[i].z = read_buffer[i].z;
            stars[i].r = read_buffer[i].r;
            stars[i].g = read_buffer[i].g;
            stars[i].b = read_buffer[i].b;
            stars[i].is_dm = false;
        }

        update_vbo(stars, num_visible);
        render_frame(num_visible);
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    fclose(in_file);
    cleanup_opengl();

#else
    while (!glfwWindowShouldClose(window)) {
        start_time = chrono::high_resolution_clock::now();
        
        process_input(window);

        // Calculate Bounds
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

        float global_min = std::min({min_x, min_y, min_z});
        float global_max = std::max({max_x, max_y, max_z});

        // Morton Sort
        __gnu_parallel::sort(stars, stars + num_stars, [global_min, global_max](const Star& a, const Star& b) {
            return get_morton_code(a.x, a.y, a.z, global_min, global_max) < 
                   get_morton_code(b.x, b.y, b.z, global_min, global_max);
        });

        // Reset Arena and Build Octree
        arena.reset();
        OctreeNode* root = arena.alloc<OctreeNode>();

        float max_span = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
        root->half_width = max_span * 0.5f;
        root->center_x = (min_x + max_x) * 0.5f;
        root->center_y = (min_y + max_y) * 0.5f;
        root->center_z = (min_z + max_z) * 0.5f;
        root->is_leaf = true;
        root->star_count = 0;

        uint32_t next_free_idx = 0;
        root->start_star_index = next_free_idx;
        next_free_idx += 32;

        for (uint32_t i = 0; i < num_stars; i++) {
            insert_star(root, i, stars, arena, leaf_star_indices, next_free_idx);
        }

        // Metadata Population
        node_physics(root, root, stars, node_masses, node_com_x, node_com_y, node_com_z, leaf_star_indices);

        // Gravity Queries
        #pragma omp parallel for schedule(dynamic, 256)
        for (int i = 0; i < num_stars; ++i) {
            query_gravity(
                root, stars[i], i, stars, 
                node_masses, node_com_x, node_com_y, node_com_z, leaf_star_indices,
                G, epsilon_sq, accels[i].ax, accels[i].ay, accels[i].az
            );
        }

        // Velocity / Position Integration
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_stars; ++i) {
            stars[i].vx += accels[i].ax * dt;
            stars[i].vy += accels[i].ay * dt;
            stars[i].vz += accels[i].az * dt;

            stars[i].x += stars[i].vx * dt;
            stars[i].y += stars[i].vy * dt;
            stars[i].z += stars[i].vz * dt;
        }

        // Render
        update_vbo(stars, num_stars);
        render_frame(num_stars);
        
        glfwSwapBuffers(window);
        glfwPollEvents();

        end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> diff = end_time - start_time;
        if (frame_count % 60 == 0) {
            cout << "Frame " << frame_count << " processing time: " << diff.count() << "s (" << 1.0/diff.count() << " FPS)" << endl;
        }
        frame_count++;
    }
    cleanup_opengl();
#endif

    delete[] leaf_star_indices;
    delete[] node_masses;
    delete[] node_com_x;
    delete[] node_com_y;
    delete[] node_com_z;
    delete[] stars;

    return 0;
}