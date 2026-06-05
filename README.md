# 3D Galaxy Collision Simulator

A high-performance, custom-built 3D Galaxy Collision Simulator written in pure C++ (using the C++17 Standard Library only). This project simulates the gravitational interaction and collision of two stable galaxies (comprising stellar discs, bulges, and massive dark matter halos) using a highly optimized Barnes-Hut octree.

---

## Project Overview & Roadmap

This project is built incrementally following a rigorous engineering roadmap:

- **Phase 1-4: Foundation & Spatial DB Integration** - Standardized particle representations, memory pooling, and importing stellar catalogs from spatial data sources. This was completed in another project first stored as **[Spatial_DB Repository](https://github.com/lakshya076/Spatial_DB)**
- **Phase 5: Initial Conditions Generator** - Mathematical generation of stable galaxies (Miyamoto-Nagai stellar discs, Hernquist bulges, and dark matter halos) placed on a collision course.
- **Phase 6: Barnes-Hut Physics Core** - Bottom-up mass aggregation and fast gravitational force calculation using the $\theta = \frac{s}{d}$ approximation.
- **Phase 7: The Master Simulation Loop** - Main temporal integration loop connecting the octree structures with Newtonian physics over time steps ($\Delta t$).
- **Phase 8: Graphics Pipeline & Multithreading** - OpenGL buffer streaming and parallelizing the gravity queries using a thread pool.

---

## 🛠️ Architecture & Constraints

To achieve extreme real-time simulation performance, the engine adheres to strict hardware-friendly memory layouts and caching constraints:

### 1. The Memory Pool (`ArenaAllocator`)
Instead of standard dynamic allocations (`new`/`delete`) which cause memory fragmentation and latency, the simulator relies on a custom `ArenaAllocator`. It provisions contiguous blocks of RAM instantly. The arena is completely wiped (pointer reset to 0) and rebuilt every single frame.

### 2. High-Density Cache-Line Octree (`OctreeNode`)
`OctreeNode` structs are strictly packed and aligned to exactly 32 bytes (`alignas(32)`). This guarantees that reading a node loads exactly half of an L1 cache-line, preventing cache misses and doubling cache density. 
* The bounds are mathematically simplified using a `center` coordinate and `half_width` instead of standard min/max bounding boxes.
* To maintain this 32-byte limit, **no physics data (like mass or center of mass) is stored inside the node struct itself**. 

### 3. The Indirection Array
Instead of limiting leaf nodes to a small array of stars, the engine uses a global Indirection Array. Leaf nodes store a 32-bit `start_star_index`, allowing them to hold up to 32 stars without increasing the node struct size. This keeps the octree incredibly shallow, eliminating pointer-chasing latency.

### 4. Unified Star Struct (Array of Structures)
All stellar particles (visible disc stars, bulge stars, and dark matter macro-particles) are stored in a unified flat array:
```cpp
struct Star {
    float x, y, z;    // 3D position
    float vx, vy, vz; // 3D velocity
    float mass;       // Particle mass
    bool is_dm;       // Dark matter flag (is_dm = true skips rendering)
};
```

### 5. Node Physics (Structure of Arrays)
Because the `OctreeNode` cannot exceed 32 bytes, all node-level physics data (such as total mass and 3D Center of Mass) is stored in parallel arrays (`float* node_masses`, etc.) and accessed using the node's index.

### 6. Memory Locality & Prefetching
* **Morton Z-Order Curve Sorting:** The `stars` array is parallel-sorted using GCC's `__gnu_parallel::sort` according to a 64-bit interleaved Morton Code before tree construction. This ensures OpenMP threads process physically adjacent subsets of the tree, maximizing spatial locality.
* **Manual L1 Memory Prefetching:** The hottest direct-gravity math loop utilizes compiler intrinsics (`__builtin_prefetch`) to actively pull the *next* star's data into the L1 cache while the FPU computes the inverse square root math for the *current* star.

---

## 🧪 Mathematical Formulations (Initial Conditions)

The initial galaxies are generated using the following astronomical density profiles:
* **Miyamoto-Nagai Disc:** Used to spawn visible stars in a flattened stellar disk with circular orbital velocities.
* **Hernquist Bulge:** Generates a tightly packed galactic core with random velocity dispersions.
* **Hernquist Dark Matter Halo:** Spawns massive, diffuse dark matter macro-particles (`is_dm = true`) to act as the gravitational glue stabilizing the galaxies.
* **Collision Setup:** The generator constructs two identical stable galaxies, applies a spatial offset (e.g., $\pm 500$ units on the X-axis), adds opposing translational velocity vectors to set them on a collision course, and merges them into the flat `stars` array.

---

## 📂 Data & Stellar Database

The parser and database used to process and parse real star catalogs into binary formats (such as the parsed geometry and payload files used in earlier versions of this simulation) are hosted in a separate repository:
👉 **[Spatial_DB Repository](https://github.com/lakshya076/Spatial_DB)**

---

## 🚀 Getting Started

### Prerequisites
* A C++17 compliant compiler (e.g., `g++` or `clang++`).

### Compiling and Running (Direct CLI)
To compile the current build containing the generator, the octree builder, and the Barnes-Hut integration loop:
```powershell
g++ -std=c++17 -O3 -fopenmp -ffast-math -march=native .\main.cpp .\generator.cpp .\engine.cpp .\gravity_aggregator.cpp -o main.exe
.\main.exe
```
