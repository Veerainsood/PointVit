// src/compute_keys.cu

#include <cuda_runtime.h>

extern "C"
__global__ void compute_keys(
    const float* __restrict__ xyz,   // [N, 3] coordinates
    int32_t*      __restrict__ keys,  // [N] output keys
    int           N,                  // number of points
    float3        voxel_size,         // (vx, vy, vz)
    float3        grid_min,           // (xmin, ymin, zmin)
    int3          grid_size           // (Gx, Gy, Gz)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // load point
    float x = xyz[3*idx + 0];
    float y = xyz[3*idx + 1];
    float z = xyz[3*idx + 2];

    // compute discrete indices
    int ix = floorf((x - grid_min.x) / voxel_size.x);
    int iy = floorf((y - grid_min.y) / voxel_size.y);
    int iz = floorf((z - grid_min.z) / voxel_size.z);

    // check bounds
    if (ix < 0 || ix >= grid_size.x ||
        iy < 0 || iy >= grid_size.y ||
        iz < 0 || iz >= grid_size.z) {
        keys[idx] = -1;
    } else {
        // flatten 3D index to 1D key
        keys[idx] = ix * (grid_size.y * grid_size.z)
                  + iy * grid_size.z
                  + iz;
    }
}
