#include "voxelizer.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/transform.h>

// forward-declare your CUDA kernels
void compute_keys_kernel( 
    const float* __restrict__ xyz,   // [N, 3] coordinates
    int32_t*      __restrict__ keys,  // [N] output keys
    int           N,                  // number of points
    float3        voxel_size,         // (vx, vy, vz)
    float3        grid_min,           // (xmin, ymin, zmin)
    int3          grid_size           // (Gx, Gy, Gz)
);
void fill_voxels_kernel( 
    const float* __restrict__ pts,      // [N×(3+F)] 
    const int32_t* __restrict__ p2v,    // [N] → voxel‐index or -1
    float*       __restrict__ voxels,   // [V, K, 3+F]
    int32_t*     __restrict__ counts,   // [V]
    int           N, int V, int K, int D  // D = 3(+features)
);


std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> voxelize(
    const torch::Tensor& points,
    const std::vector<float>& voxel_size,
    const std::vector<float>& coors_range,
    int max_pts_per_voxel,
    int max_voxels
) {

    int N = points.size(0);
    int D = points.size(1);
    auto opts = points.options();

    // grid dimensions
    int Gx = int((coors_range[3] - coors_range[0]) / voxel_size[0]);
    int Gy = int((coors_range[4] - coors_range[1]) / voxel_size[1]);
    int Gz = int((coors_range[5] - coors_range[2]) / voxel_size[2]);

    // allocate intermediate + output tensors on the same device as `points`
    auto keys      = torch::empty({N},        torch::dtype(torch::kInt32).device(opts.device()));
    auto unique_k  = torch::empty({N},        torch::dtype(torch::kInt32).device(opts.device()));
    auto p2v_idx   = torch::empty({N},        torch::dtype(torch::kInt32).device(opts.device()));
    auto voxels    = torch::zeros({max_voxels, max_pts_per_voxel, D}, opts);
    auto counts    = torch::zeros({max_voxels}, torch::dtype(torch::kInt32).device(opts.device()));


    compute_keys<<<blocks,threads>>>(
        points.data_ptr<float>(),
        keys.data_ptr<int32_t>(),
        N,
        make_float3(voxel_size[0],voxel_size[1],voxel_size[2]),
        make_float3(coors_range[0],coors_range[1],coors_range[2]),
        make_int3(Gx,Gy,Gz)
    );

    using DevIt = thrust::device_ptr<int32_t>;

    DevIt d_keys     (keys.data_ptr<int32_t>());
    DevIt d_unique   (unique_k.data_ptr<int32_t>());

    // 3a) copy only valid keys (>=0)
    auto end1 = thrust::copy_if(d_keys, d_keys+N, d_unique,
                                [] __device__(int32_t k){ return k >= 0; });
    int M = end1 - d_unique;              // number of in-bounds points

    // 3b) sort & unique
    thrust::sort(d_unique, d_unique+M);
    auto end2 = thrust::unique(d_unique, d_unique+M);
    int V_eff = end2 - d_unique;          // actual occupied voxels ≤ max_voxels


    thrust::transform(
    d_keys, d_keys+N, p2v_idx.data_ptr<int32_t>(),
    [unique = d_unique, V_eff] __device__ (int32_t k){
        if (k<0) return -1;
        // binary-search in unique[0..V_eff)
        auto it = thrust::lower_bound(unique, unique+V_eff, k);
        return (it < unique+V_eff) ? int(it - unique) : -1;
    }
    );


    fill_voxels<<<blocks,threads>>>(
    points.data_ptr<float>(),
    p2v_idx.data_ptr<int32_t>(),
    voxels.data_ptr<float>(),
    counts.data_ptr<int32_t>(),
    N, V_eff, max_pts_per_voxel, D
    );

    auto coords = torch::empty({V_eff, 3}, torch::dtype(torch::kInt32).device(opts.device()));

    int32_t* uk = unique_k.data_ptr<int32_t>();
    int32_t* cd = coords.data_ptr<int32_t>();
    for(int i=0;i<V_eff;++i){
    int key = uk[i];
    int iz  = key % Gz;
    int iy  = (key / Gz) % Gy;
    int ix  =  key / (Gy*Gz);
    cd[3*i+0] = ix;
    cd[3*i+1] = iy;
    cd[3*i+2] = iz;
    }

    return std::make_tuple(
    voxels.slice(0,0,V_eff),
    coords,
    counts.slice(0,0,V_eff)
    );

}
