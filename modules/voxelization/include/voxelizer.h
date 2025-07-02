#pragma once
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> voxelize(
    const torch::Tensor& points,
    const std::vector<float>& voxel_size,
    const std::vector<float>& coors_range,
    int max_points_per_voxel,
    int max_voxels
);
// tuple returned -> voxels , coords , counts
