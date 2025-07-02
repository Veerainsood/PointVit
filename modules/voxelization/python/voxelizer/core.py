import torch
from voxelizer import voxelize as _cuda_voxelize

def voxelize(points, voxel_size, coors_range, max_pts=32, max_vox=20000):
    """
    points: (N, ≥3) Tensor on cuda.
    returns voxels, coords, counts.
    """
    assert points.is_cuda, "points must be on GPU"
    return _cuda_voxelize(
        points, list(voxel_size), list(coors_range),
        max_pts, max_vox
    )
