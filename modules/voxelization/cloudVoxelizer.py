import torch

def voxelize(
    points: torch.Tensor,                # [N, ≥3] (x,y,z[, ...features])
    voxel_size: list[float],             # [vx, vy, vz]
    coors_range: list[float],            # [xmin, ymin, zmin, xmax, ymax, zmax]
    max_points_per_voxel: int = 32,
    max_voxels: int = 2000
):
    '''
    Args: 
        points: (B,N,features)  where features >=3
        voxel_size: size of each voxel
        coors_range: size of each voxel
        max_points_per_voxel: maximum points to cap off in a voxel
        max_voxels: 
    Returns:

    '''
    device = points.device
    xyz = points[:, :3]

    # 1) Compute discrete voxel coordinates
    vs = torch.tensor(voxel_size, device=device)
    mins = torch.tensor(coors_range[:3], device=device)
    grid_size = ((torch.tensor(coors_range[3:], device=device) - mins) / vs).long()
    coors = ((xyz - mins) / vs).floor().long()

    # 2) Mask out-of-bounds
    valid = ((coors >= 0) & (coors < grid_size)).all(dim=1)
    coors, pts = coors[valid], points[valid]

    # 3) Flatten 3D coors → 1D key
    keys = coors[:,0]*grid_size[1]*grid_size[2] + coors[:,1]*grid_size[2] + coors[:,2]
        # voxel_IndexX * num_Voxels_y * num_Voxels_Z + voxel_IndexY * num_VoxelsZ +  Voxel_IndexZ
    # 4) Unique voxels, inverse mapping, counts
    uniq_keys, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)

    # at this point inverse mapping is not capped, neither is unique_keys

    num_voxels = min(uniq_keys.size(0), max_voxels) #cap off the number of voxels considered....

    uniq_keys = uniq_keys[:num_voxels] # done so that if number of voxels exceed some fixed size then we can bound it....
    
    # at this point inverse mapping is not capped only unique keys are capped

    # 5) Keep only points in those first `num_voxels` voxels
    mask = inverse < num_voxels
    inverse, pts = inverse[mask], pts[mask]

    # 6) Prepare output buffers
    voxels = torch.zeros((num_voxels, max_points_per_voxel, points.size(1)),
                         device=device, dtype=points.dtype)
    num_per_voxel = torch.zeros(num_voxels, dtype=torch.int32, device=device)

    # 7) Simple Python loop to fill up to K points per voxel
    for idx, v in enumerate(inverse):
        cnt = num_per_voxel[v].item()
        if cnt < max_points_per_voxel:
            voxels[v, cnt] = pts[idx]
            num_per_voxel[v] += 1

    # 8) Recover 3D coords from flattened keys
    zs =  uniq_keys % grid_size[2]
    ys = (uniq_keys // grid_size[2]) % grid_size[1]
    xs =  uniq_keys // (grid_size[1]*grid_size[2])
    coords = torch.stack([xs, ys, zs], dim=1)

    return voxels, coords, num_per_voxel
