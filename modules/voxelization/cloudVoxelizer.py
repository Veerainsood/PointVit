import torch
from torch.nn import Module
from torch import Tensor

class SimpleCloudVoxelizer(Module):

    def __init__(
            self,
            voxel_size: Tensor,             # [vx, vy, vz]
            coors_range: Tensor,            # [xmin, ymin, zmin, xmax, ymax, zmax]
            device: torch.device,
            max_points_per_voxel: int = 32,
            max_voxels: int = 100
    ):
        super().__init__
        self.voxel_size = voxel_size
        self.coors_range = coors_range
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        self.device = device
        self.grid_shape = ((self.coors_range[3:] - self.coors_range[:3]) / self.voxel_size).floor().long() 

    def hybrid_voxel_selection(self,voxel_coords, counts):
        """
        voxel_coords: (M,3) ints
        counts:       (M,)  point counts per voxel
        returns:      idxs, a length-max_voxels list of selected indices
        """
        M = counts.shape[0]
        P = int(0.6 * self.max_voxels)      # keep 60% densest
        Q = self.max_voxels - P             # 40% by coverage

        # 1) densest
        _, densest_idxs = torch.topk(counts, k=min(P, M), largest=True)

        # 2) farthest‐point sampling on the others
        mask = torch.ones(M, dtype=torch.bool, device=counts.device)
        mask[densest_idxs] = False
        remaining = torch.nonzero(mask).squeeze(1)

        centroids = (voxel_coords.float() * self.voxel_size + self.coors_range[:3])  # (M,3)

        def fps(points, K):
            N = points.shape[0]
            selected = [0]
            dist = torch.full((N,), float('inf'), device=points.device)
            for _ in range(1, min(K, N)):
                last = points[selected[-1]][None]      # (1,3)
                d = (points - last).pow(2).sum(dim=1)   # (N,)
                dist = torch.min(dist, d)
                selected.append(int(dist.argmax()))
            return torch.tensor(selected, device=points.device)

        rem_centroids = centroids[remaining]
        fps_idxs = fps(rem_centroids, Q)
        fps_global = remaining[fps_idxs]

        # 3) combine
        selected = torch.cat([densest_idxs, fps_global])
        return selected

    def voxelize(self, points: torch.Tensor):
        # 1) compute per‐point voxel indices
        voxel_idx = ((points - self.coors_range[:3]) / self.voxel_size).floor().long()
        min_bound = torch.zeros_like(voxel_idx)
        max_bound = (self.grid_shape - 1)[None, :].to(voxel_idx.device)
        voxel_idx = torch.max(torch.min(voxel_idx, max_bound), min_bound)
        keys = voxel_idx[:,0] * self.grid_shape[1] * self.grid_shape[2] + \
            voxel_idx[:,1] * self.grid_shape[2] + \
            voxel_idx[:,2]

        # 2) find unique voxels
        unique_keys, inverse_map, counts = torch.unique(
            keys, return_inverse=True, return_counts=True
        )
        M = unique_keys.numel()

        # 3) recover their 3D coords
        iz =  unique_keys %  self.grid_shape[2]
        iy = (unique_keys // self.grid_shape[2]) % self.grid_shape[1]
        ix =  unique_keys // (self.grid_shape[1] * self.grid_shape[2])
        voxel_coords = torch.stack([ix, iy, iz], dim=1)   # (M,3)

        # 4) pick the best K voxels
        selected = self.hybrid_voxel_selection(voxel_coords, counts)
        K = selected.numel()

        # 5) build a mapper from unique‐voxel‐idx → [0..K-1] or -1
        newIndex = torch.full((M,), -1, dtype=torch.long, device=points.device)
        newIndex[selected] = torch.arange(K, device=points.device)

        # 6) prepare outputs
        voxels = torch.zeros((K, self.max_points_per_voxel, points.shape[1]),
                            device=points.device)
        nums   = torch.zeros((K,), dtype=torch.long, device=points.device)
        coords = torch.zeros((K,3), device=points.device)

        # 7) scatter points into their selected voxel‐slots
        for pt_i in range(points.shape[0]):
            uv = inverse_map[pt_i].item()           # which unique‐voxel this pt falls into
            slot = newIndex[uv].item()              # which output‐voxel slot (or -1)
            if slot >= 0 and nums[slot] < self.max_points_per_voxel:
                voxels[slot, nums[slot]] = points[pt_i]
                nums[slot] += 1
                coords[slot] = voxel_coords[uv]

        return voxels, coords, nums
