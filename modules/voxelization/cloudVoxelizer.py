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
            max_voxels: int = 2000
    ):
        super().__init__
        self.voxel_size = voxel_size
        self.coors_range = coors_range
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        self.device = device

    def voxelize(
            self,
            points: torch.Tensor # [N, ≥3] (x,y,z[, ...features])
        ):
        '''
        Args: 
            points: (N,features)  where features >=3
        Returns:
            voxels: (max_voxels, max_points_per_voxel, eachPoint's Dimention)
        '''
        device = points.device
        
        grid_shape = (self.coors_range[3:] - self.coors_range[:3]) // self.voxel_size

        # map point to voxels.
        voxel_indices = ( points - self.coors_range[:3] ) // self.voxel_size

        keys = voxel_indices[:,0] * grid_shape[1] * grid_shape[2] + voxel_indices[:,1] * grid_shape[2] + voxel_indices[:,2]
        # mapping of each point to some number which can retrieve the 3D indexes in the plane to find out the exact location of the voxel.

        # zaruri nahi hai ki har ek point har ek voxel me jaye to unique keys ko pick up karlo
        unqique_keys , inverse_mapping , count = torch.unique(keys,device=device,return_inverse=True,return_counts=True)

        # unique keys can now give us the exact 3D positions of voxels present in the 3D plane which have at least 1 point
        # voxel_index[i] = unique_keys[inverse_mapping[i],0] //grid_shape[1] * grid_shape[2] , unique_keys[inverse_mapping[i],1] // grid_shape[2], unique_keys[inverse_mapping[i],2] 
        # this is how we get the 3d coordinate... of a point i....

        # now choose only top k and drop all others.....
        _, indexOfTopK = torch.topk(input=count,k=self.max_voxels,largest=True,sorted=True)
        # JO BHI INDEX OF TOP K HAI VO VALE INDEXES COUNT KE CHOOSE HONGE,
        # ONCE CHOSEN   THEN we can get unique voxels that are chosen out of this... how?
        unqique_keys_v2 = unqique_keys[indexOfTopK]

        newIndexMapper = -torch.ones_like(count, dtype=torch.long, device=device)
        newIndexMapper[indexOfTopK] = torch.arange(indexOfTopK.numel(), device=device)

        voxels = torch.zeros((self.max_voxels,self.max_points_per_voxel,points.shape[1]),device=self.device)
        num_per_voxel = torch.zeros((self.max_voxels,),dtype=torch.long, device=self.device)
        voxel_coords = torch.zeros((self.max_voxels,3),device=self.device)

        topk_set = set(indexOfTopK.tolist())
        
        for i in range(points.shape[0]):
            
            unqique_keys_idx = int(inverse_mapping[i])
            
            if unqique_keys_idx in topk_set:
                
                zeroBasedIndex = newIndexMapper[unqique_keys_idx]
                
                key = unqique_keys_v2[unqique_keys_idx]

                iz =  key %  grid_shape[2]
                iy = (key // grid_shape[2]) % grid_shape[1]
                ix =  key // (grid_shape[1] * grid_shape[2])

                voxel_coords[int(zeroBasedIndex)] = torch.tensor(( ix, iy, iz ),device=self.device)
                
                if num_per_voxel[zeroBasedIndex]< self.max_points_per_voxel :
                    voxels[zeroBasedIndex, num_per_voxel[zeroBasedIndex]] = points[i]
                    num_per_voxel[zeroBasedIndex]+=1

        return voxels, voxel_coords, num_per_voxel
