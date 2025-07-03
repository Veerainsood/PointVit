import torch
from torch.nn import Module


class SimplePCAEmbedder(Module):

    def __init__(self, min_pts=3):
        
        '''
        
            min_pts: minimum number of points needed to be present in the voxel to be eligble to extract out features..
    
        '''

        super().__init__()
        
        self.min_pts = min_pts 


    def voxel_pca_features(self,voxels, number_per_voxel):

        '''
            Args:
            
                voxels: (num_voxels , maxPtsPerVoxel, PointsShape) represents whole voxelized grid
                
                number_per_voxel: represents a tensor containing the number of points per voxel
            
            Returns:

                features: (num_voxels, featureShape)

        '''
        # first filter out the eligible voxels...

        mask = number_per_voxel >= self.min_pts

        voxels = voxels[mask]

        # now for each point calculate largest eigen values...
        

        pass
