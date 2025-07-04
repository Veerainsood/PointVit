import numpy as np
import torch
from torch.nn import Module
import re
from ..voxelization.cloudVoxelizer import SimpleCloudVoxelizer

class Visualizer(Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.coors_range = torch.tensor([-1.2, -1.2, -1.2, 1.2, 1.2, 1.2], device=self.device)
        self.voxel_size  = torch.tensor([0.05, 0.05, 0.05], device=self.device)
        

    def load_off(self,filename: str) -> np.ndarray:
        """
        Reads an OFF file and returns an (N,3) array of vertex coordinates.
        Correctly handles both:
        OFF
        n_verts n_faces n_edges
        and
        OFFn_verts n_faces n_edges
        """
        with open(filename, 'r') as f:
            first = f.readline().strip()
            # Case A: correct header
            if first == "OFF":
                counts_line = f.readline().strip()
            # Case B: header+counts fused together, e.g. "OFF700 664 0"
            elif first.startswith("OFF"):
                counts_line = first[3:].strip()
            else:
                raise ValueError(f"{filename}: invalid OFF header")

            # extract three integers
            parts = counts_line.split()
            if len(parts) < 3:
                # fallback to regex if someone used commas or no spaces
                nums = re.findall(r"-?\d+", counts_line)
                if len(nums) < 3:
                    raise ValueError(f"{filename}: can't parse counts line '{counts_line}'")
                parts = nums[:3]

            n_verts, n_faces, _ = map(int, parts[:3])

            verts = []
            for _ in range(n_verts):
                line = f.readline().strip()
                if not line:
                    raise ValueError(f"{filename}: unexpected EOF reading vertices")
                x, y, z, *rest = map(float, line.split())
                verts.append((x, y, z))

            # you can ignore faces if you only need vertices
            # for _ in range(n_faces):
            #     f.readline()

        return np.array(verts, dtype=np.float32)
    
    def visualize(self,verts):
        # try:
        #     verts = self.load_off(offFilePath)
        #     print(torch.tensor(verts.min(axis=1)).min(axis=0))
        # except Exception as e:
        #     print('Unable to load .off data from file provided')
        #     print(e)
        #     return
        import open3d as o3d
        voxelizer = SimpleCloudVoxelizer(self.voxel_size, self.coors_range, self.device)

        pts_tensor = torch.from_numpy(verts).to(self.device)   # (N,3)
        
        try:
            voxels, coords, num_pts = voxelizer.voxelize(pts_tensor)

            # 1) Original point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(verts)
            pcd.paint_uniform_color([0.7, 0.7, 0.7])

            # 2) Voxel boxes
            boxes = []
            coords_np = coords.cpu().numpy()
            vrange = self.coors_range.cpu().numpy()[:3]
            vsz = self.voxel_size.cpu().numpy()

            for idx in range(coords_np.shape[0]):
                i, j, k = coords_np[idx]
                min_bound = vrange + np.array([i, j, k]) * vsz
                max_bound = min_bound + vsz
                aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                aabb.color = (1.0, 0.0, 0.0)   # red boxes
                boxes.append(aabb)

            # 3) Draw them all
            o3d.visualization.draw_geometries(
                [pcd, *boxes],
                window_name="Point Cloud + Voxels",
                point_show_normal=False
            )

        except Exception as e:
            print(e)
        return


        


    