from .voxelVisualizer import Visualizer
from ..dataset.utils.datasetHandler import PointCloudDataset
from torch.utils.data import DataLoader
import os, glob

root = "/home/kripaludas/Documents/Tamu Research/PointVit/data/ModelNet40"
classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
class_to_idx = {c:i for i,c in enumerate(classes)}

def make_split(split: str):
    files, labels = [], []
    for c in classes:
        pattern = os.path.join(root, c, split, "*.off")
        i = 0
        for fp in sorted(glob.glob(pattern)):
            files.append(fp)
            labels.append(class_to_idx[c])
            i+=1
            if i >=3:
                break
    return files, labels

train_files, train_labels = make_split("train")
val_files,   val_labels   = make_split("test")


from ..dataset.utils.datasetHandler import (
    NormalizeUnitSphere, RandomRotation,
    RandomScaleJitter, RandomJitter, RandomDropout, Compose
)
from .voxelVisualizer import Visualizer
import torch

# re-create your train‐time transform
train_transform = Compose([
    NormalizeUnitSphere(),
    RandomRotation(),
    RandomScaleJitter(0.8,1.2),
    RandomJitter(0.01,0.05),
    RandomDropout(0.1),
])

vis = Visualizer()

for file in train_files:
    # 1. load raw vertices
    print(file)
    pts = vis.load_off(file)                        # numpy (N,3)
    pts = torch.from_numpy(pts).float()         # tensor (N,3)

    # 2. apply your augmentations
    pts_t = train_transform(pts)                # tensor (N,3)

    # 3. visualize the *transformed* cloud
    vis.visualize(pts_t.cpu().numpy())