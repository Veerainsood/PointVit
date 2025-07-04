from torch.utils.data import Dataset, DataLoader
from .preprocessors.processingUtils import Compose,NormalizeUnitSphere,RandomRotation\
, RandomScaleJitter, RandomJitter,RandomDropout
import torch
import numpy as np
class PointCloudDataset(Dataset):

    def __init__(self, file_list, labels, train=True):
        self.files  = file_list      # list of paths → raw (N,3) tensors
        self.labels = labels
        if train:
            self.transform = Compose([
                NormalizeUnitSphere(),
                RandomRotation(),
                RandomScaleJitter(0.8,1.2),
                RandomJitter(0.01,0.05),
                RandomDropout(0.1),
            ])
        else:
            # at test time: only normalize
            self.transform = NormalizeUnitSphere()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pts = torch.from_numpy(np.load(self.files[idx])).float()  # or however you load
        pts = self.transform(pts)                                 # (N,3)
        label = self.labels[idx]
        return pts, label

# usage
# train_ds = PointCloudDataset(train_files, train_labels, train=True)
# val_ds   = PointCloudDataset(val_files,   val_labels,   train=False)

# train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
# val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)
    