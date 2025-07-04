import torch
import numpy as np

class NormalizeUnitSphere:
    """ Center to zero mean and scale so furthest point sits on unit sphere. """
    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        # pts: (N,3)
        centroid = pts.mean(dim=0, keepdim=True)          # (1,3)
        pts = pts - centroid                              # zero-mean
        furthest = pts.norm(dim=1).max()                  # scalar
        pts = pts / furthest                              # ∥pts∥ ≤ 1
        return pts

class RandomRotation:
    """ Rotate points by a random angle around each axis (uniform in [0,2π)). """
    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        # generate random rotation matrices for x,y,z
        angles = torch.rand(3) * 2 * np.pi  # (α,β,γ)
        Rx = torch.tensor([[1,0,0],
                           [0, torch.cos(angles[0]), -torch.sin(angles[0])],
                           [0, torch.sin(angles[0]),  torch.cos(angles[0])]])
        Ry = torch.tensor([[ torch.cos(angles[1]), 0, torch.sin(angles[1])],
                           [0,                   1,                0],
                           [-torch.sin(angles[1]),0, torch.cos(angles[1])]])
        Rz = torch.tensor([[ torch.cos(angles[2]), -torch.sin(angles[2]), 0],
                           [ torch.sin(angles[2]),  torch.cos(angles[2]), 0],
                           [0,                    0,                   1]])
        R = Rz @ Ry @ Rx               # combined rotation
        return (R.to(pts.device) @ pts.T).T

class RandomScaleJitter:
    """ Multiply points by a random scalar in [scale_low, scale_high]. """
    def __init__(self, scale_low=0.8, scale_high=1.2):
        self.low, self.high = scale_low, scale_high
    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        s = torch.empty(1).uniform_(self.low, self.high).item()
        return pts * s

class RandomJitter:
    """ Add Gaussian noise N(0,σ²) to each coordinate. """
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma, self.clip = sigma, clip
    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(pts) * self.sigma
        noise = noise.clamp(-self.clip, self.clip)
        return pts + noise

class RandomDropout:
    """ Randomly drop up to max_dropout_ratio of points (set them to the first point). """
    def __init__(self, max_dropout_ratio=0.1):
        self.ratio = max_dropout_ratio
    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        N = pts.shape[0]
        drop_idx = torch.randperm(N)[:int(N * self.ratio)]
        if len(drop_idx) > 0:
            # replace dropped points with the first point (so tensor size stays the same)
            pts[drop_idx] = pts[0]
        return pts

class Compose:
    """ Compose multiple transforms together. """
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, pts):
        for t in self.transforms:
            pts = t(pts)
        return pts
