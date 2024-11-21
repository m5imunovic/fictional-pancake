import torch

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class TargetWeights(BaseTransform):
    """Add weights vector to target values so that we can adjust the loss to be more sensitive to the values where
    error is more costly."""

    def __init__(self):
        super().__init__()

    def forward(self, data: Data) -> Data:
        y = data.y
        condition = y.eq(0) | y.eq(1)
        weights = torch.where(
            condition, torch.ones_like(y, dtype=torch.float32), torch.full_like(y, 0.5, dtype=torch.float32)
        )
        data.weights = weights
        return data
