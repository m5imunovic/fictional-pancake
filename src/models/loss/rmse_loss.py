import torch
from torch import Tensor


class RMSELoss(torch.nn.Module):
    """Root Mean square error."""

    def __init__(self, reduce: str = "mean"):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduce=reduce)

    def forward(self, input: Tensor, target: Tensor):
        return torch.sqrt(self.mse(input, target))
