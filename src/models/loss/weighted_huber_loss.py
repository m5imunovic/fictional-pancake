import torch
import torch.nn as nn


class WeightedHuberLoss(nn.Module):
    """Wraps the vanilla HuberLoss so that the loss values are weighted more for specific subsets.

    The weights are added to data through transform in dataloader
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum"]:
            raise NotImplementedError(f"Not implemented reduction method {self.reduction}")
        self.reduction = reduction
        self.criterion = nn.HuberLoss(delta=delta, reduction=None)

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        loss = self.criterion(input, target)
        loss = loss * weights

        if self.reduction == "mean":
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss)
        loss = torch.mean(loss)
        return loss
