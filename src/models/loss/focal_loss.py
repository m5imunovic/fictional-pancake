import torch.nn as nn
from torch import Tensor
from torchvision.ops.focal_loss import sigmoid_focal_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return sigmoid_focal_loss(input, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
