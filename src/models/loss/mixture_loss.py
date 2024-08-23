import torch
from torch import Tensor

from models.loss.flow_loss import FlowLoss

from models.loss.rmse_loss import RMSELoss


class MixtureLoss(torch.nn.Module):
    def __init__(self, alfa: float = 1.0, beta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.alfa = alfa
        self.beta = beta
        self.flow_loss = FlowLoss(reduction=reduction)
        self.rmse_loss = RMSELoss(reduction="mean")

    def forward(self, edge_index: Tensor, y_hat: Tensor, y: Tensor):
        fl = self.flow_loss(edge_index, y_hat)
        rl = self.rmse_loss(y_hat.to(torch.float), y.to(torch.float))

        return self.beta * rl + self.alfa * fl
