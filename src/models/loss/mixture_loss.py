import torch
from torch import Tensor

from models.loss.flow_loss import FlowLoss

from models.loss.rmse_loss import RMSELoss


class MixtureLoss(torch.nn.Module):
    def __init__(self, alfa: float = 1.0, beta: float = 1.0, gama: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.alfa = alfa
        self.beta = beta
        self.gama = gama
        self.flow_loss = FlowLoss(reduction=reduction)
        self.rmse_loss = RMSELoss(reduction="mean")
        self.int_reduction = torch.mean if reduction == "mean" else torch.sum

    def forward(self, edge_index: Tensor, y_hat: Tensor, y: Tensor):
        # integer penalty for the multiplicity predictions that deviate from integer value
        ip = self.int_reduction(torch.sin(torch.pi * y_hat) ** 2)
        fl = self.flow_loss(edge_index, y_hat)
        rl = self.rmse_loss(y_hat.to(torch.float), y.to(torch.float))

        return self.gama * ip + self.beta * rl + self.alfa * fl
