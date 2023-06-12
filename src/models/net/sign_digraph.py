import torch.nn as nn

from models.net.mlp import MLP


class SignDiGraphNet(nn.Module):
    def __init__(self, mlp: MLP):
        super().__init__()
        self.mlp = mlp

    def forward(self, x, edge_index, edge_attr=None):
        score = self.mlp(x)
        return score
