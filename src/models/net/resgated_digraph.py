import torch
import torch.nn as nn
from torch_geometric.nn.conv import ResGatedGraphConv


class ResGatedDiGraphNet(nn.Module):
    def __init__(
        self,
        num_layers: int,
        node_features: int,
        hidden_features: int,
        batch_norm: bool = False,
    ):
        super().__init__()

        self.W1 = nn.Linear(node_features, hidden_features, bias=True)
        self.W2 = nn.Linear(hidden_features, hidden_features, bias=True)

        self.gate_fw = LayeredResGatedGraphConv(
            num_layers=num_layers,
            hidden_features=hidden_features,
            flow="source_to_target",
        )
        self.gate_bw = LayeredResGatedGraphConv(
            num_layers=num_layers,
            hidden_features=hidden_features,
            flow="target_to_source",
        )

        self.scorer = nn.Linear(2 * hidden_features, out_features=1, bias=True)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.W1(x)
        h = torch.relu(h)
        h = self.W2(h)
        h_fw = self.gate_fw(h=h, edge_index=edge_index)
        h_bw = self.gate_bw(h=h, edge_index=edge_index)
        score = self.scorer(torch.cat([h_fw, h_bw], dim=1))

        return score


class LayeredResGatedGraphConv(nn.Module):
    def __init__(self, num_layers: int, hidden_features: int, flow: str):
        super().__init__()
        self.gnn = nn.ModuleList(
            [
                ResGatedGraphConv(
                    in_channels=hidden_features, out_channels=hidden_features, flow=flow
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, h, edge_index):
        for gnn_layer in self.gnn:
            h = gnn_layer(x=h, edge_index=edge_index)
        return h
