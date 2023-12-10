import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing


class ResGatedMultiDiGraphNet(nn.Module):
    def __init__(
        self, num_layers: int, node_features: int, edge_features: int, hidden_features: int, batch_norm: bool = False
    ):
        super().__init__()

        self.W11 = nn.Linear(node_features, hidden_features, bias=True)
        self.W12 = nn.Linear(hidden_features, hidden_features, bias=True)
        self.W21 = nn.Linear(edge_features, hidden_features, bias=True)
        self.W22 = nn.Linear(hidden_features, hidden_features, bias=True)

        self.gate = LayeredGatedGCN(num_layers=num_layers, hidden_features=hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)
        self.ln2 = nn.LayerNorm(hidden_features)

        self.scorer1 = nn.Linear(3 * hidden_features, hidden_features, bias=True)
        self.scorer2 = nn.Linear(hidden_features, out_features=1, bias=True)
        self.scorer = nn.Linear(3 * hidden_features, out_features=1, bias=True)

    def forward(self, x, edge_attr, edge_index) -> Tensor:
        h = self.W12(self.ln1(torch.relu(self.W11(x))))
        e = self.W22(self.ln2(torch.relu(self.W21(edge_attr))))

        h, e = self.gate.forward(h=h, edge_attr=e, edge_index=edge_index)

        src, dst = edge_index
        score = self.scorer1(torch.cat(([h[src], h[dst], e]), dim=1))
        score = torch.relu(score)
        score = self.scorer2(score)

        return score


class LayeredGatedGCN(nn.Module):
    def __init__(self, num_layers: int, hidden_features: int):
        super().__init__()
        self.gnn = nn.ModuleList([GatedGCN(hidden_features=hidden_features) for _ in range(num_layers)])

    def forward(self, h, edge_attr, edge_index):
        for gnn_layer in self.gnn:
            h, edge_attr = gnn_layer(edge_index=edge_index, h=h, edge_attr=edge_attr)
        return h, edge_attr


class GatedGCN(MessagePassing):
    def __init__(self, hidden_features: int):
        super().__init__(aggr="add")

        self.A1 = nn.Linear(hidden_features, hidden_features)
        self.A2 = nn.Linear(hidden_features, hidden_features)
        self.A3 = nn.Linear(hidden_features, hidden_features)

        self.B1 = nn.Linear(hidden_features, hidden_features)
        self.B2 = nn.Linear(hidden_features, hidden_features)
        self.B3 = nn.Linear(hidden_features, hidden_features)

        self.bn_h = nn.LayerNorm(hidden_features)
        self.bn_e = nn.LayerNorm(hidden_features)

    def forward(self, h, edge_attr, edge_index):
        A1h = self.A1(h)
        A2h = self.A2(h)
        A3h = self.A3(h)

        B1h = self.B1(edge_attr)
        B2h = self.B2(h)
        B3h = self.B3(h)

        row, col = edge_index
        bw_edge_index = torch.vstack((col, row))

        e_ji = B1h + B2h[row] + B3h[col]
        e_ik = B1h + B2h[col] + B3h[row]

        e_ji = self.bn_e(F.relu(e_ji))
        e_ik = self.bn_e(F.relu(e_ik))

        e_ji = edge_attr + e_ji
        e_ik = edge_attr + e_ik

        sigmoid_ji = torch.sigmoid(e_ji)
        sigmoid_ik = torch.sigmoid(e_ik)

        h_ji = self.propagate(x=A2h, sigma=sigmoid_ji, edge_index=edge_index)
        h_ik = self.propagate(x=A3h, sigma=sigmoid_ik, edge_index=bw_edge_index)

        h_new = A1h + h_ji + h_ik
        h_new = self.bn_h(F.relu(h_new))
        h = h + h_new

        return h, e_ji

    def message(self, x_j, sigma) -> Tensor:
        return (x_j * sigma) / (torch.sum(sigma, dim=1).unsqueeze(dim=1) + 1e-6)
