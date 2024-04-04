"""Classical multilayer perceptron (MLP) model."""

import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(
        self, node_features, hidden_features, out_features, num_layers, dropout
    ):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(node_features, hidden_features))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_features))

        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_features, hidden_features))
            self.bns.append(torch.nn.BatchNorm1d(hidden_features))

        self.lins.append(torch.nn.Linear(hidden_features, out_features))

        self.dropout = dropout

        # TODO: check if pl is calling this?
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)
        return x
