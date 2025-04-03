"""Adds indicator feature to denote if edge is multi-edge."""

import torch
from torch_geometric.transforms import BaseTransform


class MultiEdgeFeature(BaseTransform):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        _, inverse, counts = torch.unique(edge_index, dim=-1, return_inverse=True, return_counts=True)
        # convert boolean to float as every other feature is also float
        edge_mask = (counts > 1).float()
        edge_mask_full = edge_mask[inverse].unsqueeze(-1)

        # Append a new feature to edge_attr indicating whether the edge is a multi-edge
        if edge_attr is not None:
            data.edge_attr = torch.cat([edge_attr, edge_mask_full], dim=-1)
        else:
            data.edge_attr = edge_mask_full

        return data
