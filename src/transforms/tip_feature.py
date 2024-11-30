"""This transform assumes that the DegreeTransform has been called and that the first two features of node are in and
out degrees.

Order is not important
"""

from typing import Optional

import torch
from torch_geometric.transforms import BaseTransform


class TipFeature(BaseTransform):
    def __init__(self, degree_cols: Optional[tuple] = (0, 1)):
        self.incol = degree_cols[0]
        self.outcol = degree_cols[1]

    def __call__(self, data):
        # Get in-degrees and out-degrees from node features
        assert "x" in data, "Expected node features to be present"
        indegree = data.x[:, self.incol]
        outdegree = data.x[:, self.outcol]

        tip_nodes = torch.logical_xor(indegree == 0, outdegree == 0)

        # Create tip edge feature
        source, target = data.edge_index
        edge_is_tip = torch.logical_xor(tip_nodes[source], tip_nodes[target])

        if "edge_attr" in data and data.edge_attr is not None:
            data.edge_attr = torch.cat([data.edge_attr, edge_is_tip.unsqueeze(-1).float()], dim=-1)
        else:
            data.edge_attr = edge_is_tip.unsqueeze(-1).float()

        return data
