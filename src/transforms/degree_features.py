from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree


class DegreeFeatures(BaseTransform):
    """Adds the node degree the node features (i.e. x data)

    Args:
        in_degree (bool, optional):
            If set to True, will compute the in-degree of nodes instead of the out-degree.
    """

    def __init__(self, in_degree: Optional[bool] = False):
        self.in_degree = in_degree
        self.name = "in_degree" if in_degree else "out_degree"

    def __call__(self, data: Data):
        idx = data.edge_index[1 if self.in_degree else 0]
        deg = degree(idx, data.num_nodes, dtype=torch.long).unsqueeze(-1)

        if "x" in data:
            x = data.x
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.view(-1, 1)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.name}]"
