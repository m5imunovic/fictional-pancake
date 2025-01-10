from typing import Sequence

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class CovPercentiles(BaseTransform):
    def __init__(self, percentiles: Sequence | None = None, feature: str = "graph_attr"):
        if percentiles is None:
            # adds 10 features
            self.percentiles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
        else:
            self.percentiles = torch.tensor(percentiles)

    def __call__(self, data: Data) -> Data:
        cov = data.edge_attr[:, 0]
        features = torch.quantile(cov.float(), self.percentiles).unsqueeze(0)
        data.graph_attr = features

        return data
