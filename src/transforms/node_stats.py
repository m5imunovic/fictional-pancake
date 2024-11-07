import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class NodeStats(BaseTransform):
    def __init__(self, percentiles=None):
        if percentiles is None:
            self.percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.99]
        else:
            self.percentiles = percentiles

    def __call__(self, data: Data) -> Data:
        cov = data.edge_attr[:, 0]

        percentiles = torch.quantile(cov, torch.tensor(self.percentiles))
        mean_val = cov.mean().unsqueeze(0)

        # Concatenate percentiles and mean to create node features
        node_features = torch.cat([percentiles, mean_val]).repeat(data.num_nodes, 1)

        data.x = node_features
        return data
