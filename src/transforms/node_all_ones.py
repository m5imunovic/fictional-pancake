import torch
from torch_geometric.transforms import BaseTransform


class NodeAllOnes(BaseTransform):
    def __call__(self, data):
        # Assign each node a value of 1
        data.x = torch.ones((data.num_nodes, 1))
        return data
