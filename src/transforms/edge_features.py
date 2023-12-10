from typing import List

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class NormalizeEdgeFeatures(BaseTransform):
    """Adds the node degree the node features (i.e. x data)

    Args:
        in_degree (bool, optional):
            If set to True, will compute the in-degree of nodes instead of the out-degree.
    """

    def __init__(self, attrs: List[str] = ["edge_attr"]):
        self.attrs = attrs

    def __call__(self, data: Data):
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                if value.numel() > 0:
                    max_val, _ = value.max(dim=0)
                    value.div_(max_val)
                store[key] = value

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.name}]"
