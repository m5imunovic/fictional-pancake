import torch
from torch_geometric.data import Data
from torch_geometric.transforms import SIGN, BaseTransform


class SIGNMergedFeatures(BaseTransform):
    """Performs SIGN transform with K hops and merges all the features into one tensor.

    Args:
        attrs (List[str]): The names of attributes to normalize.
    """

    def __init__(self, K: int):
        self.K = K
        self.attrs = ["x"]
        self.attrs.extend(f"x{i}" for i in range(1, self.K + 1))

    def __call__(
        self,
        data: Data,
    ) -> Data:
        data = SIGN(self.K)(data)
        x_features = []
        for attr in self.attrs:
            if attr not in data:
                raise ValueError(f"Attribute {attr} not found in data")
            x_features.append(data.pop(attr))

        data.x = torch.cat(x_features, dim=-1)
        return data
