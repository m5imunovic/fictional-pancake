from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class ColZscoreLen(BaseTransform):
    """Colum-normalizes the length feature.

    Args:
        attrs (List[str]): The names of attributes to normalize.
    """

    def __init__(self, col: int = 1):
        self.col = col

    def __call__(
        self,
        data: Data,
    ) -> Data:
        assert "edge_attr" in data, "Expected edge features"
        length = data.edge_attr[:, self.col]
        mean = length.float().mean()
        stddev = length.float().std()
        norm_len = (length - mean) / (stddev + 1e-8)
        data.edge_attr[:, self.col] = norm_len
        return data
