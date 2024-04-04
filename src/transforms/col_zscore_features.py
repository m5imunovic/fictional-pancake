
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class ColZscoreFeatures(BaseTransform):
    """Colum-normalizes the attributes given in `attrs` z-scoring it along the column dimension.

    Args:
        attrs (List[str]): The names of attributes to normalize.
    """

    def __init__(self, attrs: list[str] = ["x"]):
        self.attrs = attrs

    def __call__(
        self,
        data: Data,
    ) -> Data:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                # TODO: use in place here?
                mean = value.float().mean(dim=-2, keepdim=True)
                stddev = value.float().std(dim=-2, keepdim=True)
                value = (value.float() - mean) / (stddev + 1e-8)
                store[key] = value
        return data
