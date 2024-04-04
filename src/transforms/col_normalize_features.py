
from torch.nn.functional import normalize
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class ColNormalizeFeatures(BaseTransform):
    """Colum-normalizes the attributes given in `attrs` dividing it with the p-1 norm along the column dimension.

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
                value = normalize(value, dim=-2, p=1)
                store[key] = value
        return data
