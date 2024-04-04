
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class BinarizeTargets(BaseTransform):
    """Binarize targets given in `attrs` based on their boolean value.

    Args:
        attrs (List[str]): The names of attributes to normalize.
    """

    def __init__(self, attrs: list[str] = ["y"]):
        self.attrs = attrs

    def __call__(
        self,
        data: Data,
    ) -> Data:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                # TODO: use in place here?
                value = value.bool().long()
                store[key] = value
        return data
