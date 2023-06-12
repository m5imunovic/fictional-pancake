import torch
from torch_geometric.data import Data

from transforms.col_normalize_features import ColNormalizeFeatures


def test_normalize_scale():
    assert ColNormalizeFeatures().__repr__() == "ColNormalizeFeatures()"

    x = torch.tensor([[3, 0, 1, 0], [6, 1, 2, 0], [3, 0, 0, 0]], dtype=torch.float)

    data = Data(x=x)
    data = ColNormalizeFeatures()(data)
    assert len(data) == 1
    expected_data = torch.tensor([[0.25, 0, 0.3333, 0], [0.5, 1, 0.6667, 0], [0.25, 0, 0, 0]], dtype=torch.float)
    assert torch.allclose(data.x, expected_data, atol=1e-4)
