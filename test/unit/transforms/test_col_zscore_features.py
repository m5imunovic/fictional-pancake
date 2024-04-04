import torch
from torch_geometric.data import Data

from transforms.col_zscore_features import ColZscoreFeatures


def test_normalize_scale():
    assert ColZscoreFeatures().__repr__() == "ColZscoreFeatures()"

    x = torch.tensor([[3, 0, 1, 0], [3, 1, 2, 0], [3, 2, 0, 0]], dtype=torch.float)

    data = Data(x=x)
    data = ColZscoreFeatures()(data)
    assert len(data) == 1
    expected_data = torch.tensor(
        [[0, -1, 0, 0], [0, 0, 1, 0], [0, 1, -1, 0]], dtype=torch.float
    )
    assert torch.allclose(data.x, expected_data, atol=1e-4)
