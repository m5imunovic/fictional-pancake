import torch
from torch_geometric.data import Data

from transforms.col_zscore_len import ColZscoreLen


def test_normalize_length():
    x = torch.tensor([[3, 0, 1, 0], [1, 2, 3, 4], [3, 0, 0, 0]], dtype=torch.float)

    data = Data(edge_attr=x.T)
    data = ColZscoreLen()(data)
    expected_data = torch.tensor([[3, 0, 1, 0], [-1.1619, -0.3873, 0.3873, 1.1619], [3, 0, 0, 0]], dtype=torch.float)
    assert torch.allclose(data.edge_attr, expected_data.T, atol=1e-4)
