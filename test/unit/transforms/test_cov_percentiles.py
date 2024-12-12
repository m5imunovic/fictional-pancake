import torch
from torch_geometric.data import Data

from transforms.cov_percentiles import CovPercentiles


def test_normalize_length():
    x = torch.tensor([[1, 2, 3, 4, 5], [3, 0, 0, 0, 0]], dtype=torch.float)

    # test default
    data = Data(edge_attr=x.T)
    data = CovPercentiles()(data)
    assert data.edge_attr.shape[1] == 12

    # test user spec
    data = Data(edge_attr=x.T)
    data = CovPercentiles(percentiles=[0.5])(data)
    expected_data = torch.tensor([[1, 2, 3, 4, 5], [3, 0, 0, 0, 0], [3, 3, 3, 3, 3]], dtype=torch.float)
    assert torch.allclose(data.edge_attr, expected_data.T, atol=1e-4)
