import torch
from torch_geometric.data import Data

from transforms.cov_percentiles import CovPercentiles


def test_coverage_percentiles():
    x = torch.tensor([[1, 2, 3, 4, 5], [3, 0, 0, 0, 0]], dtype=torch.float)

    # test default
    data = Data(edge_attr=x.T)
    data = CovPercentiles()(data)
    assert data.edge_attr.shape[1] == 2
    assert data.graph_attr.shape[1] == 10

    # test user spec
    data = Data(edge_attr=x.T)
    data = CovPercentiles(percentiles=[0.5])(data)
    expected_data = torch.tensor([[3.0]], dtype=torch.float)
    assert torch.allclose(data.graph_attr, expected_data, atol=1e-4)
