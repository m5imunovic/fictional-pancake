import torch
from torch_geometric.data import Data

from transforms.degree_features import DegreeFeatures


def test_degree_features():
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 7, 8, 8]],
        dtype=torch.long,
    )

    data = Data(edge_index=edge_index, **{"num_nodes": 9})

    expected_out_degree = torch.tensor([2, 2, 2, 1, 1, 1, 1, 0, 0]).unsqueeze(-1)
    data = DegreeFeatures()(data)
    assert torch.allclose(data.x, expected_out_degree)

    expected_in_degree = torch.tensor([0, 1, 1, 1, 1, 1, 1, 2, 2]).unsqueeze(-1)
    data = DegreeFeatures(in_degree=True)(data)
    assert torch.allclose(
        data.x, torch.cat([expected_out_degree, expected_in_degree], dim=-1)
    )
