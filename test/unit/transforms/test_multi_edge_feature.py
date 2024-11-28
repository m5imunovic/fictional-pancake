import torch
from torch_geometric.data import Data

from transforms.multi_edge_feature import MultiEdgeFeature


def test_multi_edge_feature():
    # Example graph: 4 edges, with two (0 -> 1) edges forming a multi-edge
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 1, 2, 3]])
    edge_attr = torch.tensor([[0.5], [0.6], [0.7], [0.8]])

    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    transform = MultiEdgeFeature()
    data = transform(data)

    assert data.edge_attr.shape == (4, 2), "Edge attributes should have 2 columns after transform"

    expected_flags = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    assert torch.allclose(data.edge_attr[:, 1:], expected_flags)
