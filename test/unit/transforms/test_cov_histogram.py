import torch
from torch_geometric.data import Data

from transforms.cov_histogram import CovHistogram


def test_cov_histogram():
    torch.manual_seed(2)
    edge_attr = (
        torch.cat(
            [
                torch.randn(9000) * 5 + 15,  # Cluster around 15
                torch.rand(1000) * 1000,  # Uniformly distributed noise
            ]
        )
        .clamp(0, 1000)
        .unsqueeze(1)
    )
    data = Data(edge_attr=edge_attr)

    transform = CovHistogram(extend=3)
    transformed_data = transform(data)

    assert hasattr(transformed_data, "graph_attr"), "selected_bins attribute missing"
    assert transformed_data.graph_attr.numel() == 6, "selected_bins should not be empty"
    assert torch.equal(
        transformed_data.graph_attr, torch.tensor([[12.0, 13.0, 14.0, 15.0, 16.0, 17.0]])
    ), "selected_bins should cluster around 15"
