import pytest
import torch
from torch_geometric.data import Data
from transforms.tip_feature import TipFeature


@pytest.fixture
def sample_data():
    # Create a simple graph as a fixture
    return Data(
        x=torch.tensor(
            [
                [0, 1],  # Node 0: indegree=0, outdegree=1
                [1, 1],  # Node 1: indegree=1, outdegree=1
                [2, 2],  # Node 2: indegree=2, outdegree=2
                [1, 0],  # Node 3: indegree=1, outdegree=0
            ],
            dtype=torch.float,
        ),
        edge_index=torch.tensor([[0, 1, 2, 2], [1, 2, 2, 3]], dtype=torch.long),
    )


def test_tip_edge_feature(sample_data):
    transform = TipFeature()
    transformed_data = transform(sample_data)

    # Expected edge_is_tip
    # Edge 0->1: Tip
    # Edge 1->2: Normal edge
    # Edge 2->2: Loop is not tip
    # Edge 2->3: Tip
    expected_edge_is_tip = torch.tensor([1, 0, 0, 1], dtype=torch.float).unsqueeze(-1)

    assert torch.equal(transformed_data.edge_attr, expected_edge_is_tip), "Edge features do not match expected values."


def test_existing_edge_attr(sample_data):
    sample_data.edge_attr = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float)

    transform = TipFeature()
    transformed_data = transform(sample_data)

    # Expected concatenated edge_attr
    expected_edge_attr = torch.tensor([[1.0, 1.0], [2.0, 0.0], [3.0, 0.0], [4.0, 1.0]], dtype=torch.float)

    assert torch.equal(
        transformed_data.edge_attr, expected_edge_attr
    ), "Edge attributes with tips not correctly appended."
