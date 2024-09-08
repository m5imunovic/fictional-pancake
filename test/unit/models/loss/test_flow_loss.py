import pytest
import torch
from torch_geometric.data import Data

from models.loss.flow_loss import FlowLoss


def sample_graph() -> Data:
    x = torch.arange(4)
    edge_index = torch.tensor([[0, 1, 1, 2], [3, 0, 3, 1]])
    coverage = torch.tensor([1, 1, 75, 76], dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=coverage)


def test_flow():
    data = sample_graph()

    flow_loss = FlowLoss()
    torch.ones_like(data.edge_attr)
    result = flow_loss(data.edge_index, data.edge_attr)
    assert result == 38
    # TODO: add more test cases


@pytest.fixture
def example_input():
    # Create a graph with 10 nodes and 20 directed edges.
    # Each column in edge_index represents an edge from source node to target node.
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
        ],
        dtype=torch.long,
    )

    # Predicted multiplicity values (y_hat) for each edge
    y_hat = torch.rand(20, 1, dtype=torch.float)

    return edge_index, y_hat


def test_sum_reduction_large_graph(example_input):
    """Test FlowLoss with sum reduction on a graph with at least 10 nodes and 20 edges."""

    edge_index, y_hat = example_input
    # Initialize the FlowLoss with sum reduction
    loss_fn = FlowLoss(reduction="sum")

    # Forward pass
    loss = loss_fn(edge_index, y_hat)

    # Manually compute the expected loss
    in_zeros = torch.zeros(10, dtype=torch.float)  # 10 nodes for incoming flow
    out_zeros = torch.zeros(10, dtype=torch.float)  # 10 nodes for outgoing flow

    # Compute ingoing and outgoing flow based on edges and predicted multiplicity
    ingoing = torch.scatter_add(in_zeros, 0, edge_index[1, :], y_hat.squeeze(-1))
    outgoing = torch.scatter_add(out_zeros, 0, edge_index[0, :], y_hat.squeeze(-1))

    expected_loss = torch.abs(ingoing - outgoing).sum()

    # Check if the calculated loss matches the expected loss
    assert torch.isclose(loss, expected_loss), f"Expected sum loss {expected_loss}, but got {loss}"
