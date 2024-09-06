import pytest
import torch
from torch_geometric.data import Data
from transforms.remove_isolated_nodes import RemoveIsolatedNodes


@pytest.fixture
def sample_graph():
    # Sample graph with 5 nodes and directed edges
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
        ],
        dtype=torch.long,
    )

    edge_attr = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=5)

    return data


@pytest.fixture
def sample_graph_with_target():
    # Sample graph with 6 nodes and directed edges
    # This looks like triangle with three hanging lines coming out of vertices
    edge_index = torch.tensor(
        [
            [0, 1, 2, 0, 1, 3],
            [1, 2, 0, 4, 5, 2],
        ],
        dtype=torch.long,
    )

    edge_attr = torch.tensor(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [1, 2],
            [3, 4],
            [5, 6],
        ],
        dtype=torch.float,
    )
    # No node features and target of the same dimension as edge_index
    y = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float)
    data = Data(y=y, edge_index=edge_index, edge_attr=edge_attr, num_nodes=6)

    return data


def test_remove_isolated_nodes(sample_graph):
    transform = RemoveIsolatedNodes()
    transformed_graph = transform(sample_graph)

    # Expected result: no nodes with zero in-degree or out-degree
    expected_edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
        ],
        dtype=torch.long,
    )
    expected_edge_attr = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    expected_num_nodes = 4  # 4 nodes remain as no nodes had zero degree
    expected_node_features = torch.tensor([1, 2, 3, 4], dtype=torch.float)

    assert torch.equal(
        transformed_graph.edge_index, expected_edge_index
    ), "Edge indices do not match the expected result"
    assert (
        transformed_graph.num_nodes == expected_num_nodes
    ), f"Expected {expected_num_nodes} nodes but got {transformed_graph.num_nodes}"
    assert torch.equal(
        transformed_graph.edge_attr, expected_edge_attr
    ), "Edge attributes do not match the expected result"
    assert torch.equal(transformed_graph.x, expected_node_features), "Node features do not match the expected result"


def test_remove_all_isolated_nodes_and_update_target(sample_graph_with_target):
    transform = RemoveIsolatedNodes()
    transformed_graph = transform(sample_graph_with_target)
    expected_edge_index = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 0],
        ],
        dtype=torch.long,
    )
    expected_edge_attr = torch.tensor(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ],
        dtype=torch.float,
    )
    expected_target_features = torch.tensor([1, 1, 1], dtype=torch.float)

    assert torch.equal(
        transformed_graph.edge_index, expected_edge_index
    ), "Edge indices do not match the expected result"
    assert torch.equal(
        transformed_graph.edge_attr, expected_edge_attr
    ), "Edge attributes do not match the expected result"
    assert torch.equal(transformed_graph.y, expected_target_features), "Node features do not match the expected result"


def test_no_isolated_nodes():
    # Create a graph where no node is isolated (all have in-degree and out-degree > 0)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    x = torch.tensor([1, 2, 3], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, num_nodes=3)

    transform = RemoveIsolatedNodes()
    transformed_graph = transform(data)

    # No nodes should be removed
    assert transformed_graph.num_nodes == 3, "No nodes should be removed"
    assert torch.equal(transformed_graph.edge_index, edge_index), "Edge indices should not change"
    assert torch.equal(transformed_graph.x, x), "Node features should not change"
