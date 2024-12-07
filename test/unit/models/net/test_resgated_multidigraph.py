import pytest
import torch

from torch_geometric.data import Data

from models.net.resgated_multidigraph import GatedGCN, LayeredGatedGCN


@pytest.fixture
def simple_graph(resgated_mdg_transform):
    """Create simple graph with two nodes connected bidirectionally with two edges."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    # len, kc
    edge_attr = torch.tensor([[0.25, 40], [0.75, 60]], dtype=torch.float)

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    data = resgated_mdg_transform(data)
    return data


def test_gate(simple_graph):
    torch.manual_seed(10)
    gcn = GatedGCN(hidden_features=2, gate_norm="feature", layer_norm=False)
    gcn.reset_parameters()
    h, e_fw = gcn(simple_graph.x, simple_graph.edge_attr, simple_graph.edge_index)

    expected_h = torch.tensor([[0.5, 0.3183], [0.5065, 0.9153], [1.0, 1.0]])
    assert torch.all(torch.isclose(h.detach(), expected_h, rtol=0.001))


def test_layered_gate_init(simple_graph):
    # for now just test the data flow works
    layered_gcn = LayeredGatedGCN(2, hidden_features=2, gate="bidirect")
    _ = layered_gcn(simple_graph.x, simple_graph.edge_attr, simple_graph.edge_index)

    layered_gcn = LayeredGatedGCN(2, hidden_features=2, gate="unidirect")
    _ = layered_gcn(simple_graph.x, simple_graph.edge_attr, simple_graph.edge_index)
