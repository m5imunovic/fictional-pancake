import pytest
import torch

from torch_geometric.data import Data

from models.net.resgated_multidigraph import GatedGCN, LayeredGatedGCN, ResGatedMultiDiGraphNet
from transforms.cov_percentiles import CovPercentiles


@pytest.fixture
def simple_graph(resgated_mdg_transform):
    """Create simple graph with two nodes connected bidirectionally with two edges."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    # len, kc
    edge_attr = torch.tensor([[0.25, 40], [0.75, 60]], dtype=torch.float)

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    # adds 2 x features
    data = resgated_mdg_transform(data)
    return data


def test_gate(simple_graph):
    torch.manual_seed(10)
    gcn = GatedGCN(hidden_features=2, gate_norm="feature", layer_norm=False)
    gcn.reset_parameters()
    h, e_fw = gcn(simple_graph.x, simple_graph.edge_attr, simple_graph.edge_index)

    expected_h = torch.tensor([[0.5779, 1.2018], [0.5000, 1.8409], [2.1267, 2.7953]])
    assert torch.all(torch.isclose(h.detach(), expected_h, rtol=0.001))


def test_layered_gate_init(simple_graph):
    # for now just test the data flow works
    layered_gcn = LayeredGatedGCN(2, hidden_features=2, gate="bidirect")
    _ = layered_gcn(simple_graph.x, simple_graph.edge_attr, simple_graph.edge_index)

    layered_gcn = LayeredGatedGCN(2, hidden_features=2, gate="unidirect")
    _ = layered_gcn(simple_graph.x, simple_graph.edge_attr, simple_graph.edge_index)


def test_with_graph_attr(simple_graph):
    # for now just test the data flow works
    g = CovPercentiles(percentiles=[0.25, 0.75])(simple_graph)
    net = ResGatedMultiDiGraphNet(num_layers=1, node_features=2, edge_features=2, hidden_features=10, graph_features=2)
    ei_ptr = torch.tensor([simple_graph.edge_index.shape[1]])
    _ = net.forward(g.x, g.edge_attr, g.edge_index, g.graph_attr, ei_ptr=ei_ptr)
