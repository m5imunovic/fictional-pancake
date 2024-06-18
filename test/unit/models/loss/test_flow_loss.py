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
