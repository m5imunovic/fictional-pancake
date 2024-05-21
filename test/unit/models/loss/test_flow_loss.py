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
    all_true = torch.ones_like(data.edge_attr)
    result = flow_loss(data.edge_index, all_true, data.edge_attr)
    assert result == 0
    # 2 -> 1 false
    result = flow_loss(data.edge_index, torch.tensor([1, 1, 1, 0]), data.edge_attr)
    assert result == 0
    # 0 -> 3 false
    result = flow_loss(data.edge_index, torch.tensor([0, 1, 1, 1]), data.edge_attr)
    assert result == 0
    # TODO: add more test cases
