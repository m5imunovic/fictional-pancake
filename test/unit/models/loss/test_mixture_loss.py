import pytest
import torch
from torch_geometric.data import Data

from models.loss.mixture_loss import MixtureLoss


def sample_graph() -> Data:
    x = torch.arange(4)
    edge_index = torch.tensor([[0, 1, 1, 2], [3, 0, 3, 1]])
    coverage = torch.tensor([1, 1, 75, 76], dtype=torch.float32)
    multiplicity = torch.tensor([0, 0, 75, 75], dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=coverage, y=multiplicity)


@pytest.fixture(scope="function")
def sample_subgraph(test_data_path) -> Data:
    subgraph_path = test_data_path / "subgraph_dataset" / "0_798429_subgraph.pt"
    return torch.load(subgraph_path)


def test_mixture_loss():
    data = sample_graph()

    loss = MixtureLoss()
    pred_all_true = torch.tensor([0, 0, 75, 75], dtype=torch.float32)
    result = loss(data.edge_index, pred_all_true, data.y)
    assert torch.isclose(result, torch.tensor(37.5))
    # # 2 -> 1 false
    # result = flow_loss(data.edge_index, torch.tensor([1, 1, 1, 0]), data.edge_attr)
    # assert result == 0
    # # 0 -> 3 false
    # result = flow_loss(data.edge_index, torch.tensor([0, 1, 1, 1]), data.edge_attr)
    # assert result == 0
    # # TODO: add more test cases


def test_mixture_loss_sub(sample_subgraph):
    pred = torch.ones_like(sample_subgraph.y)
    loss = MixtureLoss()
    result = loss(sample_subgraph.edge_index, pred, sample_subgraph.y)
    assert not torch.isclose(result, torch.tensor(0.0))


def test_mixture_uses_huber_loss():
    loss = MixtureLoss(base_loss="huber")
    assert isinstance(loss.base_loss, torch.nn.HuberLoss)
