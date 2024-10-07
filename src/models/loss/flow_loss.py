import torch
from torch import Tensor


class FlowLoss(torch.nn.Module):
    """Calculate flow loss over the graph."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum"]:
            raise NotImplementedError(f"Not implemented reduction method {self.reduction}")
        self.reduction = reduction

    def forward(self, edge_index: Tensor, y_hat: Tensor):
        assert edge_index.shape[0] == 2, "Edge index is not oriented properly"
        num_nodes = edge_index.max().item() + 1
        in_zeros = torch.zeros(num_nodes, requires_grad=True, dtype=torch.float).to(edge_index.device)
        out_zeros = torch.zeros(num_nodes, requires_grad=True, dtype=torch.float).to(edge_index.device)
        incoming = torch.scatter_add(in_zeros, -1, edge_index[1, :], y_hat.squeeze(-1).to(torch.float))
        outgoing = torch.scatter_add(out_zeros, -1, edge_index[0, :], y_hat.squeeze(-1).to(torch.float))

        if self.reduction == "mean":
            loss = torch.abs(incoming - outgoing).mean()
        else:
            loss = torch.abs(incoming - outgoing).sum()
        return loss
