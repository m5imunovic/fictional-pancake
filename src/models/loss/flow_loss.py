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
        # from torch_scatter import scatter_add
        # ingoing = scatter_add(coverage_pred, edge_index[0, :], dim_size=edge_index.shape[1])
        # outgoing = scatter_add(coverage_pred, edge_index[1, :], dim_size=edge_index.shape[1])
        in_zeros = torch.zeros_like(y_hat.squeeze(-1), requires_grad=True, dtype=torch.float)
        out_zeros = torch.zeros_like(y_hat.squeeze(-1), requires_grad=True, dtype=torch.float)
        ingoing = torch.scatter_add(in_zeros, -1, edge_index[0, :], y_hat.squeeze(-1).to(torch.float))
        outgoing = torch.scatter_add(out_zeros, -1, edge_index[1, :], y_hat.squeeze(-1).to(torch.float))

        if self.reduction == "mean":
            loss = torch.abs(ingoing - outgoing).mean()
        else:
            loss = torch.abs(ingoing - outgoing).sum()
        return loss
