import torch
from torch import Tensor


class FlowLoss(torch.nn.Module):
    """Calculate flow loss over the graph."""

    def forward(self, edge_index: Tensor, y_hat: Tensor, coverage: Tensor):
        coverage_pred = y_hat * coverage
        # from torch_scatter import scatter_add
        # ingoing = scatter_add(coverage_pred, edge_index[0, :], dim_size=edge_index.shape[1])
        # outgoing = scatter_add(coverage_pred, edge_index[1, :], dim_size=edge_index.shape[1])
        in_zeros = torch.zeros_like(coverage_pred, requires_grad=True)
        out_zeros = torch.zeros_like(coverage_pred, requires_grad=True)
        ingoing = in_zeros.scatter(-1, edge_index[0, :], coverage_pred, reduce="add")
        outgoing = out_zeros.scatter(-1, edge_index[1, :], coverage_pred, reduce="add")

        diff = (ingoing - outgoing).sum()
        return diff
