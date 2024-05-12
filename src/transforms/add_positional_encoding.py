"""Adaptation of torch_geometric transform AddRandomWalkPE with added self loops, as DBG graph lacks those connections.

We do not want to add self loops in the graph edge index permanently
"""

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    add_self_loops,
    get_self_loop_attr,
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
    to_torch_csr_tensor,
)


def add_node_attr(data: Data, value: Tensor, attr_name: str) -> Data:
    if attr_name is None:
        assert data.x is None
        data.x = value
    else:
        data[attr_name] = value

    return data


class AddRandomWalkPE(BaseTransform):
    """Adds random walk positional encoding.

    Args:
        walk_length (int): The number of random walk steps
    """

    def __init__(self, walk_length: int) -> None:
        self.walk_length = walk_length

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        assert data.num_nodes is not None
        N = data.num_nodes

        edge_index, _ = add_self_loops(data.edge_index, num_nodes=N)
        row, col = edge_index
        num_edges = max(edge_index.shape)
        value = torch.ones(num_edges, device=row.device)
        value = scatter(value, row, dim_size=N, reduce="sum").clamp(min=1)[row]
        value = 1.0 / value

        if N <= 2_00:  # Dense code path for faster computation
            adj = torch.zeros((N, N), device=row.device)
            adj[row, col] = value
            loop_index = torch.arange(N, device=row.device)
        else:
            adj = to_torch_csr_tensor(edge_index, value, size=data.size())

        def get_pe(out: Tensor) -> Tensor:
            if is_torch_sparse_tensor(out):
                return get_self_loop_attr(*to_edge_index(out), num_nodes=N)
            return out[loop_index, loop_index]

        out = adj
        pe_list = [get_pe(out)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            pe_list.append(get_pe(out))

        pe = torch.stack(pe_list, dim=-1)
        data = add_node_attr(data, pe, attr_name="x")

        return data
