"""This does not removes all the bulges but at least some of the graph gets smaller."""

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RemoveIsolatedNodes(BaseTransform):
    """Remove bulges from the graph."""

    def __call__(self, data: Data) -> Data:
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        # Calculate the degree of each node (in-degree and out-degree)
        row, col = edge_index
        out_degree = torch.bincount(row, minlength=num_nodes)
        in_degree = torch.bincount(col, minlength=num_nodes)

        # Find the nodes that have either in-degree or out-degree zero
        keep_nodes_mask = (out_degree > 0) & (in_degree > 0)
        keep_nodes = torch.nonzero(keep_nodes_mask).view(-1)

        # Create a node map (old to new node index) to preserve only the nodes we keep
        node_map = torch.full((num_nodes,), -1, dtype=torch.long)
        node_map[keep_nodes] = torch.arange(keep_nodes.size(0), dtype=torch.long)

        # Filter out the edges that are connected to nodes we are keeping
        mask = keep_nodes_mask[row] & keep_nodes_mask[col]
        edge_index = edge_index[:, mask]

        # Remap the edge_index to the new node indices
        edge_index = node_map[edge_index]

        if data.x is not None:
            data.x = data.x[keep_nodes]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[mask, :]
        if data.y is not None:
            data.y = data.y[mask]

        data.num_nodes = keep_nodes.size(0)
        data.edge_index = edge_index

        return data
