"""This is based on implementation from pytorch_geometric 2.5.1.

I just threw out the parts that I don't need to make it a bit more readable. Have some issues installing newer
pytorch_geometric versions but once that is solved this should be deleted.
"""

from typing import Any, Iterable

import networkx as nx
import torch_geometric
from torch import Tensor


def to_networkx(
    data: torch_geometric.data.Data,
    edge_attrs_src: Iterable[str] | None = None,
    edge_attrs_dst: Iterable[Iterable[str]] | None = None,
    probabilities: dict | None = None,
) -> nx.MultiDiGraph:
    r"""Converts a :class:`torch_geometric.data.Data` instance to a directed :obj:`networkx.MultiDiGraph` if
    :attr:`to_multi` is set to :obj:`True`, or a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData): A
            homogeneous or heterogeneous data object.
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        probabilities: dict

    Examples:
        >>> edge_index = torch.tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> data = Data(edge_index=edge_index, num_nodes=4)
        >>> to_networkx(data)
        <networkx.classes.digraph.DiGraph at 0x2713fdb40d0>
    """
    G = nx.MultiDiGraph()

    def to_networkx_value(value: Any) -> Any:
        return value.tolist() if isinstance(value, Tensor) else value

    G.add_nodes_from(range(data.num_nodes))

    if edge_attrs_dst:
        assert len(edge_attrs_dst) == len(edge_attrs_src), "Mapping length does not match"

    for edge_store in data.edge_stores:
        for i, (v, w) in enumerate(edge_store.edge_index.t().tolist()):
            edge_kwargs: dict[str, Any] = {}
            for j, key in enumerate(edge_attrs_src or []):
                values = to_networkx_value(edge_store[key][i])
                if edge_attrs_dst:
                    for k, edge_attr in enumerate(edge_attrs_dst[j]):
                        edge_kwargs[edge_attr] = values[k]
                else:
                    edge_kwargs[key] = values

            if probabilities:
                edge_kwargs["p"] = float(probabilities[i])

            G.add_edge(v, w, **edge_kwargs)

    return G
