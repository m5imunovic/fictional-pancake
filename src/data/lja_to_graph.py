from pathlib import Path

import torch
from torch_geometric.data import Data


def lja_to_graph(path: Path):
    assert path.exists(), f"Path {path} does not exist!"
    edge_index_path = path / "edge_index.pt"
    edge_attrs_path = path / "edge_attrs.pt"
    nodes_path = path / "nodes.pt"
    assert edge_index_path.exists(), "Missing edge index data"
    assert edge_attrs_path.exists(), "Missing edge attrs data"

    edge_index = torch.load(edge_index_path)
    edge_attr = torch.load(edge_attrs_path)
    nodes = torch.load(nodes_path)
    # len works both for list and tensor types
    num_nodes = len(nodes)

    # both edge index and attrs come in (len, 2) shape
    graph = Data(num_nodes=num_nodes, edge_index=edge_index.T, edge_attr=edge_attr)
    raw_dir = path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    torch.save(graph, raw_dir / "0.pt")
