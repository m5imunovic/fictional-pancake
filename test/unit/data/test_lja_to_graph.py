from data.lja_to_graph import lja_to_graph

import torch


def test_lja_to_graph(test_data_path):
    lja_data_path = test_data_path / "lja_graph"

    lja_to_graph(lja_data_path)
    raw_path = lja_data_path / "raw"
    graph_path = raw_path / "0.pt"
    assert graph_path.exists()
    new_g = torch.load(graph_path)
    expected_g = torch.load(lja_data_path / "0.pt")
    assert new_g.num_nodes == expected_g.num_nodes
    assert new_g.edge_index.shape == expected_g.edge_index.shape
    assert new_g.edge_attr.shape == expected_g.edge_attr.shape

    graph_path.unlink()
    raw_path.rmdir()
