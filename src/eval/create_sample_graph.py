import json
import random

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric import utils


def create_sample_graph(display: bool = False):
    assert "0.pt", "Missing sample graph to sample from"

    def to_networkx_value(value: Any) -> Any:
        return value.tolist() if isinstance(value, torch.Tensor) else value

    G_new = nx.MultiDiGraph()
    G_new.add_nodes_from(Gt.nodes)
    data = torch.load("0.pt")
    edge_attrs = ["edge_attr"]
    edge_store = data.edge_stores[0]
    for i, (v, w) in enumerate(Gt.edges(data=False)):
        edge_kwargs = {}
        for key in edge_attrs or []:
            values = to_networkx_value(edge_store[key][i])
            edge_kwargs[key] = values

            G_new.add_edge(v, w, **edge_kwargs)
    G_torch = utils.from_networkx(G_new, group_edge_attrs=edge_attrs)
    torch.save(G_torch, "graph.pt")

    inference_results = {}
    for i in range(G_new.number_of_edges()):
        inference_results[i] = f"{random.random():.2f}"
    with open("inference_result.json", "w") as f:
        json.dump(inference_results, f)

    if display:
        plt.figure(figsize=(20, 14))
        nx.draw_networkx(
            G_new, nx.drawing.nx_agraph.pygraphviz_layout(G_new, prog="fdp")
        )
