import json
from pathlib import Path

import networkx as nx
import torch

from color_map import ColorMap, ColorMapGen
from to_networkx import to_networkx


class InferenceDebugger:
    def __init__(self, graph_file: Path, inference_results_file: Path):
        self.graph = self._load_graph(graph_file)
        probabilities = self._load_inference_results(inference_results_file)
        edge_attrs_src = ["edge_attr"]
        edge_attrs_dst = [["kc", "ln"]]
        self.graph_nx = to_networkx(
            self.graph, edge_attrs_src, edge_attrs_dst, probabilities
        )
        self._color_edges(self.graph_nx)
        self.render_graph(self.graph_nx, "graph.gml")

    @staticmethod
    def _load_graph(graph_file: Path):
        if not graph_file.exists():
            raise FileNotFoundError(f"{graph_file} does not exist!")
        return torch.load(graph_file)

    @staticmethod
    def _load_inference_results(inference_results_file):
        if not inference_results_file.exists():
            raise FileNotFoundError(f"{inference_results_file} does not exist!")
        with open(inference_results_file) as f:
            return json.load(f)

    def query_subgraph(self, edge_id, neighborhood_size: int):
        start_edge = self.graph.edge_index[:, edge_id]
        print(f"Starting from {start_edge}")
        edges = self.get_subgraph_edges(start_edge, neighborhood_size)
        nbunch = {node for edge in edges for node in edge}
        subgraph = nx.subgraph(self.graph_nx, nbunch)
        return subgraph

    def get_subgraph_edges(self, start_edge: torch.Tensor, n_hops: int):
        # print(f"Nodes are {self.graph_nx.nodes}")
        # print(f"Edges are {self.graph_nx.edges}")
        bfs = nx.traversal.breadth_first_search.generic_bfs_edges
        edges = bfs(
            self.graph_nx.to_undirected(as_view=True),
            start_edge[0].item(),
            depth_limit=n_hops,
        )
        return edges

    def render_graph(self, graph, name="subgraph", format: str = "json"):
        # self._color_edges(graph)
        if format == "gml":
            nx.write_gml(graph, f"{name}.{format}")
        if format == "json":
            data = nx.readwrite.json_graph.node_link_data(graph)
            with open(f"{name}.{format}", "w") as f:
                json.dump(data, f, indent=2)

    def _color_edges(self, graph):
        color_map = ColorMapGen(ColorMap.RedYellowGreen)

        # multigraph iteration:
        for v, w in graph.edges(data=False):
            for idx in graph[v][w]:
                probability = graph[v][w][idx]["p"]
                graph[v][w][idx]["color"] = color_map.generate_color(float(probability))


if __name__ == "__main__":
    id = InferenceDebugger(Path("graph.pt"), Path("inference_result.json"))
    edge_id = 10
    neighborhood_size = 3
    subgraph = id.query_subgraph(edge_id, neighborhood_size)
    id.render_graph(subgraph)
