import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from eval.utils.color_map import ColorMap, ColorMapGen
from eval.utils.to_networkx import to_networkx

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


@dataclass
class DebuggerCfg:
    sample_path: Path
    scores_path: Path
    output_path: Path
    seed: int = 1
    neighborhood: int = 6
    samples: int | None = 2
    edge_id: int | None = None
    formats: list[str] | None = None


cs = ConfigStore.instance()
cs.store(name="inf_dbg", node=DebuggerCfg)


class InferenceDebugger:
    def __init__(self, graph_file: Path, inference_results_file: Path, render_self: bool = False):
        self.graph = self._load_graph(graph_file)
        probabilities = self._load_inference_results(inference_results_file)
        edge_attrs_src = ["edge_attr"]
        edge_attrs_dst = [["kc", "ln"]]
        logging.info("Creating graph...")
        self.graph_nx = to_networkx(self.graph, edge_attrs_src, edge_attrs_dst, probabilities)
        logging.info("Coloring graph...")
        self._color_edges(self.graph_nx)
        if render_self:
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

        if inference_results_file.suffix == ".json":
            with open(inference_results_file) as f:
                return json.load(f)
        if inference_results_file.suffix in (".np", ".npy"):
            results = np.load(inference_results_file)
            return dict(enumerate(results.flatten()))

    def query_subgraph(self, edge_id, neighborhood_size: int):
        start_edge = self.graph.edge_index[:, edge_id]
        logging.info(f"Starting from {start_edge}")
        edges = self.get_subgraph_edges(start_edge, neighborhood_size)
        nbunch = {node for edge in edges for node in edge}
        subgraph = nx.subgraph(self.graph_nx, nbunch)
        return subgraph

    def get_subgraph_edges(self, start_edge: torch.Tensor, n_hops: int):
        # logging.info(f"Nodes are {self.graph_nx.nodes}")
        # logging.info(f"Edges are {self.graph_nx.edges}")
        bfs = nx.traversal.breadth_first_search.generic_bfs_edges
        edges = bfs(
            self.graph_nx.to_undirected(as_view=True),
            start_edge[0].item(),
            depth_limit=n_hops,
        )
        return edges

    def render_graph(self, graph, name="subgraph"):  # , format: str = "json"):
        logging.info(f"Rendering graph {name}...")
        # self._color_edges(graph)
        # if format == "gml":
        nx.write_gml(graph, f"{name}.gml")
        # if format == "json":
        data = nx.readwrite.json_graph.node_link_data(graph)
        with open(f"{name}.json", "w") as f:
            json.dump(data, f, indent=2)

    def _color_edges(self, graph):
        color_map = ColorMapGen(ColorMap.RedYellowGreen)

        # multigraph iteration:
        for v, w in graph.edges(data=False):
            for idx in graph[v][w]:
                probability = graph[v][w][idx]["p"]
                graph[v][w][idx]["color"] = color_map.generate_color(float(probability))


def get_erroneous_ids(scores_path, expected_scores_path):
    scores = np.load(scores_path).flatten()
    scores = (scores > 0.5).astype(int)
    expected_scores = np.load(expected_scores_path).flatten().astype(int)
    diff = scores != expected_scores
    indices = np.arange(scores.size)
    return indices[diff]


@hydra.main(version_base=None, config_path=".", config_name="inference_debugger.yaml")
def main(cfg: DictConfig) -> None:
    assert cfg.sample_path, "Missing path to graph"
    assert cfg.scores_path, "Missing path to the inference scores"

    graph_path = Path(cfg.sample_path)
    idx = graph_path.stem
    scores_path = Path(cfg.scores_path)
    expected_scores_path = scores_path.parent / f"expected_{scores_path.name}"
    assert expected_scores_path.exists(), f"{expected_scores_path} does not exist"

    np.random.seed(cfg.seed)
    neighborhood = cfg.neighborhood

    if cfg.edge_id and cfg.samples:
        raise ValueError("Only one of `edge_id` or `sample` can be used at same time")
    if cfg.edge_id is None and cfg.samples is None:
        raise ValueError("One of `edge_id` or `sample` must be defined")

    err_ids = get_erroneous_ids(scores_path, expected_scores_path)
    logging.info(f"Found {err_ids.size} erroneous edges")

    logging.info("Selecting erroneous edges...")
    if cfg.edge_id:
        edge_ids = [int(cfg.edge_id)]
    else:
        edge_ids = [np.random.choice(err_ids) for _ in range(int(cfg.samples))]
    inf_deb = InferenceDebugger(graph_path, scores_path)

    output_path = Path(cfg.output_path) / f"{idx}"
    output_path.mkdir(parents=True, exist_ok=True)

    for edge_id in edge_ids:
        if edge_id not in err_ids:
            raise ValueError(f"Edge id {edge_id} not in erroneous ids of graph")
        logging.info(f"Selected {edge_id}")
        subgraph = inf_deb.query_subgraph(edge_id, neighborhood)
        inf_deb.render_graph(subgraph, name=f"{output_path}/{idx}_{edge_id}_subgraph")


if __name__ == "__main__":
    main()
