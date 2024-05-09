"""Uses JSON format of graph from inference debugger to reconstruct small torch graph suitable for inference."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
import torch
from networkx.readwrite.json_graph import node_link_graph
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data


@dataclass
class GraphConverterCfg:
    # path to graph file in node link format
    json_path: Path
    # path to directory where the output graph should be stored (filename is same as for the graph with pt extension)
    output_path: Path


cs = ConfigStore.instance()
cs.store(name="graph_converter_cfg", node=GraphConverterCfg)


def graph_from_json(json_path: Path) -> Data:
    with open(json_path) as handle:
        data = json.load(handle)
        nx_graph = node_link_graph(data, directed=True, multigraph=True)
        graph = from_networkx(nx_graph, group_edge_attrs=["kc", "ln"])

    return graph


def convert_json_graph_to_pyg_graph(json_path: Path, output_path: Path):
    graph = graph_from_json(json_path)
    output_graph_path = output_path / json_path.stem
    torch.save(graph, output_graph_path.with_suffix(".pt"))


@hydra.main(version_base=None, config_name="graph_converter_cfg")
def main(cfg: GraphConverterCfg):
    json_path = Path(cfg.json_path)
    graph_output_path = cfg.output_path / "raw"
    graph_output_path.mkdir(exist_ok=True, parents=True)
    convert_json_graph_to_pyg_graph(json_path, graph_output_path)
    subgraph_output_path = cfg.output_path / "subgraph"
    subgraph_output_path.mkdir(exist_ok=True, parents=True)
    shutil.copy(json_path, subgraph_output_path)
    gml_path = json_path.with_suffix(".gml")
    if gml_path.exists():
        shutil.copy(gml_path, subgraph_output_path)


if __name__ == "__main__":
    main()
