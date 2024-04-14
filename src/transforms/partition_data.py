import torch
import torch_geometric.transforms as T

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.loader.cluster import ClusterData


class PartitionData(BaseTransform):
    """Uses METIS clustering to split the data into partitions."""

    def __init__(self, num_parts: int, recursive: bool = False):
        self.num_parts = num_parts
        self.recursive = recursive

    def forward(self, data: Data) -> list[Data]:
        graph = data
        graph.node_ids = torch.arange(data.num_nodes)
        to_undirected = T.ToUndirected()

        cluster_dataset = ClusterData(
            data=to_undirected(graph.clone()),
            num_parts=self.num_parts,
            recursive=self.recursive,
            log=True,
        )

        parts = []
        for part in cluster_dataset:
            subgraph = graph.subgraph(part.node_ids)
            parts.append(subgraph)

        return parts
