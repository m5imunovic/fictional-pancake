from pathlib import Path
from typing import Optional

import torch
import torch_geometric.transforms as T
from torch_geometric.loader import ClusterData

from data.dbg_dataset import DBGDataset


def partition_dataset(root: Path, stages: list[str], num_parts: int, recursive: bool = False) -> dict[Path]:

    cluster_dirs = {}
    for stage in stages:
        dataset = DBGDataset(root=root/stage)
        to_undirected = T.ToUndirected()

        transformed_clusters_dir = dataset.transformed_dir / "clustered"
        if not transformed_clusters_dir.exists():
            transformed_clusters_dir.mkdir(parents=True)

        for idx in range(len(dataset)):
            graph = dataset[idx]
            graph.node_ids = torch.arange(graph.num_nodes)
            cluster_dataset = ClusterData(
                data = to_undirected(graph.clone()),
                num_parts=num_parts,
                recursive=recursive,
                log=True
            )
            for part, data in enumerate(cluster_dataset):
                d = graph.subgraph(data.node_ids)
                torch.save(d, transformed_clusters_dir / f"{idx}_partition_{part}.pt")
        cluster_dirs[stage] = transformed_clusters_dir

    return cluster_dirs