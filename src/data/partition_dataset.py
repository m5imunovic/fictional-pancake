import time
from pathlib import Path

import torch
import torch_geometric.transforms as T
from torch_geometric.loader import ClusterData

from data.dbg_dataset import DBGDataset


def partition_dataset(root: Path, num_parts: int, recursive: bool = False) -> list[Path]:
    dataset = DBGDataset(root)
    to_undirected = T.ToUndirected()
    clusters_dir = Path(dataset.processed_dir)
    if not clusters_dir.exists():
        clusters_dir.mkdir(parents=True)

    paths = []
    for idx in range(len(dataset)):
        graph = dataset[idx]
        graph.node_ids = torch.arange(graph.num_nodes)
        cluster_dataset = ClusterData(
            data=to_undirected(graph.clone()),
            num_parts=num_parts,
            recursive=recursive,
            log=True,
        )
        for part, data in enumerate(cluster_dataset):
            d = graph.subgraph(data.node_ids)
            output_path = clusters_dir / f"{idx}_partition_{part}.pt"
            torch.save(d, output_path)
            paths.append(output_path)

    return paths


if __name__ == "__main__":
    root = Path("/data/datasets/unittest_dataset/")
    stages = ["train", "val", "test"]
    num_parts = 8
    for stage in stages:
        start = time.time()
        partition_dataset(root / stage, num_parts)
        end = time.time() - start
        print(f"Execution time for {stage}: {end} seconds")
