import logging
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data.dataset import Dataset
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import ClusterData


logger = logging.getLogger(__name__)


class ClusteredDBGDataset(Dataset):
    def __init__(self, root: Path, transform: Callable | None = None, pre_transform: Callable | None = None, num_clusters: int = 0):
        self.num_clusters = num_clusters
        super().__init__(str(root), transform, pre_transform)

    @property
    def raw_file_names(self) -> list[Path]:
        if self.transformed_dir.exists():
            raw_entires = list(self.transformed_dir.glob("*.pt"))
        else:
            raw_entries = list(Path(self.raw_dir).glob("*.pt"))
        return raw_entires

    @property
    def transformed_dir(self) -> Path:
        return Path(self.root) / "transformed"
    
    def process(self):
        if self.num_clusters > 0:
            logger.info(f"Partitioning dataset into {self.num_clusters} clusters")
            start = datetime.now()
            partition_dataset2(Path(self.root), self.num_clusters)
            duration = datetime.now() - start
            logger.info(f"Partitioning finished after {duration}")

    @property
    def processed_file_names(self) -> list[Path]:
        if Path(self.processed_dir).exists():
            if self.num_clusters > 0:
                processed_entires = list(Path(self.processed_dir).glob("*partition*.pt"))
            else:
                # TODO: better way to exclude pre and post filter pt
                processed_entires = list(Path(self.processed_dir).glob("*[0-9].pt"))
            return processed_entires
        return []

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_file_names[idx])


class DBGDataset(Dataset):
    def __init__(self, root: Path, transform: Callable | None = None, pre_transform: Callable | None = None):
        super().__init__(str(root), transform, pre_transform)

    @cached_property
    def raw_file_names(self) -> list[Path]:
        raw_entires = list((Path(self.root) / "raw").glob("*.pt"))
        return raw_entires

    @property
    def transformed_dir(self) -> Path:
        return Path(self.root) / "transformed"

    @cached_property
    def _processed_file_names(self) -> list[str]:
        return [str(entry) for entry in self.transformed_dir.glob("*.pt")]

    @property
    def processed_file_names(self) -> str | list[str]:
        return self._processed_file_names

    def len(self) -> int:
        if len(self.processed_file_names) > 0:
            return len(self.processed_file_names)
        return len(self.raw_file_names)

    def get(self, idx) -> Data:
        if len(self.processed_file_names) > 0:
            data = torch.load(self.processed_file_names[idx])
            return data

        data = torch.load(self.raw_file_names[idx])
        return data


def partition_dataset2(root: Path, num_parts: int, recursive: bool = False):

    dataset = DBGDataset(root)
    to_undirected = T.ToUndirected()
    clusters_dir = Path(dataset.processed_dir)
    if not clusters_dir.exists():
        clusters_dir.mkdir(parents=True)

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
            torch.save(d, clusters_dir / f"{idx}_partition_{part}.pt")


def partition_dataset(root: Path, num_parts: int, recursive: bool = False) -> Path:

    dataset = ClusteredDBGDataset(root)
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

    return transformed_clusters_dir


if __name__ == "__main__":
    # root = Path("/mnt/e/data/datasets/random_species_200M_12_28")
    # stages = []
    root = Path("/data/datasets/chm13_01_16")
    stages = ["train/multidigraph", "val/multidigraph", "test/multidigraph"]
    num_parts = 8
    for stage in stages:
        start = time.time()
        partition_dataset(root / stage, num_parts)
        end = time.time() - start
        print(f"Execution time for {stage}: {end} seconds")