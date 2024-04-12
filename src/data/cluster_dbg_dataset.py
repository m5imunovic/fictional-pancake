import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from torch_geometric.data import Data, Dataset

from data.partition_dataset import partition_dataset

logger = logging.getLogger(__name__)


class ClusteredDBGDataset(Dataset):
    def __init__(
        self,
        root: Path,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        num_clusters: int = 2,
    ):
        assert num_clusters >= 2, "Clustered dataset should have at least two clusters"
        self.num_clusters = num_clusters
        super().__init__(str(root), transform, pre_transform)

    @property
    def raw_file_names(self) -> list[Path]:
        if self.transformed_dir.exists():
            raw_entires = list(self.transformed_dir.glob("*.pt"))
        else:
            list(Path(self.raw_dir).glob("*.pt"))
        return raw_entires

    @property
    def transformed_dir(self) -> Path:
        return Path(self.root) / "transformed"

    def process(self):
        if self.num_clusters > 0:
            logger.info(f"Partitioning dataset into {self.num_clusters} clusters")
            start = datetime.now()
            partition_dataset(Path(self.root), self.num_clusters)
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
