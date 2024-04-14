import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from torch_geometric.data import Data, Dataset

from transforms.partition_data import PartitionData

logger = logging.getLogger(__name__)


class ClusteredDBGDataset(Dataset):
    def __init__(
        self,
        root: Path,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        num_clusters: int = 2,
    ):
        assert num_clusters >= 2, "Clustered dataset should have at least two partitions per sample"
        self.num_parts = num_clusters
        pre_transform = pre_transform or PartitionData(num_parts=self.num_parts)
        super().__init__(str(root), transform=transform, pre_transform=pre_transform)

    @property
    def raw_file_names(self) -> list[Path]:
        return list(Path(self.raw_dir).glob("*.pt"))

    def process(self):
        processed_dir = Path(self.processed_dir)
        clustered_dir = processed_dir / str(self.num_parts)
        if clustered_dir.exists():
            return
        clustered_dir.mkdir(parents=True)

        logger.info(f"Partitioning dataset into {self.num_parts} partitions")
        start = datetime.now()
        for raw_file in self.raw_file_names:
            data = torch.load(raw_file)
            parts = self.pre_transform(data)
            for part_idx, part in enumerate(parts):
                save_path = clustered_dir / f"{raw_file.stem}_part_{part_idx}.pt"
                torch.save(part, save_path)

        logger.info(f"Partitioning finished after {datetime.now() - start}")

    @property
    def processed_file_names(self) -> list[Path]:
        processed_dir = Path(self.processed_dir)
        clustered_dir = processed_dir / str(self.num_parts)
        if not clustered_dir.exists():
            return []
        processed_entires = list(Path(clustered_dir).glob("*part*.pt"))
        return processed_entires

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_file_names[idx])
