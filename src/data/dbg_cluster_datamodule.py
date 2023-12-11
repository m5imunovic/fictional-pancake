from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch_geometric.transforms as T
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader

from data.dbg_dataset import DBGDataset


def save_dir_helper(save_dir: Optional[str], suffix: str) -> Optional[str]:
    if save_dir is None:
        return None
    new_save_dir = Path(save_dir) / suffix
    if not new_save_dir.exists():
        new_save_dir.mkdir(parents=True)
    return new_save_dir


def path_helper(path_specialized: Path, path_default: Path, path_descriptor: str) -> Path:
    if path_specialized is None:
        assert "*" in str(path_default), "Expected special character * in the string"
        return Path(str(path_default).replace("*", path_descriptor))
    return path_specialized


class DBGClusterDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: Optional[Path] = None,
        val_path: Optional[Path] = None,
        test_path: Optional[Path] = None,
        dataset_path: Optional[Path] = None,
        transform: T.Compose = None,
        batch_size: int = 1,
        num_workers: int = 0,
        num_parts: int = 4,
        recursive: bool = True,
        save_dir: Optional[str] = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

    def _setup_dataloader(self, path: Path, stage: str) -> List[DataLoader]:
        dataset = DBGDataset(root=path, transform=self.hparams.transform)
        save_dir = save_dir_helper(self.hparams.save_dir, suffix=stage)
        cluster_data = [
            ClusterData(
                data=dataset[idx],
                num_parts=self.hparams.num_parts,
                recursive=self.hparams.recursive,
                save_dir=save_dir_helper(save_dir, str(idx)),
            )
            for idx in range(len(dataset))
        ]
        return [
            ClusterLoader(
                cluster_data=cluster, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers
            )
            for cluster in cluster_data
        ]

    def common_dataloader(self, stage: str) -> List[DataLoader]:
        path = path_helper(self.hparams.dataset_path, self.hparams.dataset_path, stage)
        return self._setup_dataloader(path, stage)

    def train_dataloader(self) -> List[DataLoader]:
        return self.common_dataloader("train")

    def val_dataloader(self) -> List[DataLoader]:
        return self.common_dataloader("val")

    def test_dataloader(self) -> List[DataLoader]:
        return self.common_dataloader("test")
