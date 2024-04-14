from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from data.cluster_dbg_dataset import ClusteredDBGDataset
from data.dbg_dataset import DBGDataset


class DBGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: Optional[Path] = None,
        val_path: Optional[Path] = None,
        test_path: Optional[Path] = None,
        dataset_path: Optional[Path] = None,
        transform: T.Compose = None,
        shuffle: bool = True,
        batch_size: int = 1,
        num_workers: int = 0,
        num_clusters: int = 2,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_ds: Optional[ClusteredDBGDataset] = None
        self.val_ds: Optional[ClusteredDBGDataset] = None
        self.test_ds: Optional[DBGDataset] = None

    @staticmethod
    def path_helper(path_specialized: Path | None, path_default: Path, path_descriptor: str) -> Path:
        if not path_specialized:
            return Path(path_default) / path_descriptor
        return path_specialized

    def train_dataloader(self) -> DataLoader:
        path = self.path_helper(self.hparams.train_path, self.hparams.dataset_path, "train")
        self.train_ds = ClusteredDBGDataset(
            root=path,
            transform=self.hparams.transform,
            num_clusters=self.hparams.num_clusters,
        )
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=self.hparams.shuffle,
        )

    def val_dataloader(self) -> DataLoader:
        path = self.path_helper(self.hparams.val_path, self.hparams.dataset_path, "val")
        self.val_ds = ClusteredDBGDataset(
            root=path,
            transform=self.hparams.transform,
            num_clusters=self.hparams.num_clusters,
        )
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=self.hparams.shuffle,
        )

    def test_dataloader(self) -> DataLoader:
        path = self.path_helper(self.hparams.test_path, self.hparams.dataset_path, "test")
        self.test_ds = DBGDataset(root=path, transform=self.hparams.transform)
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        # TODO: implement predict dataloader for data without labels, for now use test data
        path = self.path_helper(self.hparams.test_path, self.hparams.dataset_path, "test")
        self.test_ds = DBGDataset(root=path, transform=self.hparams.transform)
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
