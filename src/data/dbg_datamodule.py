from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from data.dbg_dataset import DBGDataset


class DBGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: Optional[Path] = None,
        val_path: Optional[Path] = None,
        test_path: Optional[Path] = None,
        dataset_path: Optional[Path] = None,
        transform: T.Compose = None,
        batch_size: int = 1,
        num_workers: int = 0,
        clustered: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_ds: Optional[DBGDataset] = None
        self.val_ds: Optional[DBGDataset] = None
        self.test_ds: Optional[DBGDataset] = None

    @staticmethod
    def path_helper(path_specialized: Path, path_default: Path, path_descriptor: str) -> Path:
        if path_specialized is None:
            return Path(str(path_default).replace("*", path_descriptor))
        return path_specialized

    def train_dataloader(self) -> DataLoader:
        path = self.path_helper(self.hparams.train_path, self.hparams.dataset_path, "train")
        self.train_ds = DBGDataset(root=path, transform=self.hparams.transform, clustered=self.hparams.clustered)
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        path = self.path_helper(self.hparams.val_path, self.hparams.dataset_path, "val")
        self.val_ds = DBGDataset(root=path, transform=self.hparams.transform, clustered=self.hparams.clustered)
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
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
