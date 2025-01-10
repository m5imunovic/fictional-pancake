from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from data.dbg_dataset_entry import datasample_collate_fn


class CombinedDBGDataModule(pl.LightningDataModule):
    """Implements Lightning Datamodule for loading concatedated datasets for training purposes.

    Args:
        train_datasets: Concatenated datasets for model training round
        val_datasets: Concatenated datasets for model validation round
        shuffle: Should we shuffle the training data, validation is not shuffled for evaluation consistency
        batch_size: Size of a batch
        num_workers: Number of workers to load the dataset in memory
        transform: Transformation placeholder, this is actually already used by hydra interpolation in configuration
        for conactedated datasets, we only need to keep it here not to break initialization.
    """

    def __init__(
        self,
        train_datasets: ConcatDataset,
        val_datasets: ConcatDataset,
        shuffle: bool = True,
        batch_size: int = 1,
        num_workers: int = 0,
        transform: Any = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.train_ds = train_datasets
        self.val_ds = val_datasets

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            collate_fn=datasample_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=datasample_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError("The combined dataloader is not implemented for testing")

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError("The combined dataloader is not implemented for prediction")
