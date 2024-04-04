from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import ClusterData, DataLoader

from data.dbg_dataset import ClusteredDBGDataset


def save_dir_helper(save_dir: Optional[str], suffix: str) -> Optional[str]:
    if save_dir is None:
        return None
    new_save_dir = Path(save_dir) / suffix
    if not new_save_dir.exists():
        new_save_dir.mkdir(parents=True)
    return new_save_dir


def path_helper(
    path_specialized: Path, path_default: Path, path_descriptor: str
) -> Path:
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
        shuffle: bool = True,
        save_dir: Optional[str] = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

    def _setup_dataloader(self, path: Path, stage: str) -> list[DataLoader]:
        dataset = ClusteredDBGDataset(root=path, transform=self.hparams.transform)
        save_dir = save_dir_helper(self.hparams.save_dir, suffix=stage)
        to_undirected = T.ToUndirected()

        partition_dataloaders = []
        for idx in range(len(dataset)):
            graph = dataset[idx]
            graph.node_ids = torch.arange(graph.num_nodes)
            cluster_dataset = ClusterData(
                data=to_undirected(graph.clone()),
                num_parts=self.hparams.num_parts,
                recursive=self.hparams.recursive,
                save_dir=save_dir_helper(save_dir, str(idx)),
            )
            for data in cluster_dataset:
                d = graph.subgraph(data.node_ids)
                loader = DataLoader(
                    d,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    shuffle=self.hparams.shuffle,
                )
                partition_dataloaders.append(loader)

        return partition_dataloaders

    def common_dataloader(self, path: str, stage: str) -> list[DataLoader]:
        path = path_helper(path, self.hparams.dataset_path, stage)
        return self._setup_dataloader(path, stage)

    def train_dataloader(self) -> list[DataLoader]:
        return self.common_dataloader(self.hparams.train_path, "train")

    def val_dataloader(self) -> list[DataLoader]:
        return self.common_dataloader(self.hparams.val_path, "val")

    def test_dataloader(self) -> list[DataLoader]:
        return self.common_dataloader(self.hparams.test_path, "test")
