from pathlib import Path
from typing import Union

import torch
import torch_geometric.transforms as T
from pytorch_lightning.core.mixins import HyperparametersMixin

from data.dbg_dataset import DBGDataset


class DBGDataTransformer(HyperparametersMixin):
    def __init__(
        self,
        train_path: Path,
        val_path: Path,
        test_path: Path,
        transform: T.Compose = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.remove_transform_dirs()

    def remove_transform_dirs(self):
        self._remove_transform_dir(self.hparams.train_path)
        self._remove_transform_dir(self.hparams.val_path)
        self._remove_transform_dir(self.hparams.test_path)

    def _remove_transform_dir(self, path: Union[Path, str]):
        transform_path = self.get_transform_path(Path(path), create=False)
        if transform_path.exists():
            for file in transform_path.glob("*.pt"):
                file.unlink()
            transform_path.rmdir()

    def get_transform_path(self, path: Path, create: bool = True) -> Path:
        transformed_path = path / "transformed"
        if create:
            transformed_path.mkdir(exist_ok=True)
        return transformed_path

    def _transform_and_save(self, path: Union[Path, str]):
        path = Path(path)
        if path.exists():
            ds = DBGDataset(path, transform=self.hparams.transform)
            transform_path = self.get_transform_path(path)
            for i in range(len(ds)):
                data = ds[i]
                name = ds.processed_file_names[i].name
                savepath = transform_path / name
                torch.save(data, savepath)
                print(f"Saved data to {savepath}")
        else:
            print(f"Path {path} not found, skipping!")

    def transform_and_save(self):
        self._transform_and_save(self.hparams.train_path)
        self._transform_and_save(self.hparams.val_path)
        self._transform_and_save(self.hparams.test_path)
