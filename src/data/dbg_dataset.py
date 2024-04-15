from functools import cached_property
from pathlib import Path
from typing import Callable

import torch
from torch_geometric.data import Dataset

from data.dbg_dataset_entry import DataSample


class DBGDataset(Dataset):
    def __init__(
        self,
        root: Path,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        super().__init__(str(root), transform, pre_transform)

    @cached_property
    def raw_file_names(self) -> list[Path]:
        raw_entires = list((Path(self.root) / "raw").glob("*.pt"))
        return raw_entires

    @property
    def transformed_dir(self) -> Path:
        return Path(self.processed_dir)

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

    def get(self, idx) -> DataSample:
        path = self.raw_file_names[idx]
        if len(self.processed_file_names) > 0:
            path = self.processed_file_names[idx]
        data = torch.load(path)
        data = data if self.transform is None else self.transform(data)
        return DataSample(data, path)

    def __getitem__(self, idx) -> DataSample:
        data = self.get(self.indices()[idx])
        return data
