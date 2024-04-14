from functools import cached_property
from pathlib import Path
from typing import Callable

import torch
from torch_geometric.data import Data, Dataset


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
