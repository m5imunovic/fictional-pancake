from functools import cached_property
from pathlib import Path
from typing import List, Union

import torch
from torch_geometric.data import Data, Dataset
from typeguard import typechecked


class DBGDataset(Dataset):
    def __init__(self, root: Path, transform=None, pre_transform=None):
        super().__init__(str(root), transform, pre_transform)

    @cached_property
    def _raw_file_names(self) -> List[str]:
        with open(Path(self.root) / "raw.csv") as f:
            raw_entries = f.read().splitlines()
            filelist_relative = [entry.split(",")[0] for entry in raw_entries]

        file_parent_path = Path(self.raw_dir)
        return [file_parent_path / filename for filename in filelist_relative]

    @property
    def raw_file_names(self) -> List[str]:
        return self._raw_file_names

    @cached_property
    def _processed_file_names(self) -> List[str]:
        with open(Path(self.root) / "processed.csv") as f:
            raw_entries = f.read().splitlines()
            filelist_relative = [entry.split(",")[0] for entry in raw_entries]

        file_parent_path = Path(self.processed_dir)
        return [file_parent_path / filename for filename in filelist_relative]

    @property
    def processed_file_names(self) -> List[str]:
        return self._processed_file_names

    @property
    def transformed_dir(self) -> Path:
        return Path(self.root) / "transformed"

    @cached_property
    def _transformed_file_names(self) -> List[str]:
        return [str(entry) for entry in self.transformed_dir.glob("*.pt")]

    @property
    def transformed_file_names(self) -> Union[str, List[str]]:
        return self._transformed_file_names

    def len(self) -> int:
        if len(self.transformed_file_names) > 0:
            return len(self.transformed_file_names)

        return len(self.processed_file_names)

    def get(self, idx) -> Data:
        if len(self.transformed_file_names) > 0:
            data = torch.load(self.transformed_file_names[idx])
            return data

        data = torch.load(self.processed_file_names[idx])
        return data
