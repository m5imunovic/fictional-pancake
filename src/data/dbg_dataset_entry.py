from pathlib import Path
from typing import NamedTuple, Sequence

from torch_geometric.data import Batch, Data


class DataSample(NamedTuple):
    """We want to be able to track down the file from which sample is loaded. For batched samples it is also important
    to know where graphs are.

    With shuffling this is not possible without using metadata
    """

    data: Data | Sequence[Data]
    path: Path | Sequence[Path]
    ei_ptr: list


def datasample_collate_fn(batch: list[DataSample]) -> DataSample:
    data = [item.data for item in batch]
    paths = [item.path for item in batch]
    ei_ptr = [item.data.edge_index.shape[1] for item in batch]
    return DataSample(data=Batch.from_data_list(data), path=paths, ei_ptr=ei_ptr)
