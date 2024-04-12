from data.dbg_dataset import DBGDataset
from torch_geometric.transforms import Compose, Constant


def test_dbg_dataset(unittest_ds_path):
    ds_train = DBGDataset(unittest_ds_path / "train")

    assert len(ds_train) == 3
    expected_names = ["0.pt", "1.pt", "2.pt"]
    for raw_name in ds_train.raw_file_names:
        assert raw_name.name in expected_names, f"Name {raw_name.name} not expected in dataset"

    assert ds_train.processed_file_names == []
    assert ds_train.transformed_dir == unittest_ds_path / "train" / "transformed"

    transform = Compose([Constant(value=0.5)])
    ds_val = DBGDataset(unittest_ds_path / "val", transform=transform)
    assert len(ds_val) == 2
    assert ds_val.processed_file_names == []
    expected_names = ["0.pt", "1.pt"]
    for raw_name in ds_val.raw_file_names:
        assert raw_name.name in expected_names, f"Name {raw_name.name} not expected in dataset"
