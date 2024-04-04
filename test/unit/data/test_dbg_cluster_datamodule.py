from pathlib import Path

import pytest
from data.dbg_cluster_datamodule import DBGClusterDataModule, path_helper
from torch.utils.data import DataLoader


@pytest.fixture(params=["multidigraph"])
def datamodule(rs_20000_data_path, tmp_path, request):
    train_path = rs_20000_data_path / "train" / request.param
    val_path = rs_20000_data_path / "val" / request.param
    test_path = rs_20000_data_path / "test" / request.param
    return DBGClusterDataModule(train_path, val_path, test_path, save_dir=tmp_path)


@pytest.fixture(params=["multidigraph"])
def datamodule_full(rs_full_30000_data_path, tmp_path, request):
    train_path = rs_full_30000_data_path / "train" / request.param
    val_path = rs_full_30000_data_path / "val" / request.param
    test_path = rs_full_30000_data_path / "test" / request.param
    return DBGClusterDataModule(train_path, val_path, test_path, save_dir=tmp_path)


def test_path_helper(datamodule):
    # Test path_helper method
    path_specialized = Path("/path/to/specialized")
    path_default = Path("/path/to/default")
    path_descriptor = "descriptor"

    result = path_helper(path_specialized, path_default, path_descriptor)

    assert isinstance(result, Path)
    assert result == path_specialized
    result = path_helper(None, path_default / "*", path_descriptor)
    assert result == path_default / path_descriptor

    with pytest.raises(
        AssertionError, match=r"Expected special character \* in the string"
    ):
        result = path_helper(None, path_default, path_descriptor)


def test_train_dataloader(datamodule):
    # Test train_dataloader method
    train_loader = datamodule.train_dataloader()

    assert isinstance(train_loader, list)
    assert all(isinstance(loader, DataLoader) for loader in train_loader)


def test_val_dataloader(datamodule):
    # Test val_dataloader method
    val_loader = datamodule.val_dataloader()

    assert isinstance(val_loader, list)
    assert all(isinstance(loader, DataLoader) for loader in val_loader)


def test_test_dataloader(datamodule):
    # Test test_dataloader method
    test_loader = datamodule.test_dataloader()

    assert isinstance(test_loader, list)
    assert all(isinstance(loader, DataLoader) for loader in test_loader)


def test_full_train_dataloader(datamodule_full):
    train_loader = datamodule_full.train_dataloader()
    assert isinstance(train_loader, list)
    assert all(isinstance(loader, DataLoader) for loader in train_loader)

    assert len(train_loader) == 24

    num_nodes = [
        [148, 150, 149, 149],
        [161, 159, 160, 160],
        [151, 148, 149, 150],
        [123, 125, 125, 123],
        [161, 159, 160, 160],
        [142, 143, 143, 142],
    ]
    num_edges = [
        [398, 410, 398, 408],
        [442, 436, 438, 438],
        [412, 414, 416, 408],
        [328, 332, 332, 328],
        [438, 442, 444, 434],
        [392, 376, 396, 374],
    ]

    for l_idx, loader in enumerate(train_loader):
        for b_idx, batch in enumerate(loader):
            assert batch.num_nodes == num_nodes[l_idx][b_idx]
            assert batch.num_edges == num_edges[l_idx][b_idx]


def test_full_val_dataloader(datamodule_full):
    val_loader = datamodule_full.val_dataloader()
    assert isinstance(val_loader, list)
    assert all(isinstance(loader, DataLoader) for loader in val_loader)

    assert len(val_loader) == 8
