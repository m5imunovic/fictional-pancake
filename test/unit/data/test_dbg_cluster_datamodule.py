from pathlib import Path

import pytest
from data.dbg_cluster_datamodule import DBGClusterDataModule, path_helper
from torch.utils.data import DataLoader


@pytest.fixture(params=["multidigraph"])
def datamodule(random_species_data_path, tmp_path, request):
    train_path = random_species_data_path / "train" / request.param
    val_path = random_species_data_path  / "val" / request.param
    test_path = random_species_data_path / "test" / request.param
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

    with pytest.raises(AssertionError, match="Expected special character \* in the string"):
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
