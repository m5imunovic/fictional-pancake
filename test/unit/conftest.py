import shutil
import zipfile
from pathlib import Path
from typing import Generator

import hydra
import pytest
import torch
import torch_geometric as tg
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from utils.path_helpers import get_config_root, get_test_root


@pytest.fixture(scope="session")
def test_data_path() -> Path:
    return get_test_root() / "data"


@pytest.fixture(scope="session")
def test_cfg_root(test_data_path) -> Path:
    return get_test_root() / "configs"


@pytest.fixture(scope="session")
def unittest_ds_zip(test_data_path) -> Path:
    return test_data_path / "unittest_dataset.zip"


@pytest.fixture(scope="function")
def unittest_ds_path(unittest_ds_zip, tmp_path) -> Generator:
    datasets_path = tmp_path / "datasets"
    with zipfile.ZipFile(unittest_ds_zip, "r") as zip_file:
        zip_file.extractall(datasets_path)

    yield datasets_path / "unittest_dataset"
    shutil.rmtree(tmp_path)


@pytest.fixture(scope="function")
def test_train_cfg(test_cfg_root, unittest_ds_path) -> DictConfig:
    data_dir = unittest_ds_path.parent.parent
    overrides = [f"paths.data_dir={data_dir}"]
    with initialize_config_dir(version_base=None, config_dir=str(test_cfg_root)):
        cfg = compose(config_name="test_train_config.yaml", overrides=overrides, return_hydra_config=True)
        return cfg


@pytest.fixture(scope="function")
def test_train_regress_cfg(test_cfg_root, unittest_ds_path) -> DictConfig:
    data_dir = unittest_ds_path.parent.parent
    overrides = [
        f"paths.data_dir={data_dir}",
        "models/criterion=test_poisson_loss",
        "datamodules/transform=test_tf_pe_zscore",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(test_cfg_root)):
        cfg = compose(config_name="test_train_config.yaml", overrides=overrides, return_hydra_config=True)
        return cfg


@pytest.fixture(scope="function")
def test_inference_cfg(test_cfg_root, unittest_ds_path) -> DictConfig:
    data_dir = unittest_ds_path.parent.parent
    overrides = [f"paths.data_dir={data_dir}"]
    with initialize_config_dir(version_base=None, config_dir=str(test_cfg_root)):
        cfg = compose(config_name="test_inf_config.yaml", overrides=overrides, return_hydra_config=True)
        return cfg


@pytest.fixture
def tg_simple_data():
    x = torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.float).t()
    edge_index = torch.tensor([[0, 0, 1, 1, 2], [1, 1, 0, 2, 1]])
    data = tg.data.Data(x=x, edge_index=edge_index).to(torch.device("cpu"))
    data.validate(raise_on_error=True)
    return data


@pytest.fixture
def sign_transform():
    transforms_path = get_config_root() / "datamodules" / "transform"
    with initialize_config_dir(str(transforms_path), version_base="1.2"):
        cfg = compose("transform_sign_digraph.yaml")
        sign_transform = hydra.utils.instantiate(cfg)
        return sign_transform


@pytest.fixture
def resgated_mdg_transform(test_cfg_root):
    transforms_path = test_cfg_root / "datamodules" / "transform"
    with initialize_config_dir(version_base=None, config_dir=str(transforms_path)):
        cfg = compose("test_transform_multidigraph.yaml")
        transform = hydra.utils.instantiate(cfg)
        return transform
