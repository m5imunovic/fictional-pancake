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
def unittest_ds_path(unittest_ds_zip, tmp_path) -> Path:
    datasets_path = tmp_path / "datasets"
    with zipfile.ZipFile(unittest_ds_zip, "r") as zip_file:
        zip_file.extractall(datasets_path)

    yield datasets_path / "unittest_dataset"
    shutil.rmtree(tmp_path)


@pytest.fixture(scope="function")
def unittest_ds_path_combined(unittest_ds_zip, tmp_path) -> tuple[Path, Path]:
    datasets_path = tmp_path / "datasets"
    with zipfile.ZipFile(unittest_ds_zip, "r") as zip_file:
        zip_file.extractall(datasets_path)

    path1 = datasets_path / "unittest_dataset"
    path2 = datasets_path / "unittest_dataset_2"
    shutil.copytree(path1, path2)

    yield path1, path2
    shutil.rmtree(tmp_path)


@pytest.fixture(scope="function")
def lja_data_path(test_data_path, tmp_path) -> Generator:
    # dataset
    datasets_path = tmp_path / "datasets"
    lja_data_path = test_data_path / "lja_graph"
    lja_tmp_path = datasets_path / "lja_graph"
    lja_tmp_path.mkdir(parents=True)
    shutil.copy(lja_data_path / "edge_index.pt", lja_tmp_path)
    shutil.copy(lja_data_path / "edge_attrs.pt", lja_tmp_path)
    shutil.copy(lja_data_path / "nodes.pt", lja_tmp_path)
    # model
    model_subpath = "models/regression/baseline/E2_L2_H16_C4"
    regression_model_path = test_data_path / model_subpath / "best_model.ckpt"
    tmp_model_path = tmp_path / "storage" / model_subpath
    tmp_model_path.mkdir(parents=True)
    # results
    results_path = tmp_path / "storage/inference"
    results_path.mkdir(parents=True)
    shutil.copy(regression_model_path, tmp_model_path)
    yield tmp_path
    shutil.rmtree(tmp_path)


@pytest.fixture(scope="function")
def test_train_cfg(test_cfg_root, unittest_ds_path) -> DictConfig:
    data_dir = unittest_ds_path.parent.parent
    overrides = [f"paths.data_dir={data_dir}"]
    with initialize_config_dir(version_base=None, config_dir=str(test_cfg_root)):
        cfg = compose(config_name="test_train_config.yaml", overrides=overrides, return_hydra_config=True)
        return cfg


@pytest.fixture(scope="function")
def test_combined_train_cfg(test_cfg_root, unittest_ds_path_combined) -> DictConfig:
    # there is a pair path
    data_dir = unittest_ds_path_combined[0].parent.parent
    overrides = [f"paths.data_dir={data_dir}"]
    with initialize_config_dir(version_base=None, config_dir=str(test_cfg_root)):
        cfg = compose(config_name="test_combined_train_config.yaml", overrides=overrides, return_hydra_config=True)
        return cfg


@pytest.fixture(scope="function")
def test_train_regress_cfg(test_cfg_root, unittest_ds_path) -> DictConfig:
    data_dir = unittest_ds_path.parent.parent
    overrides = [
        f"paths.data_dir={data_dir}",
        "models=test_models_regression",
        "datamodules/transform=test_tf_pe_zscore",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(test_cfg_root)):
        cfg = compose(config_name="test_train_config.yaml", overrides=overrides, return_hydra_config=True)
        return cfg


@pytest.fixture(scope="function")
def test_train_regress_batch_cfg(test_cfg_root, unittest_ds_path) -> DictConfig:
    data_dir = unittest_ds_path.parent.parent
    overrides = [
        f"paths.data_dir={data_dir}",
        "models=test_models_regression",
        "datamodules/transform=test_tf_pe_zscore",
        "datamodules.batch_size=2",
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


@pytest.fixture(scope="function")
def test_inference_regress_cfg(test_cfg_root, unittest_ds_path) -> DictConfig:
    data_dir = unittest_ds_path.parent.parent
    overrides = [
        f"paths.data_dir={data_dir}",
        f"models.storage_path={data_dir/'storage/inference'}",
        "datamodules/transform=test_tf_pe_zscore",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(test_cfg_root)):
        cfg = compose(config_name="test_inf_regression_config.yaml", overrides=overrides, return_hydra_config=True)
        return cfg


@pytest.fixture(scope="function")
def test_predict_cfg(test_cfg_root, lja_data_path) -> DictConfig:
    data_dir = lja_data_path
    overrides = [
        f"paths.data_dir={data_dir}",
        "models/criterion=test_poisson_loss",
        f"models.storage_path={data_dir/'storage/inference'}",
        "datamodules/transform=test_tf_pe_zscore",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(test_cfg_root)):
        cfg = compose(config_name="test_pred_config.yaml", overrides=overrides, return_hydra_config=True)
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
    overrides = ["transforms.0.walk_length=2"]
    with initialize_config_dir(version_base=None, config_dir=str(transforms_path)):
        cfg = compose("test_transform_multidigraph.yaml", overrides=overrides)
        transform = hydra.utils.instantiate(cfg)
        return transform
