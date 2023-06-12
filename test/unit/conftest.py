import os
from pathlib import Path

import hydra
import pytest
import torch
import torch_geometric as tg
from hydra import compose, initialize, initialize_config_dir
from omegaconf import DictConfig

from utils.path_helpers import get_config_root, get_test_root


@pytest.fixture
def cfg_project_train() -> DictConfig:
    file_parent_path = Path(__file__).parent
    relative_config_path = os.path.relpath(get_config_root(), file_parent_path)
    with initialize(version_base="1.2", config_path=relative_config_path):
        cfg = compose(config_name="train_cfg.yaml", return_hydra_config=True)
        return cfg


@pytest.fixture
def test_data_path() -> Path:
    return get_test_root() / "data"


@pytest.fixture
def random_species_data_path(test_data_path) -> Path:
    return test_data_path / "random_species_10_13"


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
