import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_project_train_config(cfg_project_train: DictConfig):
    assert cfg_project_train
    assert cfg_project_train.paths

    # This only works if the return_hydra_config=True is set in the init call
    HydraConfig().set_config(cfg_project_train)

    hydra.utils.instantiate(cfg_project_train.paths)
