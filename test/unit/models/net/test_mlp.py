import hydra
import torch
from hydra import compose, initialize_config_dir

from utils.path_helpers import get_config_root


def test_init_mlp():
    mlp_config_path = get_config_root() / "models" / "net" / "mlp"
    with initialize_config_dir(str(mlp_config_path), version_base="1.2"):
        cfg = compose("mlp.yaml")
        mlp = hydra.utils.instantiate(cfg)
        input = torch.rand((128, cfg.node_features))
        output = mlp(input)
        assert output.shape == (128, cfg.out_features)
