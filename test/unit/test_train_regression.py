from pathlib import Path

from train import train


def test_train(test_train_regress_cfg):
    train(test_train_regress_cfg)
    expected_model_path = Path(test_train_regress_cfg.model_output_path) / "best_model.ckpt"
    assert expected_model_path.exists()
    # TODO: assert that the wandb is handled properly
