from pathlib import Path

from train import train


def test_combined_train(test_combined_train_cfg):
    train(test_combined_train_cfg)
    expected_model_path = Path(test_combined_train_cfg.model_output_path) / "best_model.ckpt"
    assert expected_model_path.exists()
    # TODO: assert that the wandb is handled properly
