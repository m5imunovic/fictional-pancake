from unittest import mock
from pathlib import Path

from train import train


@mock.patch("train.upload_model_to_wandb")
def test_train(mock_upload_model, test_train_regress_cfg):
    train(test_train_regress_cfg)
    expected_model_path = Path(test_train_regress_cfg.model_output_path) / "best_model.ckpt"
    assert expected_model_path.exists()
    # TODO: assert that the wandb is handled properly
    assert mock_upload_model.call_count == 1
