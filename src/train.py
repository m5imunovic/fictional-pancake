import shutil
from pathlib import Path
from typing import List, Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from utils.cfg_helpers import instantiate_component_list
from utils.logger import get_logger
from utils.path_helpers import get_config_root

log = get_logger(__name__)


def train(cfg: DictConfig) -> None:
    """Train the model using pytorch lightning.

    Args:
        cfg (DictConfig): Configurations for training
    """
    log.info("Init data module...")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodules)
    log.info("Init model...")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.models)
    log.info("Init callbacks...")
    callbacks: List[pl.Callback] = instantiate_component_list(cfg.callbacks)
    log.info("Init train loggers...")
    loggers = instantiate_component_list(cfg.loggers)

    log.info("Init trainer...")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers, callbacks=callbacks)

    log.info("Save experiment hyperparameters...")

    if cfg.get("train", False):
        log.info("Start training...")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    ckpt_path = trainer.checkpoint_callback.best_model_path
    if "model_output_path" in cfg:
        if ckpt_path is not None:
            output_dir = Path(cfg.model_output_path)
            output_dir.mkdir(exist_ok=True)
            output_file_path = output_dir / "best_model.ckpt"
            log.info(f"Saving model {ckpt_path} to {output_file_path} ...")
            shutil.copy(ckpt_path, output_file_path)

    if cfg.get("test", False):
        log.info("Start testing...")
        if ckpt_path is None:
            log.warning("Could not find best checkpoint path! Using current weights for testing...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    log.info("Experiment finished.")
    return None


@hydra.main(version_base=None, config_path=str(get_config_root()), config_name="train_cfg.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Training script entry point.

    Args:
        cfg (DictConfig): composed configuration

    Returns:
        Optional[float]: output of training function
    """
    if "only_print_config" in cfg:
        print(OmegaConf.to_yaml(cfg))
        return

    return train(cfg)


if __name__ == "__main__":
    main()
