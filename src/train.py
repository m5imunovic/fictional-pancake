import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from utils.cfg_helpers import instantiate_component_list
from utils.logger import get_logger
from utils.path_helpers import get_config_root

log = get_logger(__name__)


def upload_model_to_wandb(model_path, artifact_name, dataset_name, metadata=None):
    log.info(f"Uploading model {model_path.name} to wandb.")

    run = wandb.init(project=f"chm13-models-{dataset_name}", job_type="add-model")

    artifact = wandb.Artifact(name=artifact_name, type="ml-model", incremental=True, metadata=metadata)
    artifact.add_file(local_path=model_path)
    if HydraConfig().initialized():
        hydra_cfg_dir = Path(HydraConfig().get().run.dir).absolute() / ".hydra"
        artifact.add_dir(local_path=str(hydra_cfg_dir), name="metadata")
    run.log_artifact(artifact)

    log.info("Model uploaded to wandb.")


def train(cfg: DictConfig) -> None:
    """Train the model using pytorch lightning.

    Args:
        cfg (DictConfig): Configurations for training
    """
    start = datetime.now()
    log.info("Init data module...")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodules)
    log.info("Init model...")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.models)
    log.info("Init callbacks...")
    callbacks: list[pl.Callback] = instantiate_component_list(cfg.callbacks)
    log.info("Init train loggers...")
    loggers = instantiate_component_list(cfg.loggers)

    log.info("Init trainer...")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers, callbacks=callbacks)

    log.info("Save experiment hyperparameters...")

    if cfg.get("train", False):
        log.info("Start training...")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    ckpt_path = trainer.checkpoint_callback.best_model_path or None
    if "model_output_path" in cfg:
        if ckpt_path:
            output_dir = Path(cfg.model_output_path)
            output_dir.mkdir(exist_ok=True, parents=True)
            output_file_path = output_dir / "best_model.ckpt"
            log.info(f"Saving model {ckpt_path} to {output_file_path} ...")
            shutil.copy(ckpt_path, output_file_path)

            metadata = {
                "start": start,
                "duration": datetime.now() - start,
                "dataset": cfg.dataset_name,
                "baseline": cfg.baseline,
            }

            if cfg.metadata:
                metadata.update(dict(cfg.metadata))

            upload_model_to_wandb(output_file_path, cfg.baseline, cfg.dataset_name, metadata=metadata)

    if cfg.get("test", False):
        log.info("Start testing...")
        if not ckpt_path:
            log.warning("Could not find best checkpoint path! Using current weights for testing...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    duration = datetime.now() - start
    log.info(f"Training finished after {duration}")
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
