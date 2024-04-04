from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from data.dbg_datatransformer import DBGDataTransformer
from utils.logger import get_logger
from utils.path_helpers import get_config_root

log = get_logger(__name__)


def transform(cfg: DictConfig) -> None:
    """Transform the data using various built-in and custom transforms.

    Args:
        cfg (DictConfig): Configurations for data transformation
    """
    log.info("Starting data transformation...")

    if cfg.get("datatransforms", False):
        transformer: DBGDataTransformer = hydra.utils.instantiate(cfg.datatransforms)
        log.info(f"Start transforming {cfg.graph.type} data...")
        transformer.transform_and_save()

    log.info("Data transformation finished.")
    return None


@hydra.main(
    version_base=None,
    config_path=str(get_config_root()),
    config_name="transform_cfg.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    """Transformation script entry point.

    Args:
        cfg (DictConfig): composed configuration
    """
    if "only_print_config" in cfg:
        print(OmegaConf.to_yaml(cfg))
        return

    return transform(cfg)


if __name__ == "__main__":
    main()
