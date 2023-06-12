from typing import List, TypeVar

import hydra
from omegaconf import DictConfig

from utils.logger import get_logger

T = TypeVar("T")
log = get_logger(__name__)


def instantiate_component_list(cfg: DictConfig) -> List[T]:
    """Instantiates multiple components from config."""
    components: List[T] = []

    if not cfg:
        log.warning("Config is empty. Returning empty list as result!")
        return components

    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Component list config must be a DictConfig type, got {type(cfg)}!")

    for _, component_cfg in cfg.items():
        if not isinstance(component_cfg, DictConfig):
            log.warning(
                f"Component config must be a DictConfig type, got {type(component_cfg)}! Skipping component <{component_cfg}>"
            )
            continue
        if "_target_" not in component_cfg:
            log.warning(f"Missing _target_ class definition. Skipping component <{component_cfg}>")
            continue
        log.info(f"Instantiating component <{component_cfg._target_}>")
        components.append(hydra.utils.instantiate(component_cfg))

    return components
