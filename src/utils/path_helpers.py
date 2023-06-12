from pathlib import Path

from typeguard import typechecked


@typechecked
def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


@typechecked
def get_data_path() -> Path:
    return get_project_root() / "data"


@typechecked
def get_config_root() -> Path:
    return get_project_root() / "configs"


@typechecked
def get_test_root() -> Path:
    return get_project_root() / "test"


@typechecked
def project_root_append(path: str) -> Path:
    return get_project_root() / path
