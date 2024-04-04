import pytest

from data.partition import partition_dataset


@pytest.fixture(params=["multidigraph"])
def train_multi_path(rs_full_30000_data_path, request):
    return rs_full_30000_data_path / "train" / request.param


@pytest.mark.parametrize("graph", ["multidigraph"])
def test_partition_dataset(rs_full_30000_data_path, graph):
    num_parts = 4
    stages = [f"train/{graph}", f"val/{graph}", f"test/{graph}"]
    partition_paths = partition_dataset(rs_full_30000_data_path, stages, num_parts)

    expected_lens = {
        stages[0]: num_parts * 6,
        stages[1]: num_parts * 2,
        stages[2]: num_parts * 2,
    }
    for stage, partition_path in partition_paths.items():
        partitions = list(partition_path.glob("*.pt"))
        assert len(partitions) == expected_lens[stage]
        for partition in partitions:
            partition.unlink()
        partition_path.rmdir()
