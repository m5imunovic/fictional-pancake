from data.partition_dataset import partition_dataset


def test_partition_dataset(unittest_ds_path):
    num_parts = 4
    stages = ["train", "val"]
    partition_paths = {}
    for stage in stages:
        partition_paths[stage] = partition_dataset(unittest_ds_path / stage, num_parts)

    expected_lens = {
        stages[0]: num_parts * 3,
        stages[1]: num_parts * 2,
    }
    for stage, partition_paths in partition_paths.items():
        assert len(partition_paths) == expected_lens[stage]
        for partition in partition_paths:
            partition.unlink()
        partition_paths[0].parent.rmdir()
