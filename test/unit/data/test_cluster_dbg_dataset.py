from data.cluster_dbg_dataset import ClusteredDBGDataset


def test_clustered_dbg_dataset(unittest_ds_path):
    train_ds = ClusteredDBGDataset(unittest_ds_path / "train", num_clusters=2)
    assert len(train_ds) == 6
    expected_names = [
        "0_part_0.pt",
        "0_part_1.pt",
        "1_part_0.pt",
        "1_part_1.pt",
        "2_part_0.pt",
        "2_part_1.pt",
    ]

    for processed_name in train_ds.processed_file_names:
        assert processed_name.name in expected_names, f"Name {processed_name} not expected in dataset"
