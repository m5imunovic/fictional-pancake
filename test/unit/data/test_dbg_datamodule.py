from data.dbg_datamodule import DBGDataModule


def test_dbg_datamodule(unittest_ds_path):
    datamodule = DBGDataModule(dataset_path=unittest_ds_path, shuffle=False, num_workers=1, num_clusters=2)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    assert len(val_loader) == 4
    assert len(test_loader) == 4

    num_nodes = [1626, 3470, 3470, 1777, 1777, 1626]
    num_edges = [2363, 5057, 5057, 2584, 2584, 2363]
    for l_idx, sample in enumerate(train_loader):
        print(sample)
        print(sample.data.num_nodes, sample.data.num_edges)
        print(num_nodes[l_idx], num_edges[l_idx])
        assert sample.data.num_nodes == num_nodes[l_idx]
        assert sample.data.num_edges == num_edges[l_idx]


def test_dbg_datamodule_two_parts(unittest_ds_path):
    datamodule1 = DBGDataModule(dataset_path=unittest_ds_path, shuffle=False, num_workers=1, num_clusters=2)
    datamodule2 = DBGDataModule(dataset_path=unittest_ds_path, shuffle=False, num_workers=1, num_clusters=3)
    train_loader1 = datamodule1.train_dataloader()
    train_loader2 = datamodule2.train_dataloader()
    assert len(train_loader1) == 6
    assert len(train_loader2) == 9
    assert len(train_loader1) == 6
