from data.dbg_datamodule import DBGDataModule


def test_dbg_datamodule(unittest_ds_path):
    datamodule = DBGDataModule(dataset_path=unittest_ds_path, shuffle=False, num_workers=1, num_clusters=2)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    assert len(val_loader) == 4
    assert len(test_loader) == 4

    num_nodes = [1626, 1777, 1626, 1777, 3470, 3470]
    num_edges = [2363, 2584, 2363, 2584, 5057, 5057]
    for l_idx, loader in enumerate(train_loader):
        print(loader.num_nodes, loader.num_edges)
        print(num_nodes[l_idx], num_edges[l_idx])
        assert loader.num_nodes == num_nodes[l_idx]
        assert loader.num_edges == num_edges[l_idx]
