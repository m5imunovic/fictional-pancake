from data.dbg_datamodule import DBGDataModule


def test_dbg_datamodule(unittest_ds_path):
    datamodule = DBGDataModule(unittest_ds_path, shuffle=False, num_workers=3, num_clusters=2)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    assert len(val_loader) == 4
    assert len(test_loader) == 2

    num_nodes = [
        [148, 150, 149, 149],
        [161, 159, 160, 160],
        [151, 148, 149, 150],
        [123, 125, 125, 123],
        [161, 159, 160, 160],
        [142, 143, 143, 142],
    ]
    num_edges = [
        [398, 410, 398, 408],
        [442, 436, 438, 438],
        [412, 414, 416, 408],
        [328, 332, 332, 328],
        [438, 442, 444, 434],
        [392, 376, 396, 374],
    ]

    for l_idx, loader in enumerate(train_loader):
        for b_idx, batch in enumerate(loader):
            assert batch.num_nodes == num_nodes[l_idx][b_idx]
            assert batch.num_edges == num_edges[l_idx][b_idx]
