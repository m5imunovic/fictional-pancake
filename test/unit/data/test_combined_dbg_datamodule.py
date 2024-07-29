from data.combined_dbg_datamodule import CombinedDBGDataModule

import hydra
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)


def test_init_combined_data_module(test_combined_train_cfg):
    datamodule = hydra.utils.instantiate(test_combined_train_cfg.datamodules)

    assert datamodule is not None
    assert isinstance(datamodule, CombinedDBGDataModule)

    dataloader = datamodule.train_dataloader()

    data_paths = set()
    for data in dataloader:
        data_paths.add(data.path[0].name)

    expected_paths = {
        "0_part_0.pt",
        "1_part_0.pt",
        "2_part_0.pt",
        "0_part_1.pt",
        "1_part_1.pt",
        "2_part_1.pt",
        "0.pt",
        "1.pt",
        "2.pt",
    }

    assert len(data_paths) == 9
    assert data_paths == expected_paths
