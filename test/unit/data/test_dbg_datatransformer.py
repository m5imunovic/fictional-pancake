import pytest
from torch_geometric.transforms import Compose, Constant

from data.dbg_datatransformer import DBGDataTransformer


@pytest.mark.parametrize("gtype", ["multidigraph"])
def test_dbg_datatransformer(rs_20000_data_path, gtype):
    transform = Compose([Constant(value=0.5)])

    transformer = DBGDataTransformer(
        train_path=rs_20000_data_path / "train" / gtype,
        val_path=rs_20000_data_path / "val" / gtype,
        test_path=rs_20000_data_path / "test" / gtype,
        transform=transform,
    )

    transformer.transform_and_save()

    assert (transformer.hparams.train_path / "transformed" / "0.pt").exists()
    assert (transformer.hparams.val_path / "transformed" / "0.pt").exists()
    assert (transformer.hparams.test_path / "transformed" / "0.pt").exists()

    # clean up
    transformer.remove_transform_dirs()
