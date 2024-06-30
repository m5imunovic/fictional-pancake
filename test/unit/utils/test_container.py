from pathlib import Path

import torch

from utils.container import Container


def test_container_saving(tmp_path):
    entry = {"multiplicity": torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)}
    container = Container(entry)
    path = Path(tmp_path)
    container.save(path)
    container_path = path / "container.pt"
    assert container_path.exists()
    c = torch.jit.load(container_path)
    stored_tensor = getattr(c, "multiplicity")
    assert torch.equal(entry["multiplicity"], stored_tensor)
