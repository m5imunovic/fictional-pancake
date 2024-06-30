from pathlib import Path

import torch


class Container(torch.nn.Module):
    def __init__(self, entries: dict):
        super().__init__()
        for key, val in entries.items():
            # assert val.dtype is torch.float
            setattr(self, key, val)

    def save(self, path: Path):
        assert path.is_dir(), f"Path {path} is expected to be directory"
        jit_container = torch.jit.script(self)
        jit_container.save(path / "container.pt")
