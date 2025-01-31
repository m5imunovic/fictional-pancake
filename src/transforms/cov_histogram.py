import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class CovHistogram(BaseTransform):
    def __init__(self, extend: int = 5, use_range: bool = False):
        self.extend = extend
        self.use_range = use_range

    def __call__(self, data: Data) -> Data:
        edge_attr = data.edge_attr[:, 0]  # Take the first column
        clamped_values = torch.clamp(edge_attr, min=0.0, max=100.0)
        if self.use_range:
            hist, bins = torch.histogram(clamped_values, bins=100, range=(0, 100))
        else:
            hist, bins = torch.histogram(clamped_values, bins=100)

        max_index = torch.argmax(hist[5:95]) + 5  # Offset by 5 to get actual index
        start_idx = max_index - self.extend + 1
        end_idx = max_index + self.extend
        selected_bins = bins[start_idx : end_idx + 1]
        data.graph_attr = selected_bins.unsqueeze(0)
        return data
