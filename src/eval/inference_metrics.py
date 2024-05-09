from pathlib import Path

import torch
import pandas as pd
from typeguard import typechecked
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class InferenceMetrics:
    columns = ["Accuracy", "Precision", "Recall", "F1Score"]

    def __init__(self, threshold: float = 0.5, multidim_average: str = "samplewise"):
        self.metrics = MetricCollection(
            [
                BinaryAccuracy(threshold=threshold, multidim_average=multidim_average),
                BinaryPrecision(threshold=threshold, multidim_average=multidim_average),
                BinaryRecall(threshold=threshold, multidim_average=multidim_average),
                BinaryF1Score(threshold=threshold, multidim_average=multidim_average),
            ]
        )
        self.names = []

    @typechecked
    def update(self, scores: torch.Tensor, expected_scores: torch.Tensor, sample_path: Path | list[Path]):
        self.metrics.update(scores, expected_scores)
        names = [sp.name for sp in sample_path] if isinstance(sample_path, list) else [sample_path.name]
        self.names.extend(names)

    def reset(self):
        self.metrics.reset()
        self.names = []

    def finalize(self) -> pd.DataFrame:
        data = []
        for name in self.metrics.keys():
            if name.removeprefix("Binary") in InferenceMetrics.columns:
                value = self.metrics[name].compute()
                data.append(value)
        data = torch.stack(data).T
        if data.dim() == 1:
            # to avoid "shape of passed value is X, indices imply Y" error
            data = data.unsqueeze(0)

        df = pd.DataFrame(columns=InferenceMetrics.columns, data=data.tolist())
        df.insert(loc=0, column="Name", value=self.names)
        self.reset()

        return df
