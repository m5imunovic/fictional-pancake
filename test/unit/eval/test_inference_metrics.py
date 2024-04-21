from pathlib import Path

import pandas as pd
import torch
from pandas.testing import assert_series_equal

from eval.inference_metrics import InferenceMetrics


def test_inference_metrics_improving():
    """The function tests for the equality with the following dataframe:
    ```
    Name  Accuracy Precision   Recall  F1Score
    10pct      0.1      0.25 0.142857 0.181818
    20pct      0.2      0.40 0.285714 0.333333
    30pct      0.3      0.50 0.285714 0.363636
    40pct      0.4      0.60 0.428571 0.500000
    50pct      0.5      0.75 0.428571 0.545455
    60pct      0.6      1.00 0.428571 0.600000
    ```
    """

    im = InferenceMetrics(threshold=0.5)

    expected_scores = [1, 1, 0, 1, 0, 0, 1, 1, 1, 1]
    scores = {
        "10pct": [0.8, 0.4, 0.6, 0.4, 0.7, 0.7, 0.2, 0.3, 0.4, 0.3],
        "20pct": [0.8, 0.8, 0.6, 0.4, 0.7, 0.7, 0.2, 0.3, 0.4, 0.3],
        "30pct": [0.8, 0.8, 0.2, 0.4, 0.7, 0.7, 0.2, 0.3, 0.4, 0.3],
        "40pct": [0.8, 0.8, 0.2, 0.9, 0.7, 0.7, 0.2, 0.3, 0.4, 0.3],
        "50pct": [0.8, 0.8, 0.2, 0.9, 0.2, 0.7, 0.2, 0.3, 0.4, 0.3],
        "60pct": [0.8, 0.8, 0.2, 0.9, 0.2, 0.1, 0.2, 0.3, 0.4, 0.3],
    }

    for name, score in scores.items():
        score = torch.tensor(score).unsqueeze(0)
        expected_score = torch.tensor(expected_scores).unsqueeze(0)
        im.update(score, expected_score, Path(name))

    result = im.finalize()
    """Name  Accuracy Precision   Recall  F1Score 10pct      0.1      0.25 0.142857 0.181818 20pct      0.2      0.40
    0.285714 0.333333 30pct      0.3      0.50 0.285714 0.363636 40pct      0.4      0.60 0.428571 0.500000 50pct
    0.5      0.75 0.428571 0.545455 60pct      0.6      1.00 0.428571 0.600000."""

    assert_series_equal(pd.Series([k for k in scores], name="Name"), result["Name"])
    assert_series_equal(pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], name="Accuracy"), result["Accuracy"])
    assert_series_equal(pd.Series([0.25, 0.40, 0.50, 0.60, 0.75, 1.00], name="Precision"), result["Precision"])


def test_inference_metrics_mix():
    """The function tests for the equality with the following dataframe:
    ```
    Name    Accuracy Precision   Recall  F1Score
    20pct_1      0.2      0.40 0.285714 0.333333
    50pct_1      0.5      0.75 0.428571 0.545455
    20pct_2      0.2      0.40 0.285714 0.333333
    50pct_2      0.5      0.75 0.428571 0.545455
    ```
    """
    im = InferenceMetrics(threshold=0.5)

    expected_scores = [1, 1, 0, 1, 0, 0, 1, 1, 1, 1]
    scores = {
        "20pct_1": [0.8, 0.8, 0.6, 0.4, 0.7, 0.7, 0.2, 0.3, 0.4, 0.3],
        "50pct_1": [0.8, 0.8, 0.2, 0.9, 0.2, 0.7, 0.2, 0.3, 0.4, 0.3],
        "20pct_2": [0.8, 0.8, 0.6, 0.4, 0.7, 0.7, 0.2, 0.3, 0.4, 0.3],
        "50pct_2": [0.8, 0.8, 0.2, 0.9, 0.2, 0.7, 0.2, 0.3, 0.4, 0.3],
    }

    for name, score in scores.items():
        score = torch.tensor(score).unsqueeze(0)
        expected_score = torch.tensor(expected_scores).unsqueeze(0)
        im.update(score, expected_score, Path(name))

    result = im.finalize()
    """Name    Accuracy Precision   Recall  F1Score 20pct_1      0.2      0.40 0.285714 0.333333 50pct_1      0.5
    0.75 0.428571 0.545455 20pct_2      0.2      0.40 0.285714 0.333333 50pct_2      0.5      0.75 0.428571
    0.545455."""
    assert_series_equal(pd.Series([k for k in scores], name="Name"), result["Name"])
    assert_series_equal(pd.Series([0.2, 0.5, 0.2, 0.5], name="Accuracy"), result["Accuracy"])
    assert_series_equal(pd.Series([0.40, 0.75, 0.40, 0.75], name="Precision"), result["Precision"])
