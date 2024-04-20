from pathlib import Path

import numpy as np
import pytest

from eval.utils.inference_debugger import get_erroneous_ids


@pytest.fixture
def scores_paths(tmp_path) -> tuple:
    scores = np.array([[0.5], [0.3], [0.8], [0.6], [0.7]])
    expected_scores = np.array([[1], [0], [1], [0], [1]])

    scores_path = Path(tmp_path) / "0.npy"
    expected_scores_path = Path(tmp_path) / "expected_0.npy"
    np.save(scores_path, scores)
    np.save(expected_scores_path, expected_scores)
    return scores_path, expected_scores_path


def test_get_erroneous_ids(scores_paths):
    scores_path, expected_scores_path = scores_paths

    err_ids = get_erroneous_ids(scores_path, expected_scores_path)
    assert np.array_equal(err_ids, [0, 3])
