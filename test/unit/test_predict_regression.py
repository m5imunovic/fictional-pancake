from inference import infere


def test_predict_regression(test_predict_cfg):
    # For now we are just testing that the thing does not crashes, later we will expand
    # the test to check that everything gets logged properly
    predictions = infere(test_predict_cfg)
    assert predictions is not None
