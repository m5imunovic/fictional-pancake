from inference import infere


def test_inference(test_inference_cfg):
    # For now we are just testing that the thing does not crashes, later we will expand
    # the test to check that everything gets logged properly
    infere(test_inference_cfg)
    assert True
