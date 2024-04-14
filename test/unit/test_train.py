from train import train


def test_train(test_train_cfg):
    train(test_train_cfg)
    assert True

    # tests that the model checkpoint exists
