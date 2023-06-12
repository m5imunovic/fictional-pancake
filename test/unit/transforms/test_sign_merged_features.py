from transforms.sign_merged_features import SIGNMergedFeatures


def test_sign_merged_features(tg_simple_data):

    K = 5
    num_nodes = tg_simple_data.num_nodes
    num_node_features = tg_simple_data.num_node_features
    expected_num_node_features = num_node_features + num_node_features * K

    sign_transform = SIGNMergedFeatures(K)
    tg_simple_data_transformed = sign_transform(tg_simple_data)

    assert tg_simple_data_transformed.x.shape == (num_nodes, expected_num_node_features)
