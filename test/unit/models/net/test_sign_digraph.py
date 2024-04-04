import hydra
from hydra import compose, initialize_config_dir

from utils.path_helpers import get_config_root


def test_init_sign_digraph(tg_simple_data, sign_transform):
    num_nodes = tg_simple_data.num_nodes
    num_node_features = tg_simple_data.num_node_features
    K = sign_transform.transforms[1].K
    num_node_features += num_node_features * K

    mlp_config_path = get_config_root() / "models" / "net"
    with initialize_config_dir(str(mlp_config_path), version_base="1.2"):
        cfg = compose("sign_digraph.yaml", [f"mlp.node_features={num_node_features}"])
        sign_digraph_net = hydra.utils.instantiate(cfg)
        input = sign_transform(tg_simple_data)
        output = sign_digraph_net(input.x, input.edge_index)
        assert output.shape == (num_nodes, 1)
