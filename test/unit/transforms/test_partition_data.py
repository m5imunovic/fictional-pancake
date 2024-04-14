import torch
from torch_geometric.data import Data

from transforms.partition_data import PartitionData


def test_partition_data():
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 7, 8, 8]],
        dtype=torch.long,
    )

    data = Data(edge_index=edge_index, **{"num_nodes": 9})

    partitions = PartitionData(num_parts=2)(data)
    assert len(partitions) == 2
    assert torch.equal(partitions[0].node_ids, torch.tensor([0, 1, 3, 4, 7]))
    assert torch.equal(partitions[1].node_ids, torch.tensor([2, 5, 6, 8]))
