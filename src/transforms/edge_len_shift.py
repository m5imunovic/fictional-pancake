from torch_geometric.transforms import BaseTransform


class EdgeLenShift(BaseTransform):
    """This is necessary as we K to edge len when creating datasets from jumboDBG."""

    def __init__(self, constant=501):
        # Initialize with a default constant value of 501
        self.constant = constant

    def __call__(self, data):
        # Check if edge attributes are present
        if data.edge_attr is not None:
            # Ensure the second column exists
            if data.edge_attr.size(1) > 1:
                # Add the constant to the second column of edge attributes
                data.edge_attr[:, 1] += self.constant
        return data
