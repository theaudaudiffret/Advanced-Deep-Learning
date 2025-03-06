import torch
import torch_geometric
from torch_geometric.data import Data

def image_to_graph(
    image: torch.Tensor, conv2d: torch.nn.Conv2d | None = None
) -> Data:
    """
    Converts an image tensor to a PyTorch Geometric Data object.

    Each pixel is treated as a node with features given by its channel values.
    Edges are added between a pixel and all pixels in its receptive field, determined
    by a 3x3 neighborhood. If conv2d is provided, its parameters are used for validation.

    Arguments:
    ----------
    image : torch.Tensor
        Image tensor of shape (C, H, W).
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None. Used to determine the receptive field.

    Returns:
    --------
    Data
        Graph representation of the image.
    """
    # Validate image dimensions.
    assert image.dim() == 3, f"Expected 3D tensor, got {image.dim()}D tensor."

    # If conv2d is provided, ensure its parameters match the expected receptive field.
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 1, "Expected padding of 1 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, "Expected kernel size of 3x3."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."
    # For this implementation, we assume a 3x3 receptive field.
    kernel_size = 3
    pad = kernel_size // 2  # pad=1

    # Get image dimensions.
    C, H, W = image.shape

    # Create node features:
    # Reshape the image so that each pixel (node) has a feature vector of length C.
    # (C, H, W) -> (H*W, C)
    x = image.view(C, -1).transpose(0, 1)

    # Build edges: each pixel is connected to all pixels in its 3x3 neighborhood.
    # We compute the neighbor offsets for a 3x3 grid.
    offsets = [(di, dj) for di in range(-pad, pad + 1) for dj in range(-pad, pad + 1)]

    src, dst, edge_attr = [], [], []
    for i in range(H):
        for j in range(W):
            current_index = i * W + j
            for k, (di, dj) in enumerate(offsets):
                ni, nj = i + di, j + dj
                # Only add valid neighbors (within image bounds)
                if 0 <= ni < H and 0 <= nj < W:
                    neighbor_index = ni * W + nj
                    src.append(current_index)
                    dst.append(neighbor_index)
                    # Create one-hot edge attributes to represent kernel elements.
                    attr = torch.zeros(len(offsets), dtype=torch.float)
                    attr[k] = 1.0
                    edge_attr.append(attr)

    # Construct edge index tensor.
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # Stack edge attributes.
    edge_attr = torch.stack(edge_attr)

    # Return the graph as a PyG Data object.
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


import torch

def graph_to_image(
    data: torch.Tensor, height: int, width: int, conv2d: torch.nn.Conv2d | None = None
) -> torch.Tensor:
    """
    Converts a graph representation of an image to an image tensor.

    Each row of the graph data is assumed to be the node features corresponding
    to one pixel of the image. The image is recovered by reshaping the node features
    to match the original spatial dimensions.

    Arguments:
    ----------
    data : torch.Tensor
        Graph data representation of the image with shape (H*W, C).
    height : int
        Height of the image.
    width : int
        Width of the image.
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None. Used for validating expected parameters.

    Returns:
    --------
    torch.Tensor
        Image tensor of shape (C, H, W).
    """
    # Assumptions: data is a 2D tensor with shape (num_nodes, num_features).
    assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 1, "Expected padding of 1 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, "Expected kernel size of 3x3."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    num_nodes, num_features = data.shape
    assert num_nodes == height * width, "Mismatch between provided height, width and data shape."

    # Recover the image by reshaping the node features.
    # Original node features were arranged as (H*W, C). We need to produce (C, H, W).
    image = data.transpose(0, 1).view(num_features, height, width)

    return image


