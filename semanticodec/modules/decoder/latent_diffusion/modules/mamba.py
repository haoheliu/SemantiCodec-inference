import torch
from mamba_ssm import Mamba
import torch.nn as nn


def count_parameters(model):
    """
    Calculate the total number of parameters in a PyTorch model.

    Parameters:
    - model (nn.Module): The PyTorch model.

    Returns:
    - int: The total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MambaBlocks(nn.Module):
    def __init__(self, dim, n_block=4):
        super(MambaBlocks, self).__init__()
        self.mamba_blocks = nn.ModuleList(
            [
                Mamba(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=dim,  # Model dimension d_model
                    d_state=256,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=16,  # Block expansion factor
                )
                for i in range(n_block)
            ]
        )
        self.mamba_norm = nn.ModuleList(
            [nn.LayerNorm(dim, eps=1e-6) for i in range(n_block)]
        )

    def forward(self, x):
        for i, (block, norm) in enumerate(zip(self.mamba_blocks, self.mamba_norm)):
            x = block(x) + x
            if i != len(self.mamba_blocks) - 1:
                x = norm(x)
        return x


if __name__ == "__main__":
    batch, length, dim = 2, 512, 768
    x = torch.randn(batch, length, dim).to("cuda")
    model = MambaBlocks(n_block=4).to("cuda")

    print("Number of parameters:", count_parameters(model))

    y = model(x)

    assert y.shape == x.shape
