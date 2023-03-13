"""Implementation of Pointwise Feedforward neural networks."""

import torch
from typing import List


class PointWiseFeedForward(torch.nn.Module):
    """A pointwise feedforward network.

    The implementation is based on the described network in Vawani et al.
    https://arxiv.org/pdf/1706.03762.pdf.

    Attributes:
    inp_dim: The input dimension (coming from the attention layer).
    hidden_dims: A list of two integers that determine the hidden layers.
    use_convolution_instead: Whether we want to use a 1x1 conv to represent
      the pointwise-fully connected network.
    """

    def __init__(self,
                 inp_dim: int,
                 hidden_dims: List[int],
                 use_1x1_conv: bool = False):
        """Initializer of the PointWiseFeedForward class."""

        super().__init__()
        self.inp_dim = inp_dim
        self.hidden_dims = hidden_dims
        # in our experiments, nn.Linear is faster than nn.Conv1d
        self.use_convolution_instead = use_1x1_conv

        if self.use_convolution_instead:
            params = {
                'in_channels': self.inp_dim,
                'out_channels': self.hidden_dims[0],
                'kernel_size': 1,
                'stride': 1,
                'bias': True
            }
            self.first_layer = torch.nn.Sequential(torch.nn.Conv1d(**params),
                                                   torch.nn.ReLU())
            params = {
                'in_channels': self.hidden_dims[0],
                'out_channels': self.hidden_dims[1],
                'kernel_size': 1,
                'stride': 1,
                'bias': True
            }
            self.second_layer = torch.nn.Conv1d(**params)
        else:
            self.first_layer = torch.nn.Sequential(
                torch.nn.Linear(self.inp_dim, self.hidden_dims[0]),
                torch.nn.ReLU())
            self.second_layer = torch.nn.Linear(self.hidden_dims[0],
                                                self.hidden_dims[1])

        self.normalization = torch.nn.LayerNorm(self.inp_dim)

    def forward(self, inputs: torch.Tensor):
        """The forward call of the PWFF mechanism.

        Args:
            inputs: The input tensor that we need to pass through the network.

        Returns:
            The output tensor of the PWFF operations.

        Raises:
            None.

        """
        if self.use_convolution_instead:
            inputs = inputs.permute(0, 1)

        outputs = self.second_layer(self.first_layer(inputs))
        # Applying the residual connection.
        outputs += inputs

        if self.use_convolution_instead:
            return self.normalization(outputs.permute(0, 2, 1))
        else:
            return self.normalization(outputs)
