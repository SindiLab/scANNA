"""Implementation of scANNA with additive attention."""

from .base._base_model import BaseNeuralAttentionModel
from .base._constants_and_types import _DEVICE_TYPES

import torch


class AdditiveModel(BaseNeuralAttentionModel):
    """Implementation of additive attention and scANNA from base class.

    The additive attention + FFNN is based on Colin Raffel and Daniel P. Ellis,
    which can be found at https://arxiv.org/abs/1512.08756.

    Attributes:

    """

    def __init__(self,
                 input_dimension: int = 5000,
                 output_dimension: int = 11,
                 device: _DEVICE_TYPES = "cpu"):

        super().__init__()
        self.device = device
        self.num_features = input_dimension
        self.out_dim = output_dimension
        self.attention = torch.nn.Linear(self.num_features, self.num_features)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, 1024), torch.nn.Tanh(),
            torch.nn.Linear(1024, 512), torch.nn.Tanh(),
            torch.nn.Linear(512, 256),
            torch.nn.Tanh(), torch.nn.Linear(256, 128), torch.nn.Tanh(),
            torch.nn.Linear(128, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, self.out_dim), torch.nn.Tanh())

    def forward(self, x: torch.Tensor, training: bool = True):
        """ Forward pass for the Feed Forward Attention network.

        Args:
            x: Gene expression matrix from scRNAseq.
            training: A boolean indicating weather we are in training or not.

        Returns:
            Forward call returns three tensors:

            (1) outputs: the output of the task module, in this case
            classification probabilities after a hyperbolic tangent activation.

            (2) alpha: the attention weights.

            (3) x_c: the gene scores.

        Raises:
            None.
        """
        self.training = training
        alphas = self._softmax(self.attention(x))
        gene_scores = self._gene_scores(alphas, x)
        outputs = self._parallel_eval(self.network, gene_scores)
        return outputs, alphas, gene_scores
