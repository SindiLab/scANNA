"""Implementation of scANNA model with attention + projection blocks."""

from .base._base_projection_model import BaseProjectionModel
from .base._constants_and_types import _DEVICE_TYPES
import torch


class ProjectionAttention(BaseProjectionModel):
    """scANNA model with attention and projection blocks.

    Attributes:
        device: A string (either "cuda" or "cpu") determining which device
          computations should be performed on.
        masking_frac: The fraction of each input cell that should be masked.
          This value should be set to zero except for when training the
          unsupervised contrastive learning mode.
        num_features: The number of inputted genes (input dimension).
        attention_module: The attention module of scANNA.
        attention_dim: The dimension of the attention layer (the same as the
          input feature unless an encoding is applied at the start).
        proj_block1: The first projection block of scANNA.
        proj_block2: The second and last projection block of scANNA.
        pwff: The pointwise Feedforward neural network that is used as the
          activation of of the projection blocks.
        task_module: The task module of scANNA. In this implementation, the task
          module is defined to be the number of cell types.
    """

    def __init__(self,
                 input_dimension: int = 5000,
                 task_module_output_dimension: int = 11,
                 dropout: float = 0.0,
                 number_of_projections: int = 8,
                 masking_fraction: float = None,
                 device: _DEVICE_TYPES = 'cpu'):
        """Initializer of the ProjectionAttention class."""

        super().__init__()
        self.device = device
        self.num_features = input_dimension
        self.out_dim = task_module_output_dimension
        self.masking_frac = masking_fraction
        self.attention_dim = input_dimension

        # We use masking for self-supervised contrastive learning only.
        if masking_fraction not in (0.0, None):
            self.masking_layer = torch.nn.Dropout(p=masking_fraction)
        else:
            self.masking_layer = torch.nn.Identity()

        # scANNA Components are as follows:
        # Component (1): Attention Module
        # Component (2): Projection Blocks and Poinstwise Feedforward NN
        # Component (3): Task Module.
        # We number these modules in the declarations below for readibility.

        # Component (1)
        self.attention_module = torch.nn.Linear(self.num_features,
                                                self.num_features)

        # Component (2)
        self.projection_block1 = self._projection_block(
            attention_dimension=self.attention_dim,
            number_of_projection_branches=number_of_projections,
            dropout_probability=dropout)
        self.projection_block2 = self._projection_block(
            attention_dimension=self.attention_dim,
            number_of_projection_branches=number_of_projections,
            dropout_probability=dropout)
        # Component (2)
        self.pwff = self._pointwise_activation(
            input_dimension=self.attention_dim,
            hidden_dimensions=[128, self.attention_dim],
            use_1x1_conv=False)

        # Component (3)
        self.task_module = torch.nn.Sequential(
            torch.nn.Linear(self.attention_dim, self.out_dim),
            torch.nn.LeakyReLU())

    def forward(self,
                input_tensor: torch.Tensor,
                training: bool = True,
                device='cpu'):
        """Forward pass of the projection-based scANNA model.

        Args:
            input_tensor: A tensor containing input data for training.
            training: The mode that we are calling the forward function. True
              indicates that we are training the model

        Returns:
            The forward method returns three tensors:
            (1) logits: the actual logits for predictions.

            (2) alphas: The attention tensor containing the weights for all
                genes.

            (3) gamma: A tensor containing the gene scores.

        Raises:
            None.
        """
        self.training = training

        if not self.training:
            self.device = device

        x_masked = self._parallel_eval(self.masking_layer,input_tensor)
        alphas = self._softmax(self.attention_module(input_tensor))
        gamma = self._gene_scores(alphas, x_masked)

        # The abbrevation "gse" stands for gene stacked event (gse), which is
        # the output of the a projection block (with all branches).
        gse = self._parallel_eval(self.projection_block1, gamma)
        x_activated = self._parallel_eval(self.pwff, gse)

        gse2 = self._parallel_eval(self.projection_block2, x_activated + gamma)
        x_activated2 = self._parallel_eval(self.pwff, gse2)

        logits = self._parallel_eval(self.task_module, x_activated2 + gamma)

        return logits, alphas, gamma
