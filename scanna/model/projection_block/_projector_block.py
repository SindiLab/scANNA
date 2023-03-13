"""Implementation of projection modules for scANNA's."""

import torch


class Projection(torch.nn.Module):
    """Implementation of the projection blocks.

    Projection blocks are a modification of the multi-head attention modules
    introduced by Vaswani et al. (https://arxiv.org/pdf/1706.03762.pdf).
    scANNA's projection blocks are discussed in the following paper:
    https://arxiv.org/pdf/2206.04047.pdf.

    Attributes:
        model_dim: The dimension of the projection module.
        number_of_projections: The number of parallel linear projections
          (or branches).
        projection_dims: Dimension of each projection branch.
        projection: The linear projection module.
        output_dropout: The dropout module of the projections.
        normalization: The layernorm normalization method.
    """

    def __init__(self,
                 model_dim: int = 5000,
                 number_of_branches: int = 8,
                 dropout: float = 0.0):
        """Initializer of the Projection class."""

        super().__init__()
        # Since we implement all projections in one tensor (for efficency), we
        # need to make sure that the model dimension is divisible by the number
        # of projection branches.
        if model_dim % number_of_branches != 0:
            raise ValueError("Dimension of the attention module modulo"
                             "number of projection branches should == 0. Input"
                             f"module is {model_dim} % {number_of_branches} ="
                             f"{model_dim % number_of_branches} != 0")

        self.model_dim = model_dim
        self.number_of_projections = number_of_branches
        self.projection_dims = int(self.model_dim / self.number_of_projections)
        # Linear projections done in one tensor (similar to Vaswani et al.)
        # for efficiency.
        self.projection = torch.nn.Linear(
            self.model_dim,
            self.projection_dims * self.number_of_projections)

        self.output_dropout = torch.nn.Dropout(p=dropout)
        self.normalization = torch.nn.LayerNorm(self.model_dim)

    def forward(self, x_context: torch.Tensor) -> torch.Tensor:
        """ Forward pass for computing the multi-branching projections.

        Args:
            x_context: The input tensor (which should be the gene score after
              the attention module).

        Returns:
            A tensor containing the gene scores (after residual + layer norm).

        Raises:
            None.

        """
        batch_size = x_context.size(0)

        # Performing linear projections.
        x_proj = self.projection(x_context)
        x_all_heads = x_proj.view(batch_size, -1, self.number_of_projections,
                                  self.projection_dims)
        x = x_all_heads.permute(2, 0, 1, 3).contiguous().view(
            batch_size * self.number_of_projections, -1, self.projection_dims)
        # Restore input shapes.
        x_inp_shape = torch.cat(torch.chunk(x,
                                            self.number_of_projections,
                                            dim=0),
                                dim=2)
        x = x_inp_shape.squeeze()
        # Applying a residual connection before normalization.
        x += x_context

        return self.normalization(x)
