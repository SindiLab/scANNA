""" Base class for projection-based scANNA models."""

from ._base_model import BaseNeuralAttentionModel
from ..projection_block._projector_block import Projection
from ..projection_block._pointwise_feedforward_net import PointWiseFeedForward
from typing import List


class BaseProjectionModel(BaseNeuralAttentionModel):
    """ Base model for the projection-based scANNA model.

    Attributes:
        None.
    """

    def _pointwise_activation(self,
                              input_dimension: int,
                              hidden_dimensions: List[int],
                              use_1x1_conv: bool = False):
        """ Pointwise activation method for the projection class.

        Args:
            input_dimension: An integer determining the input dimension of the
              pointwise feedforward neural network (PWFF).
            hidden_dimensions: The list of hidden dimensions of the PWFF.
            use_1x1_conv: A boolean to determine whether we should use 1x1
              convolutions instead of feedforward layers. This is only for
              computational considerations and both methods should yeild the
              same results.

        Returns:
            An initialized instance of the PointWiseFeedForward class.

        Raises:
            None.
        """

        return PointWiseFeedForward(inp_dim=input_dimension,
                                    hidden_dims=hidden_dimensions,
                                    use_1x1_conv=use_1x1_conv)

    def _projection_block(self,
                          attention_dimension: int,
                          number_of_projection_branches: int = 8,
                          dropout_probability: float = 0.1):
        """ Class method for the proposed projection blocks.

        Args:
            attention_dimension: An integer indicating the input dimension of
              the projection block, which should be the same as the attention
              dimension.
            number_of_projection_branches: An integer determining the number of
              branching we want to do in the calculations (similar to attention-
              heads in transformers).
            dropout_probability: A float indicating the dropout probability in
            the model.

        Returns:
            An initialized instance of the Projection class.

        Raises:
            None.
        """

        return Projection(model_dim=attention_dimension,
                          number_of_branches=number_of_projection_branches,
                          dropout=dropout_probability)
