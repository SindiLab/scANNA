"""Base class for scANNA."""
from __future__ import annotations

from ..sparsemax.sparsemax import SparseMax
import torch
from torch.nn.parallel import data_parallel

class BaseNeuralAttentionModel(torch.nn.Module):
    """ Base model for any scANNA (additive attention or projection).

    Original additive attention is from Bahdanau et al, located at
    https://arxiv.org/pdf/1409.0473.pdf.

    Attributes:
        device: A string (either "cuda" or "cpu")

    """
    def __init__(self):
        """Initializer of the base method."""
        super().__init__()
        self.device = "cpu"

    def _parallel_eval(
        self,
        method: BaseNeuralAttentionModel,
        *args,
    ):
        """ Universal evaluator based on the chosen device.

        Args:
            method: Class method for which we want to do evaluations.
            *args: Appropriate arguments relating to the class method passed on.

        Returns:
            The same return as the class method passed on.

        Raises:
            The same rais behaviors as in the class method passed on.

        """

        if self.device == "cpu":
            return method(*args)
        else:
            return data_parallel(method, *args)

    def _softmax(self, e_t: torch.Tensor) -> torch.Tensor:
        """ Softmax method for the alignment score e_t.

        Args:
            e_t: Alignment scores which are the output of the attention layer.

        Returns:
            The computed probability tensor of the attention layer.

        Raises:
            None.
        """
        return torch.nn.Softmax(dim=1)(e_t)

    def _sparsemax(self, e_t: torch.Tensor) -> torch.Tensor:
        """ SparseMax method for the alignment score e_t

        Args:
            e_t: Alignment scores which are the output of the attention layer.

        Returns:
            The computed SparseMax values of the attention layer outputs as a
            tensor.

        Raises:
            None.
        """
        return SparseMax(dim=1)(e_t)

    def _gene_scores(self, alpha_t: torch.Tensor,
                     x_t: torch.Tensor) -> torch.Tensor:
        """ Method for computing gene scores.

        This method computes the gene scores (traditionally referred to as
        "context" in natural language processing) using the attention values and
        the gene expressions.

        Args:
            alpha_t: The attention values (computed probabilities over e_t).
            x_t : Raw gene counts from scRNAseq matrix.

        Returns:
            A tensor containing the gene scores.

        Raises:
            None.
        """
        return torch.mul(alpha_t, x_t)
