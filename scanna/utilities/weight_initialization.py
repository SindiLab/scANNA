"""Utility functions initializing neural network weights."""
import torch


def init_weights_xavier_uniform(model: torch.nn):
    """Initializing the weights of a model with Xavier uniform distribution

    Args:
        model: The pytorch model that will be initilized with Xavier weights.

    Returns:
        None. All changes will be inplace.

    Raises:
        None.
    """
    # We want to initialize the linear layers.
    if isinstance(model, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(0.01)


def init_weights_xavier_normal(model: torch.nn):
    """Initializing the weights of a model with Xavier normal distribution

    Args:
        model: The pytorch model that will be initilized with Xavier weights.

    Returns:
        None. All changes will be inplace.

    Raises:
        None.
    """
    # We want to initialize the linear layers.
    if isinstance(model, torch.nn.Linear):
        torch.nn.init.xavier_normal_(model.weight)
        model.bias.data.fill_(0.01)
