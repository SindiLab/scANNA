"""Utilty function for counting model parameters."""
from prettytable import PrettyTable
import torch


def count_parameters(model: torch.nn) -> int:
    """Count the total number of parameters in a torch model.

    This utility function will provide a total number of parameters in a model,
    which does not provide per module parameter information.

    Args:
        model: The torch neural network that we want the parameter counts for.

    Returns:
        An integer representing the number of *trainable* parameters in a model.

    Raises:
        None.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def detailed_count_parameters(model: torch.nn) -> int:
    """Count number of parameters in a model and printed in a prettytable.

    This utility function gets a count of trainable parameters in a torch model,
    including per module parameter counts.

    Args:
        model: The torch nerual network that the function will count its
          parameters.

    Returns:
        An integer representing the number of *trainable* parameters in a model.
        Note that this utility function will also preint a prettytable object
        containing the detailed number of parameters in each module.

    Raises:
        None.
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
