"""Utility functions for loading and saving neural network models."""
import os
import torch


def load_model(model: torch.nn, pretrained_path: str):
    """Function for loading a pre-trained model from a path.

    Args:
        model: The pytorch model which will be updated with the pre-trained
          weights from the path.
        pretrained_path: The path to where the pre-trained models is saved.

    Returns:
        The function will return the updated model that was initially passed on,
        and an integer indicating the number of the epochs that the pre-trained
        model was trained.

    Raises:
        None.
    """
    weights = torch.load(pretrained_path)
    try:
        trained_epoch = weights["epoch"]
    except Exception as error:
        print("Epoch information was not found in the pre-trained model."
              f"Error: {error}.")
        # 50 epochs is the recommended number of epochs for training scANNA.
        # When we don't have the epoch information, we use 50 to set the epoch
        # information needed for various functionalities of scANNA.
        print("Setting number of trained epochs to default (50)")
        trained_epoch = 50
    pretrained_dict = weights["Saved_Model"].state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if k in model_dict
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model, trained_epoch


def save_checkpoint_classifier(model: torch.nn,
                               epoch: int,
                               iteration: int,
                               prefix: str = "",
                               dir_path: str = None):
    """Function for saving pre-trained model for inference.

    Args:
        model: The neural network we want to save.
        epoch: The number of epochs that the model has been trained up to (which
          will be used in the filename).
        iteration: Current iteration count (will be used in the filename)
        prefix: Any prefix that should be added to the filename.
        dir_path: The path where the model should be saved to (optional).

    Returns:
        None.

    Raises:
        None.
    """

    if dir_path is None:
        dir_path = "./scANNA-Weights/"

    model_out_path = dir_path + prefix + (f"model_epoch_{epoch}"
                                          f"_iter_{iteration}.pth")
    state = {"epoch": epoch, "Saved_Model": model}
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    torch.save(state, model_out_path)
    print(f"==> Classifier checkpoint was saved to {model_out_path}")


def save_best_classifier(model, prefix="", dir_path=None):
    """Function for saving the best model.

    The best model must be determined in the main script based on some defined
    metric.

    Args:
        model: The best neural network iteration that should be saved.
        prefix: Any prefix that should be added to the filename.
        dir_path: The path where the model should be saved to (optional).

    Returns:
        None.

    Raises:
        None.
    """

    if not dir_path:
        dir_path = "./BestModelWeights/"

    model_out_path = dir_path + prefix + "model_Best.pth"
    state = {"Saved_Model": model}
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    torch.save(state, model_out_path)
    print(f"==> The best model weights were saved to {model_out_path}")
