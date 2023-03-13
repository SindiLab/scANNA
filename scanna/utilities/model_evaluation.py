"""Utility function for evaluating classification performance."""
import numpy as np
import scanpy
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report as class_rep
from .sparsity_handling import sparse_to_dense
import torch


def evaluate_classifier(valid_data_loader: torch.utils.data.DataLoader,
                        classification_model: torch.nn,
                        classification_report: bool = False,
                        device: str = None,
                        use_gpu: bool = True):
    """Function for evaluating peformance on validation/test data split.

    Args:
        valid_data_loader: A dataloader of the validation or test dataset.
        classification_model: The model which we want to use for validation.
        classification_report: Boolean determining whether user requires
          classification report or not.
        device: The device ('cuda' or 'cpu') on which model evaluation should be
          performed on.
        use_gpu: A boolean indicating if we should use GPU devices when they are
          available.

    Returns
    -------
        None

    """
    if device is None and use_gpu is True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    classification_model = classification_model.to(device)
    print("==> Evaluating on Validation Set:")
    total = 0
    correct = 0
    # for sklearn metrics
    y_true = np.array([])
    y_pred = np.array([])
    with torch.no_grad():
        for _, data in enumerate(valid_data_loader):
            features, labels = data
            labels = labels.to(device)
            outputs, _, _ = classification_model(features.float().to(device),
                                                 training=False)
            _, predicted = torch.max(outputs.squeeze(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true = np.append(y_true, labels.detach().cpu().numpy())
            y_pred = np.append(y_pred, predicted.detach().cpu().numpy())

    print("    -> Accuracy of classifier network on validation set:"
          f"{(100 * correct / total):4.4f} %")
    # calculating the precision/recall based multi-label F1 score
    macro_score = f1_score(y_true, y_pred, average="macro")
    w_score = f1_score(y_true, y_pred, average="weighted")
    traditional_accuracy = accuracy_score(y_true, y_pred)
    print(f"    -> Non-Weighted F1 Score on validation set: {macro_score:4.4f}")
    print(f"    -> Weighted F1 Score on validation set: {w_score:4.4f}")
    if classification_report:
        print(class_rep(y_true, y_pred))
    return y_true, y_pred, macro_score, w_score, traditional_accuracy


def transfer_cell_types(scanpy_data: scanpy.AnnData,
                        classification_model: torch.nn,
                        inplace: bool = True,
                        device: str = None,
                        use_gpu: bool = True):
    """
    Evaluating the performance of the network on validation/test dataset

    Args:
        valid_data_loader: A dataloader of the validation or test dataset.
        classification_model: The model which we want to use for validation.
        inplace: If we want to make the modifications directly to the passed on
          scanpy object.
        device: The device ('cuda' or 'cpu') on which model evaluation should be
          performed on.
        use_gpu: A boolean indicating if we should use GPU devices when they are
          available.

    Returns
    -------
        None

    """
    if device is None and use_gpu is True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    classification_model = classification_model.to(device)
    np_data = sparse_to_dense(scanpy_data)
    print("==> Making predictions:")
    with torch.no_grad():
        features = torch.from_numpy(np_data)
        outputs, _, _ = classification_model(features.float().to(device),
                                             training=False)
        _, predicted = torch.max(outputs.squeeze(), 1)

    scanna_labels = predicted.detach().cpu().numpy()
    if inplace:
        scanpy_data.obs["scANNA_Labels"] = scanna_labels
        scanpy_data.obs["scANNA_Labels"] = scanpy_data.obs[
            "scANNA_Labels"].astype("category")
    else:
        scanpy_data_copy = scanpy_data.copy()
        scanpy_data_copy.obs["scANA_Labels"] = scanna_labels
        return scanpy_data_copy

    print(">-< Done")
