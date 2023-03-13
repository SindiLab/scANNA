"""Main functions for handling h5ad objects for training and testing NNs."""

import torch
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader
from ..utilities.sparsity_handling import sparse_to_dense

def scanpy_to_dataloader(file_path: str = None,
                         scanpy_object: sc.AnnData = None,
                         train_only: bool = False,
                         test_only: bool = False,
                         batch_size: int = 128,
                         workers: int = 12,
                         log_transform: bool = False,
                         log_base: int = None,
                         log_method: str = "scanpy",
                         annotation_key:str = "cluster",
                         # For compatibility reasons we keep the `test_no_valid`
                         # argumnet. This will be removed in future relases.
                         test_no_valid:bool = None,
                         verbose:bool = True,
                         raw_x: bool = True):

    """Function to read in (or use an existing) H5AD files to make dataloaders.

    Args:
        file_path: A string that is the path to the .h5ad file.
        scanpy_object: An existing scanpy object that should be used for the
          dataloaders.
        train_only: Whether we want a dataloader consisting of only training
          data.
        test_only: Whether we want a dataloader consisting of only testing
          samples.
        batch_size: An integer indicating the batch size to be used for the
          Pytorch dataloader.
        workers: Number of workers to load/lazy load in data.
        log_transform: Whether we want to take log transorm of the data or not.
        log_base: The log base we want to use. If None, we will use natural log.
        log_method: If we want to take the log using scanpy or PyTorch.
        annotation_key: A string containing the annotation key which will be
          used as the label for the dataloader.
        verbose: Verbosity option indicated as a boolean.
        raw_x:bool: This is a dataset- and platform-dependant variable. This
          option enables using the "raw" X matrix, as defined in Seurat. Useful
          for when preprocessing in R and running N-ACT in PyTorch.

    Returns:
        This function will return two dataloaders:

        (1) A Training data loader consisting of the data (at batch[0]) and
        labels (at batch[1]).

        (2) A Testing data loader consisting of the data (at batch[0]) and
        labels (at batch[1])

    Raises:
        ValueError: If both "train_only" and "test_only" arguments are set to
          true.
        ValueError: If neither a path to an h5ad file or an existing scanpy
          object is provided.
        ValueError: If log base falls outside of the implemented ones.

    """
    if train_only is True and test_only is True:
        raise ValueError("Both options for 'train_only' and 'test_only' are "
                         "passed as True, which cannot be. Please check the "
                         "args and try again.")
    if scanpy_object is None and file_path is not None:
        print("==> Reading in Scanpy/Seurat AnnData")
        adata = sc.read(file_path)
    elif scanpy_object is not None and file_path is None:
        adata = scanpy_object
    else:
        raise ValueError("Pleaes either provide a path to a h5ad file, or"
                         " provide an existing scanpy object.")

    if raw_x:
        print("    -> Trying adata.raw.X instead of adata.X!")
        try:
            adata.X = adata.raw.X
        except Exception as e:
            print(f"    -> Failed with message: {e}")
            print("    -> Reverting to adata.X if possible")

    if log_transform and log_method == "scanpy":
        print("    -> Doing log(x+1) transformation with Scanpy")
        sc.pp.log1p(adata, base=log_base)

    print("    -> Splitting Train and Test Data")

    if train_only:
        train_adata = adata[adata.obs["split"].isin(["train"])]
        test_adata = None
    elif test_only:
        train_adata = None
        test_adata = adata[adata.obs["split"].isin(["test"])]
    else:
        train_adata = adata[adata.obs["split"].isin(["train"])]
        test_adata = adata[adata.obs["split"].isin(["test"])]

    # turn the cluster numbers into labels
    print(f"==> Using {annotation_key} to generating train and testing labels")
    y_train = None
    y_test = None

    if train_only:
        y_train = [int(x) for x in train_adata.obs[annotation_key].to_list()]
    elif test_only:
        y_test = [int(x) for x in test_adata.obs[annotation_key].to_list()]
    else:
        y_train = [int(x) for x in train_adata.obs[annotation_key].to_list()]
        y_test = [int(x) for x in test_adata.obs[annotation_key].to_list()]

    print("==> Checking if we have sparse matrix into dense")
    norm_count_train = None
    norm_count_test = None
    train_data = None
    test_data = None

    if train_only:
        norm_count_train = sparse_to_dense(train_adata)
        train_data = torch.torch.from_numpy(norm_count_train)
    elif test_only:
        norm_count_test = sparse_to_dense(test_adata)
        test_data = torch.torch.from_numpy(norm_count_test)
    else:
        norm_count_train = sparse_to_dense(train_adata)
        norm_count_test = sparse_to_dense(test_adata)
        train_data = torch.torch.from_numpy(norm_count_train)
        test_data = torch.torch.from_numpy(norm_count_test)

    if log_transform and log_method == "torch":
        print("    -> Doing log(x+1) transformation with torch")
        if log_base is None:
            train_data = torch.log(1 + train_data)
            if not train_only:
                test_data = torch.log(1 + test_data)
        elif log_base == 2:
            train_data = torch.log2(1 + train_data)
            if not train_only:
                test_data = torch.log2(1 + test_data)
        elif log_base == 10:
            train_data = torch.log2(1 + train_data)
            if not train_only:
                test_data = torch.log2(1 + test_data)
        else:
            raise ValueError(
                "    -> We have only implemented log base e, 2 and 10 for torch"
            )

    training_data_and_labels = []
    testing_data_and_labels = []
    if not test_only:
        for i in range(len(train_data)):
            training_data_and_labels.append([norm_count_train[i], y_train[i]])

    if not train_only:
        for i in range(len(test_data)):
            testing_data_and_labels.append([norm_count_test[i], y_test[i]])

    if verbose:
        if not test_only:
            print("==> sample of the training data:")
            print(f"{train_data}")

        if not train_only:
            print("==> sample of the test data:")
            print(f"{test_data}")

    train_data_loader = None
    test_data_loader = None
    if not test_only:
        train_data_loader = DataLoader(training_data_and_labels,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       sampler=None,
                                       batch_sampler=None,
                                       num_workers=workers,
                                       collate_fn=None,
                                       pin_memory=True)

    if not train_only:
        test_data_loader = DataLoader(testing_data_and_labels,
                                       batch_size=len(test_data),
                                       shuffle=True,
                                       sampler=None,
                                       batch_sampler=None,
                                       num_workers=workers,
                                       collate_fn=None,
                                       pin_memory=True)


    return train_data_loader, test_data_loader
