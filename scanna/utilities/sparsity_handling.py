"""Utility function for handling sparse count matrices."""

import numpy as np
import scanpy
from scipy import sparse

def sparse_to_dense(annotated_data: scanpy.AnnData):
    """Utility function to return a dense count matrix.

    Args:
        annotated_data: The scanpy AnnData object with count matrix at "data.X".

    Returns:
        A numpy array of the dense matrix, either after conversion from a sparse
        matrix or the count itself if it was dense.

    Raises:
        None.

    """
    if sparse.issparse(annotated_data.X):
        return np.asarray(annotated_data.X.todense())
    else:
        return np.asarray(annotated_data.X)
