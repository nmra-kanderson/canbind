import numpy as np


def compute(matrix):
    """
    Compute GBC values from a correlation matrix of `x`.

    Parameters
    ----------
    x : (N,N) np.ndarray

    Returns
    -------
    (N,1) np.ndarray

    """
    n = matrix.shape[0]
    masked = np.ma.masked_array(data=matrix, mask=np.eye(n).astype(bool))
    matrix_z = np.ma.arctanh(masked)
    return matrix_z.mean(axis=0).data
