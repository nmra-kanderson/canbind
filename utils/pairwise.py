import numpy as np

__all__ = ['pearsonr_multi', 'pairwise_pearson']


def pearsonr_multi(x, y):
    """
    Vectorized & very fast multi-dimensional Pearson correlations.

    Parameters
    ----------
    x : (N,P) np.ndarray
    y : (M,P) np.ndarray

    Returns
    -------
    (N,M) np.ndarray

    Notes
    -----
    Implements the mathematical logic for pearson correlation in a way that
    leverages vectorization of numpy arrays.

    Raises
    ------
    TypeError: inputs arent numpy arrays
    ValueError: shapes are incompatible or dimensions exceed 2

    """

    # check dtypes
    if type(x) != np.ndarray != type(y):
        raise TypeError('inputs must be numpy arrays')

    # check x shape
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim > 2:
        raise ValueError('x must be one or two dimensional')

    # check y shape
    if y.ndim == 1:
        y = y.reshape(1, -1)
    elif y.ndim > 2:
        raise ValueError('y must be one or two dimensional')

    # check second dimension of input for compatibility
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must be same size in 2nd dimension.')

    # do math
    mu_x = x.mean(axis=1)
    mu_y = y.mean(axis=1)
    s_x = x.std(axis=1, ddof=n - 1)
    s_y = y.std(axis=1, ddof=n - 1)
    cov = np.dot(x, y.T) - n * np.dot(
        mu_x[:, np.newaxis], mu_y[np.newaxis, :])

    with np.errstate(divide='ignore', invalid='ignore'):
        # ignore occasional "RuntimeWarning: invalid value encountered in true_divide" messages
        # https://stackoverflow.com/a/23116937/554481
        return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def pairwise_pearson(x):
    """
    Compute pairwise pearson correlation matrix between rows of `x`.

    Parameters
    ----------
    x : (N,P) np.ndarray

    Returns
    -------
    (N,N) np.ndarray

    """
    return pearsonr_multi(x, x)
