import os
import numpy as np
from numpy.typing import NDArray

def truncate_svd(matrix: NDArray[np.float64], singular_values_kept: int) -> NDArray[np.float64]:
    """
    Truncate the singular value decomposition (SVD) of a matrix.

    This function takes a matrix, computes its singular value decomposition (SVD),
    and truncates all singular values beyond a specified number.

    Parameters:
    -----------
    matrix : NDArray[np.float64]
        The input matrix to decompose.
    singular_values_kept : int
        The number of singular values to keep.

    Returns:
    --------
    NDArray[np.float64]
        The matrix reconstructed from the truncated SVD.
    """
    U, S, V = np.linalg.svd(matrix, full_matrices=False)
    S_truncated = np.zeros_like(S)
    S_truncated[:singular_values_kept] = S[:singular_values_kept]
    return np.dot(np.dot(U, np.diag(S_truncated)), V)


def pp_matrix(matrix):
    # pretty print a matrix using sympy
    import sympy
    return sympy.latex(sympy.Matrix(matrix))


def ensure_unique_filename(path):
    """Generate a unique file path by appending a counter to the filename if it already exists."""
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path