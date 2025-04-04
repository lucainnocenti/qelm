import numpy as np
import qutip
from numpy.typing import NDArray
from typing import List

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
    return np.dot(U, np.diag(S_truncated), V)


def vectorize_density_matrix(rho: NDArray[np.complex128], basis: str = 'paulis') -> NDArray[np.complex128]:
    """
    Vectorize a density matrix.

    This function takes a density matrix and vectorizes it in some specified basis.

    Parameters:
    -----------
    rho : NDArray[np.complex128]
        The density matrix to vectorize.
    basis : str, optional
        The basis in which to vectorize the matrix (default is 'paulis').

    Returns:
    --------
    NDArray[np.complex128]
        The vectorized density matrix.
    """
    if basis == 'flatten':
        return rho.flatten()
    elif basis == 'paulis':
        single_qubit_ops = [
            np.eye(2, dtype=np.complex128),
            qutip.sigmax().full(),
            qutip.sigmay().full(),
            qutip.sigmaz().full()
        ]
        if rho.shape[0] == 2:
            op_basis = single_qubit_ops
        elif rho.shape[0] == 4:
            op_basis = [np.kron(op1, op2) for op1 in single_qubit_ops for op2 in single_qubit_ops]
        else:
            raise ValueError('Only 1 and 2 qubits for now.')
        return np.array([np.trace(rho @ op) for op in op_basis])
    else:
        raise ValueError("Unsupported basis provided.")


def devectorize_density_matrix(rho_vec: NDArray[np.complex128], basis: str = 'paulis') -> NDArray[np.complex128]:
    """
    Devectorize a density matrix.

    This function takes a vectorized density matrix and devectorizes it.
    It is intended to be the inverse operation of vectorize_density_matrix.

    Parameters:
    -----------
    rho_vec : NDArray[np.complex128]
        The vectorized density matrix.
    basis : str, optional
        The basis in which the matrix was vectorized (default is 'paulis').

    Returns:
    --------
    NDArray[np.complex128]
        The density matrix.
    """
    d = int(np.sqrt(len(rho_vec)))  # dimension of the density matrix
    if basis == 'flatten':
        return rho_vec.reshape((d, d))
    elif basis == 'paulis':
        single_qubit_ops: List[NDArray[np.complex128]] = [
            np.eye(2, dtype=np.complex128),
            qutip.sigmax().full(),
            qutip.sigmay().full(),
            qutip.sigmaz().full()
        ]
        single_qubit_ops = [op / 2 for op in single_qubit_ops]  # proper inversion normalization
        if d == 2:
            op_basis = single_qubit_ops
        elif d == 4:
            op_basis = [np.kron(op1, op2) for op1 in single_qubit_ops for op2 in single_qubit_ops]
        else:
            raise ValueError('Only 1 and 2 qubits for now.')
        return np.sum([rho_vec[i] * op for i, op in enumerate(op_basis)])
    return rho_vec.reshape((d, d))


def ket_to_dm(ket: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Convert a ket (state vector) to a density matrix.

    Parameters:
    -----------
    ket : NDArray[np.complex128]
        The state vector, expected to be a one-dimensional or column vector.

    Returns:
    --------
    NDArray[np.complex128]
        The corresponding density matrix.
    """
    ket = np.asarray(ket, dtype=np.complex128).reshape(-1, 1)
    return ket @ ket.conj().T


def kets_to_vectorized_states_matrix(list_of_kets: List[NDArray[np.complex128]], basis: str = 'paulis') -> NDArray[np.complex128]:
    """
    Convert a list of kets to a vectorized states matrix.

    This function takes a list of kets and converts them to a matrix of vectorized states.
    The result is a matrix of dimensions (dim^2, num_states), where dim is the dimension of the density matrix.

    Parameters:
    -----------
    list_of_kets : List[NDArray[np.complex128]]
        The list of kets to convert.
    basis : str, optional
        The basis in which to vectorize the states (default is 'paulis').

    Returns:
    --------
    NDArray[np.complex128]
        The matrix of vectorized states.
    """
    return np.array(
        [vectorize_density_matrix(ket_to_dm(ket), basis=basis) for ket in list_of_kets]
    ).T

def pp_matrix(matrix):
    # pretty print a matrix using sympy
    import sympy
    return sympy.latex(sympy.Matrix(matrix))