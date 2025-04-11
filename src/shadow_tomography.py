import numpy as np
import qutip
from typing import List, Union, Optional, Sequence, Iterable, Literal
import itertools
import math

# Define Type Aliases for clarity
MatrixInput = Union[qutip.Qobj, np.ndarray]
MatrixListInput = Union[MatrixInput, Iterable[MatrixInput]]
BasisType = Union[Literal['pauli', 'flatten'], Iterable[np.ndarray]]
RescalingType = Literal['none', 'trace']
RescalingInput = Union[RescalingType, List[float]]

# Helper function to get Pauli basis for n qubits
def _get_pauli_basis_product(num_qubits: int, normalized: bool = True) -> List[np.ndarray]:
    """
    Generates the tensor product Pauli basis for n qubits.

    The basis elements B_k satisfy Tr(B_j B_k) = d * δ₍ⱼₖ₎, where d=2ⁿ.
    The basis always includes the identity operator.

    Args:
        num_qubits: Number of qubits (must be > 0).
        normalized: If True, normalize each basis element so that Tr(σᵢ²) = 1.

    Returns:
        List[np.ndarray]: A list of dxd NumPy arrays representing the basis elements.
    """
    if num_qubits <= 0:
        raise ValueError("Number of qubits must be positive.")

    d = 2**num_qubits
    single_qubit_paulis = {
        'I': qutip.qeye(2),
        'X': qutip.sigmax(),
        'Y': qutip.sigmay(),
        'Z': qutip.sigmaz()
    }

    # Generate all Pauli strings of length num_qubits.
    pauli_strings = itertools.product(['I', 'X', 'Y', 'Z'], repeat=num_qubits)
    basis_qobj = [qutip.tensor(*(single_qubit_paulis[p] for p in ps)) for ps in pauli_strings]
    basis_np = [op.full() for op in basis_qobj]

    if normalized:
        # Normalize such that each basis element has unit trace square.
        basis_np = [b / np.sqrt(d) for b in basis_np]

    return basis_np


def _preprocess_matrices(matrices_in: MatrixListInput) -> List[np.ndarray]:
    """
    Converts the input to a list of NumPy arrays and checks validity.
    This is just to ensure the matrices are in the correct format and type.

    Args:
        matrices_in: A single matrix (qutip.Qobj or np.ndarray) or a sequence of such matrices.

    Returns:
        List[np.ndarray]: List of matrices as NumPy arrays.
    """
    # Avoid iterating over a NumPy array representing a single matrix.
    if isinstance(matrices_in, (qutip.Qobj, np.ndarray)):
        matrices = [matrices_in]
    elif isinstance(matrices_in, Sequence):
        matrices = list(matrices_in)
    else:
        raise TypeError("Input must be a qutip.Qobj, numpy.ndarray, or a sequence of these.")

    if not matrices:
        raise ValueError("Input matrix list cannot be empty.")

    matrices_np = []
    for mat in matrices:
        if isinstance(mat, qutip.Qobj):
            if not mat.isoper:
                raise TypeError("Input Qobj must be an operator.")
            mat_np = mat.full()
        elif isinstance(mat, np.ndarray):
            mat_np = mat
        else:
            raise TypeError(f"Unsupported matrix type: {type(mat)}. Use qutip.Qobj or numpy.ndarray.")
        matrices_np.append(mat_np)
    return matrices_np

def _unvectorize_matrix(vector: np.ndarray,
                        d: Optional[int] = None,
                        basis: BasisType = 'pauli',
                        pauli_basis: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    """
    Converts a flattened vector back into a dxd matrix using the specified basis.

    Args:
        vector: The flattened matrix as a 1D NumPy array.
        d: The target dimension. If None, it is inferred from the vector length.
        basis: Either 'pauli', 'flatten', or an iterable of basis matrices.
        pauli_basis: Optional precomputed Pauli basis (if basis is 'pauli').

    Returns:
        np.ndarray: The reconstructed dxd matrix.
    """
    # make sure the vector is 1D or equivalent to 1D (ie 2D with one column)
    if vector.ndim != 1:
        if vector.ndim == 2 and vector.shape[1] == 1:
            vector = vector.flatten()
        else:
            raise ValueError(f"Input vector must be 1D. Got an array with {vector.ndim} dimensions.")
    # compute the dimension if not provided, assuming the provided vector comes vectorizing a d*d matrix
    if d is None:
        d = int(math.sqrt(vector.size))
        if d * d != vector.size:
            raise ValueError(f"Vector length {vector.size} is not a perfect square.")

    if basis == 'flatten':
        return vector.reshape((d, d))
    elif basis == 'pauli':
        if d <= 0 or (d & (d - 1)) != 0:
            raise ValueError(f"Dimension {d} must be a power of 2 for Pauli basis.")
        num_qubits = int(math.log2(d))
        if pauli_basis is None:
            pauli_basis = _get_pauli_basis_product(num_qubits, normalized=True)
        if len(pauli_basis) != d * d:
            raise RuntimeError("Mismatch in the number of Pauli basis elements.")
        matrix = np.zeros((d, d), dtype=np.complex128)
        for k in range(d * d):
            matrix += vector[k] * pauli_basis[k]
        return matrix
    # If a list of matrices is provided, use them for unvectorization
    elif isinstance(basis, Iterable):
        basis_list = list(basis)
        if len(basis_list) != d * d:
            raise ValueError(f"Provided basis length {len(basis_list)} does not match d*d = {d*d}.")
        matrix = np.zeros((d, d), dtype=np.complex128)
        for k in range(d * d):
            matrix += vector[k] * basis_list[k]
        return matrix
    else:
        raise ValueError(f"Unknown basis type: {basis}")


# --- Main Functions ---

def vectorize_density_matrix(matrices_in: MatrixListInput,
                             basis: BasisType = 'pauli') -> np.ndarray:
    r"""
    Vectorizes one or more Hermitian matrices using the specified basis.

    If basis == 'pauli', the vectorization is defined as
      $$ v_k = \mathrm{Tr}(M B_k), $$
    where $$B_k$$ are the Pauli basis elements.
    For basis == 'flatten', the matrix is simply flattened row by row.

    Args:
        matrices_in: A single matrix (qutip.Qobj or np.ndarray) or sequence of matrices.
        basis: The vectorization basis: 'pauli', 'flatten', or a custom iterable of matrices.

    Returns:
        np.ndarray: A 2D array of shape (d*d, num_matrices) whose columns are the vectorized matrices.
    """
    matrices_np = _preprocess_matrices(matrices_in)
    d = matrices_np[0].shape[0]  # Assuming all matrices share the same dimension.
    vectorized_mats = []

    if basis == 'pauli':
        num_qubits = math.log2(d)
        if not num_qubits.is_integer() or num_qubits <= 0:
            raise ValueError(f"Pauli basis is only supported for dimensions d = 2^n (n > 0). Got d={d}.")
        pauli_basis = _get_pauli_basis_product(int(num_qubits), normalized=True)
        if len(pauli_basis) != d * d:
            raise RuntimeError("Internal error: Generated Pauli basis size is incorrect.")
        for mat in matrices_np:
            vec = np.array([np.trace(mat @ p).real for p in pauli_basis], dtype=float)
            vectorized_mats.append(vec)
    elif basis == 'flatten':
        for mat in matrices_np:
            vectorized_mats.append(mat.flatten())
    elif isinstance(basis, Iterable):
        basis_list = list(basis)
        if len(basis_list) != d * d:
            raise ValueError(f"Provided basis length {len(basis_list)} does not match d*d = {d*d}.")
        for mat in matrices_np:
            vec = np.array([np.trace(mat @ b) for b in basis_list], dtype=complex)
            vectorized_mats.append(vec)
    else:
        raise ValueError(f"Unknown basis type: {basis}")
    # Stack the vectorized matrices into a 2D array
    # Each column corresponds to a vectorized matrix.
    return np.column_stack(vectorized_mats)


def frame_operator(
    povm: Iterable[MatrixInput],
    basis: BasisType = 'pauli',
    rescaling: RescalingInput = 'none'
) -> np.ndarray:
    r"""
    Computes the frame operator F for a given POVM.

    The frame operator is defined as
      $$ F = \sum_b \alpha_b^2 \, |v_b\rangle\langle v_b|, $$
    where $$ v_b $$ is the vectorized POVM element and $$ \alpha_b $$ are the rescaling factors.
    
    Rescaling options:
      - 'none': $$ \alpha_b = 1 $$ for all b.
      - 'trace': $$ \alpha_b = \mathrm{Tr}(\mu_b)/d $$, where d is the matrix dimension.
      - list: A custom list of rescaling factors (must match the number of POVM elements).

    Args:
        povm: A non-empty iterable of POVM elements (qutip.Qobj or np.ndarray).
        basis: The vectorization basis ('pauli', 'flatten', or custom).
        rescaling: The rescaling strategy.

    Returns:
        np.ndarray: The frame operator F as an array of shape (d*d, d*d).
    """
    povm_list = list(povm)
    if not povm_list:
        raise ValueError("POVM must be a non-empty sequence of matrices.")

    povm_np = _preprocess_matrices(povm_list)
    num_elements = len(povm_np)
    d = povm_np[0].shape[0]
    vec_dim = d * d

    # Determine rescaling factors.
    if isinstance(rescaling, list):
        if len(rescaling) != num_elements:
            raise ValueError(f"Custom rescaling list length ({len(rescaling)}) must match POVM length ({num_elements}).")
        alpha_b = np.array(rescaling, dtype=float)
    elif rescaling == 'trace':
        traces = np.array([np.trace(mu) for mu in povm_np])
        alpha_b = np.real(traces) / d
    elif rescaling == 'none':
        alpha_b = np.ones(num_elements, dtype=float)
    else:
        raise ValueError(f"Unknown rescaling option: {rescaling}")
    # if using the pauli basis, the frame operator is represented as a real matrix
    if basis == 'pauli':
        frame_op = np.zeros((vec_dim, vec_dim), dtype=float)
    else:
        # For other bases, we use complex numbers    
        frame_op = np.zeros((vec_dim, vec_dim), dtype=complex)
    vectorized_povm = vectorize_density_matrix(povm_np, basis=basis)  # Shape: (d*d, num_elements)

    for b in range(num_elements):
        v_b = vectorized_povm[:, b:b+1]  # Column vector (d*d, 1)
        frame_op += (alpha_b[b] ** 2) * (v_b @ v_b.conj().T)

    return frame_op


def shadow_estimator(
    povm: Iterable[MatrixInput],
    basis: BasisType = 'pauli',
    rescaling: RescalingInput = 'none'
) -> List[np.ndarray]:
    r"""
    Computes the dual POVM (shadow estimator) for a given POVM.

    The dual elements $$E_b$$ are determined so that their vectorized forms satisfy:
      $$ w_b = F^{-1} (\alpha_b^2 \, v_b), $$
    where $$ v_b $$ is the vectorized original POVM element and F is the frame operator.
    
    Args:
        povm: A non-empty sequence of POVM elements (qutip.Qobj or np.ndarray).
        basis: The vectorization basis ('pauli', 'flatten', or a custom iterable).
        rescaling: Rescaling strategy ('none', 'trace', or a list of custom factors).
    
    Returns:
        List[np.ndarray]: A list of the dual POVM elements (as NumPy arrays).
    """
    povm_np = _preprocess_matrices(povm)
    if not povm_np:
        raise ValueError("POVM must be a non-empty sequence of matrices.")

    d = povm_np[0].shape[0]
    num_elements = len(povm_np)

    # 1. Compute the frame operator and its pseudo-inverse.
    frame_op = frame_operator(povm_np, basis=basis, rescaling=rescaling)
    try:
        frame_op_inv = np.linalg.pinv(frame_op)
    except np.linalg.LinAlgError:
        raise RuntimeError("Failed to compute the inverse of the frame operator.")

    # 2. Determine rescaling factors.
    if isinstance(rescaling, list):
        if len(rescaling) != num_elements:
            raise ValueError(f"Custom rescaling list length ({len(rescaling)}) must match POVM length ({num_elements}).")
        alpha_b = np.array(rescaling, dtype=float)
    elif rescaling == 'trace':
        traces = np.array([np.trace(mu) for mu in povm_np])
        alpha_b = np.real(traces) / d
    elif rescaling == 'none':
        alpha_b = np.ones(num_elements, dtype=float)
    else:
        raise ValueError(f"Unknown rescaling option: {rescaling}")

    # 3. Vectorize the original POVM elements.
    vectorized_povm = vectorize_density_matrix(povm_np, basis=basis)

    # 4. Compute the vectorized dual POVM elements.
    vectorized_dual = np.zeros_like(vectorized_povm, dtype=complex)
    for b in range(num_elements):
        v_b = vectorized_povm[:, b]
        scaled_v_b = (alpha_b[b] ** 2) * v_b
        w_b = frame_op_inv @ scaled_v_b
        vectorized_dual[:, b] = w_b

    # 5. Prepare the Pauli basis (if needed) for unvectorization.
    precomputed_pauli: Optional[Sequence[np.ndarray]] = None
    if basis == 'pauli':
        num_qubits = math.log2(d)
        if not num_qubits.is_integer() or num_qubits <= 0:
            raise ValueError(f"Pauli basis requires d = 2^n with n > 0. Got d = {d}.")
        precomputed_pauli = _get_pauli_basis_product(int(num_qubits), normalized=True)

    # 6. Unvectorize the dual POVM vectors back into matrix form.
    dual_matrices = []
    for b in range(num_elements):
        w_b = vectorized_dual[:, b]
        E_b = _unvectorize_matrix(w_b, d=d, basis=basis, pauli_basis=precomputed_pauli)
        dual_matrices.append(E_b)

    return dual_matrices