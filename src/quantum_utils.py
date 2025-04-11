import itertools

import qutip
import numpy as np

from numpy.typing import NDArray
import typing
from typing import List, Union, Iterable, Optional, Literal

from src.POVM import POVM

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
    elif isinstance(matrices_in, Iterable):
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

def unvectorize_matrix(vector: np.ndarray,
                        d: Optional[int] = None,
                        basis: BasisType = 'pauli',
                        pauli_basis: Optional[Iterable[np.ndarray]] = None) -> np.ndarray:
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
        d = int(np.sqrt(vector.size))
        if d * d != vector.size:
            raise ValueError(f"Vector length {vector.size} is not a perfect square.")

    if basis == 'flatten':
        return vector.reshape((d, d))
    elif basis == 'pauli':
        if d <= 0 or (d & (d - 1)) != 0:
            raise ValueError(f"Dimension {d} must be a power of 2 for Pauli basis.")
        num_qubits = int(np.log2(d))
        if pauli_basis is None:
            pauli_basis = _get_pauli_basis_product(num_qubits, normalized=True)
        else:
            pauli_basis = list(pauli_basis)
        
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
        num_qubits = np.log2(d)
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


def ket2dm(ket: NDArray[np.complex128]) -> NDArray[np.complex128]:
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


def kets_to_vectorized_states_matrix(list_of_kets: List[NDArray[np.complex128]], basis: BasisType = 'pauli') -> NDArray[np.complex128]:
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
        [vectorize_density_matrix(ket2dm(ket), basis=basis) for ket in list_of_kets]
    ).T


def measure_povm(
        state: Union[qutip.Qobj, Iterable[qutip.Qobj]],
        povm: Union[Iterable[qutip.Qobj], POVM],
        statistics: float,
        return_frequencies: bool = False
    ) -> NDArray:
    """
    Measure a POVM on a quantum state.
    Returns the measurement results as a list of integers, each integer representing one outcome of the POVM.

    Parameters:
    -----------
    state : qutip.Qobj
        The quantum state to be measured.
    povm : list[qutip.Qobj]
        A list of POVM operators.
        Each operator must be a Hermitian operator.
    statistics : int or float
        The number of measurements to perform.
        The float is to accept the case of infinite statistics.
        NOTE: non-integer non-infty floats will be rounded down to the nearest integer.
    return_frequencies : bool, optional
        If True, return the frequencies instead of the raw outcomes.
        Default is False.
    """
    # make povm a list if it is not already
    if not isinstance(povm, POVM):
        povm = POVM(povm)
    
    if not isinstance(state, qutip.Qobj):
        state = list(state)
        # if a list of states is provided, we measure each one separately and return a matrix as output
        results = [measure_povm(s, povm, statistics, return_frequencies) for s in state]
        results = np.asarray(results)
        return results.T

    # if we're here then state is a single Qobj, not a list
    if state.type != 'oper':
        # convert it to a density matrix
        state = qutip.ket2dm(state)

    # compute the output probabilities
    probabilities = [typing.cast(float, qutip.expect(state, effect)) for effect in povm]

    # Check if statistics is infinity
    if np.isinf(statistics):
        if return_frequencies:
            # In the infinite statistics limit, frequencies equal the probabilities
            return np.array(probabilities)
        else:
            raise ValueError("Cannot return raw outcomes for infinite statistics. Please set return_frequencies=True.")
    # sample from the output probabilities with statistics `statistics`
    sampled_outcomes = np.random.choice(a=len(povm), size=round(statistics), p=probabilities)
    

    if return_frequencies:
        # return the frequencies of each outcome
        frequencies = np.bincount(sampled_outcomes, minlength=len(povm)) / statistics
        return frequencies
    # otherwise, return the sampled outcomes
    return sampled_outcomes

# define a function that returns the single-qubit SIC-POVM
def sic_povm() -> POVM:
    """
    Get the single-qubit SIC-POVM.

    Returns:
    --------
    POVM
        A POVM object containing the SIC-POVM operators with associated label.
    """
    # Define the SIC-POVM operators for a single qubit
    povm = [
        qutip.Qobj([[1, 0], [0, 0]]) / 2,
        qutip.ket2dm(qutip.Qobj([1/np.sqrt(3), np.sqrt(2/3)])) / 2,
        qutip.ket2dm(qutip.Qobj([1/np.sqrt(3), np.sqrt(2/3) * np.exp(2 * np.pi * 1j / 3)])) / 2,
        qutip.ket2dm(qutip.Qobj([1/np.sqrt(3), np.sqrt(2/3) * np.exp(4 * np.pi * 1j / 3)])) / 2
    ]
    return POVM(povm, label="SIC")

# extract the projections over the eigenstates of the three Pauli matrices
def mub_povm():
    """
    Generate a POVM consisting of the projections over the eigenstates of the three Pauli matrices.
    The resulting POVM is a list of 6 rank-1 projectors.
    """
    ops = [
        qutip.ket2dm(qutip.basis(2, 0)), # |0><0|
        qutip.ket2dm(qutip.basis(2, 1)), # |1><1|
        qutip.ket2dm(qutip.basis(2, 0) + qutip.basis(2, 1)) / 2,
        qutip.ket2dm(qutip.basis(2, 0) - qutip.basis(2, 1)) / 2,
        qutip.ket2dm(qutip.basis(2, 0) + 1j * qutip.basis(2, 1)) / 2,
        qutip.ket2dm(qutip.basis(2, 0) - 1j * qutip.basis(2, 1)) / 2
    ]
    normalized_povm = [op / 3 for op in ops]
    return POVM(normalized_povm, label="MUB")

def random_rank1_povm(dim: int, num_outcomes: int, seed: Optional[bool] = None):
    """
    Generate a random rank-1 POVM with d outcomes in a d-dimensional Hilbert space.
    The returned list [E_1, ..., E_d] satisfies sum_i E_i = I_d, 
    and each E_i is a rank-1 projector.
    
    Parameters
    ----------
    d : int
        Dimension of the Hilbert space.
    num_outcomes : int
    seed : int, optional
        Seed for reproducible random generation.
    
    Returns
    -------
    povm : list of ndarray
        A list [E_1, ..., E_d] of d rank-1 projectors,
        each a d x d complex ndarray.
    """

    if seed is not None:
        np.random.seed(seed)
    
    random_unitary = qutip.rand_unitary(num_outcomes).full()[:, :dim]
    
    # Build POVM elements as rank-1 projectors onto the columns of U
    povm = []
    for i in range(num_outcomes):
        col = random_unitary[i]
        E_i = np.outer(col, col.conjugate())
        povm.append(qutip.Qobj(E_i))
    
    return POVM(povm, label="Random rank-1, {} outcomes".format(num_outcomes))

def get_permutation_operator(d: int, k: int, perm: tuple) -> np.ndarray:
    r"""
    Generates the matrix representation of a permutation operator P_sigma
    acting on the k-fold tensor product space (C^d)^(\otimes k).

    Args:
        d: Dimension of the single Hilbert space H.
        k: Number of tensor factors (k copies of H).
        perm: A tuple of length k representing the permutation sigma.
              perm[i] = j means the i-th subsystem is moved to the j-th position.
              Example: For k=3, perm=(1, 2, 0) corresponds to the cyclic
                       permutation sigma: 0->1, 1->2, 2->0 (P_123).
                       perm=(1, 0, 2) corresponds to swapping systems 0 and 1 (P_12).

    Returns:
        A (d^k x d^k) numpy array representing the permutation operator P_sigma.
        P_sigma |i_0>|i_1>...|i_{k-1}> = |i_{sigma^{-1}(0)}> |i_{sigma^{-1}(1)}> ... |i_{sigma^{-1}(k-1)}>
    """
    dim_total = d**k
    P = np.zeros((dim_total, dim_total), dtype=complex)

    # Compute the inverse permutation: if perm maps i to j, inv_perm maps j to i
    inv_perm = [0] * k
    for i, p_i in enumerate(perm):
        inv_perm[p_i] = i

    # Iterate through all standard basis states |i_0>|i_1>...|i_{k-1}>
    # indices_in corresponds to (i_0, i_1, ..., i_{k-1})
    for indices_in in itertools.product(range(d), repeat=k):
        # Calculate the linear index for the input basis state
        # Equivalent to i_0*d^(k-1) + i_1*d^(k-2) + ... + i_{k-1}
        idx_in = np.ravel_multi_index(indices_in, dims=(d,) * k)

        # Determine the output basis state indices after permutation
        # indices_out = (i_{sigma^{-1}(0)}, i_{sigma^{-1}(1)}, ..., i_{sigma^{-1}(k-1)})
        indices_out = tuple(indices_in[inv_perm[j]] for j in range(k))

        # Calculate the linear index for the output basis state
        idx_out = np.ravel_multi_index(indices_out, dims=(d,) * k)

        # Set the corresponding matrix element to 1
        # P[idx_out, idx_in] represents <basis_out | P_sigma | basis_in>
        P[idx_out, idx_in] = 1.0

    return P

# ----------------------------------------
# Expectation value for 1 copy: E[rho]
# ----------------------------------------
def average_rho(d: int) -> np.ndarray:
    """
    Computes the average state E[rho] = I/d for a random pure state rho
    in a d-dimensional Hilbert space.

    Args:
        d: The dimension of the Hilbert space.

    Returns:
        A (d x d) numpy array representing the average state (I/d).
    """
    if d <= 0:
        raise ValueError("Dimension d must be positive.")
    identity = np.eye(d, dtype=complex)
    return identity / d

# ----------------------------------------
# Expectation value for 2 copies: E[rho⊗rho]
# ----------------------------------------
def average_rho_tensor_rho(d: int) -> np.ndarray:
    """
    Computes the average state E[rho⊗rho] = (I⊗I + S_12) / (d(d+1))
    for a random pure state rho in a d-dimensional Hilbert space.

    Args:
        d: The dimension of the single Hilbert space.

    Returns:
        A (d^2 x d^2) numpy array representing the average state E[rho⊗rho].
    """
    if d <= 0:
        raise ValueError("Dimension d must be positive.")

    dim_total = d * d

    # Identity on H⊗H
    identity_total = np.eye(dim_total, dtype=complex)

    # Swap operator S_12 (permutation (1, 0))
    swap_12 = get_permutation_operator(d, 2, (1, 0))

    # Normalization factor
    norm = d * (d + 1)
    if norm == 0: # Handles d=0 case technically, though prevented by check
        return np.zeros((dim_total, dim_total), dtype=complex)

    return (identity_total + swap_12) / norm

# ----------------------------------------
# Expectation value for 3 copies: E[rho⊗rho⊗rho]
# ----------------------------------------
def average_rho_tensor_rho_tensor_rho(d: int) -> np.ndarray:
    """
    Computes the average state E[rho⊗rho⊗rho] = (Sum_{sigma in S3} P_sigma) / (d(d+1)(d+2))
    for a random pure state rho in a d-dimensional Hilbert space.

    Args:
        d: The dimension of the single Hilbert space.

    Returns:
        A (d^3 x d^3) numpy array representing the average state E[rho⊗rho⊗rho].
    """
    if d <= 0:
        raise ValueError("Dimension d must be positive.")

    dim_total = d**3
    k = 3

    # Sum of permutation operators for S3
    sum_permutations = np.zeros((dim_total, dim_total), dtype=complex)

    # Permutations in S3 (using 0-based indexing)
    # Represented as tuples (sigma(0), sigma(1), sigma(2))
    permutations_s3 = [
        (0, 1, 2),  # id
        (1, 0, 2),  # (0 1) -> S_12
        (2, 1, 0),  # (0 2) -> S_13
        (0, 2, 1),  # (1 2) -> S_23
        (1, 2, 0),  # (0 1 2) -> P_123
        (2, 0, 1)   # (0 2 1) -> P_132
    ]

    for perm in permutations_s3:
        sum_permutations += get_permutation_operator(d, k, perm)

    # Normalization factor
    norm = d * (d + 1) * (d + 2)
    if norm == 0: # Handles d=0 case
         return np.zeros((dim_total, dim_total), dtype=complex)

    return sum_permutations / norm