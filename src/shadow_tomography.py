import numpy as np
import qutip
from typing import List, Union, Optional, Tuple, Literal, Iterable, Sequence
import itertools
import math

# Define Type Aliases for clarity
MatrixInput = Union[qutip.Qobj, np.ndarray]
MatrixListInput = Union[MatrixInput, Sequence[MatrixInput]]
BasisType = Union[Literal['pauli', 'flatten'], Iterable[np.ndarray]]
RescalingType = Literal['none', 'trace']
RescalingInput = Union[RescalingType, List[float]]

# Helper function to get Pauli basis for n qubits
def _get_pauli_basis_product(num_qubits: int, normalized: bool = True) -> List[np.ndarray]:
    """
    Generates the tensor product Pauli basis for n qubits.

    The basis elements B_k satisfy Tr(B_j B_k) = d * delta_{jk}, where d=2^n.
    The basis includes the identity matrix.

    Args:
        num_qubits: Number of qubits.
        normalized: If True, normalize the return basis elements so that Tr[sigma_i^2]=1 for all i.

    Returns:
        List[np.ndarray]: List of d*d Pauli basis matrices as NumPy arrays.
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

    basis_matrices_qobj = []
    # Generate all combinations of Pauli strings of length num_qubits
    pauli_strings = itertools.product(['I', 'X', 'Y', 'Z'], repeat=num_qubits)

    for ps in pauli_strings:
        operator_list = [single_qubit_paulis[p] for p in ps]
        # Compute tensor product using qutip.tensor
        tensor_op = qutip.tensor(*operator_list)
        basis_matrices_qobj.append(tensor_op)

    # Convert to numpy arrays
    basis_matrices_np = [op.full() for op in basis_matrices_qobj]

    if normalized:
        # Normalize the basis matrices
        for i in range(len(basis_matrices_np)):
            basis_matrices_np[i] /= np.sqrt(d)

    return basis_matrices_np

# Helper function to convert input to list of numpy arrays and get dimension
def _preprocess_matrices(matrices_in: MatrixListInput) -> List[np.ndarray]:
    """Converts input matrices to a list of numpy arrays and checks their dimensions."""
    # If the input is not a list, wrap it into a list for uniform processing.
    if not isinstance(matrices_in, list):
        matrices_in = [matrices_in]

    # Ensure that the list is not empty.
    if not matrices_in:
        raise ValueError("Input matrix list cannot be empty.")

    matrices_np = []
    
    # Loop over each matrix to validate and convert it.
    for mat in matrices_in:
        # If the matrix is a qutip.Qobj, ensure it is an operator and convert it to a numpy array.
        if isinstance(mat, qutip.Qobj):
            if not mat.isoper:
                raise TypeError("Input Qobj must be an operator.")
            mat_np = mat.full()
        # If the matrix is already a numpy array, use it directly.
        elif isinstance(mat, np.ndarray):
            mat_np = mat
        # If the matrix type is not supported, raise an error.
        else:
            raise TypeError(f"Unsupported matrix type: {type(mat)}. Use qutip.Qobj or numpy.ndarray.")

        # Optional: Uncomment the following lines to enforce matrix Hermiticity.
        # if not np.allclose(mat_np, mat_np.conj().T):
        #     print(f"Warning: Matrix is not Hermitian:\n{mat_np}")

        # Append the processed numpy array to the list.
        matrices_np.append(mat_np)

    return matrices_np

# Helper function to unvectorize a matrix
def _unvectorize_matrix(vector: np.ndarray, basis: BasisType, d: Optional[int] = None) -> np.ndarray:
    """
    Converts a vector back into a d x d matrix using the specified basis.

    Args:
        vector: The d*d dimensional vector.
        basis: The vectorization basis used ('pauli' or 'flatten', or the explicit list of matrices).
        d: The dimension of the target matrix. If None, it is inferred from the vector length.

    Returns:
        np.ndarray: The d x d matrix.
    """
    
    if vector.ndim != 1:
        if vector.ndim == 2 and vector.shape[1] == 1:
            vector = vector.flatten()
        else:
            raise ValueError(f"Input vector must be a vector. Got {vector.ndim}D array instead.")

    if d is None:
        d = int(math.sqrt(vector.shape[0]))
        if d * d != vector.shape[0]:
            raise ValueError(f"Vector length {vector.shape[0]} is not a perfect square.")

    if basis == 'flatten':
        return vector.reshape((d, d))
    elif basis == 'pauli':
        # check that d is a power of 2
        if d <= 0 or (d & (d - 1)) != 0:
            raise ValueError(f"Dimension {d} must be a power of 2 for Pauli basis.")
        num_qubits = int(math.log2(d))
        # Generate the Pauli basis for the given number of qubits
        pauli_basis = _get_pauli_basis_product(num_qubits, normalized=True)
        # Unvectorize using the Pauli basis
        matrix = np.zeros((d, d), dtype=np.complex128)
        for k in range(d*d):
            matrix += vector[k] * pauli_basis[k]
        return matrix
    elif isinstance(basis, Iterable):
        basis_list = list(basis)
        # If a list of matrices is provided, use them for unvectorization
        if len(basis_list) != d*d:
            raise ValueError(f"Basis list length {len(basis_list)} does not match vector length {d*d}.")
        matrix = np.zeros((d, d), dtype=np.complex128)
        for k in range(d*d):
            matrix += vector[k] * basis_list[k]
        return matrix
    else:
        raise ValueError(f"Unknown basis type: {basis}")

# --- Main Functions ---

def vectorize_density_matrix(
    matrices_in: MatrixListInput,
    basis: BasisType = 'pauli'
) -> np.ndarray:
    """
    Vectorizes one or more Hermitian matrices using the specified basis.

    Args:
        matrices_in: A single matrix (qutip.Qobj or np.ndarray) or a list of matrices.
        basis: The vectorization basis to use:
            'pauli': Use the tensor product Pauli basis (requires dimension d=2^n).
                     Vector components are v_k = Tr(M B_k), where B_k are Pauli basis elements.
            'flatten': Flatten the matrix row by row.

    Returns:
        np.ndarray: A 2D NumPy array where each column is the vectorized version
                    of the corresponding input matrix. The shape is (d*d, num_matrices).
    """
    matrices_np = _preprocess_matrices(matrices_in)
    d = matrices_np[0].shape[0] # Assuming all matrices have the same dimension

    vectorized_matrices = []
    if basis == 'pauli':
        num_qubits = math.log2(d)
        if not num_qubits.is_integer() or num_qubits <= 0:
            raise ValueError(f"Pauli basis is currently only supported for dimensions d=2^n (n > 0). Got d={d}")
        pauli_basis = _get_pauli_basis_product(int(num_qubits))

        if len(pauli_basis) != d*d:
             raise RuntimeError("Internal error: Pauli basis generation failed.") # Should be caught earlier
        
        for _, mat in enumerate(matrices_np):
            vectorized_matrices.append(
                np.array([np.trace(mat @ p_k) for p_k in pauli_basis], dtype=np.complex128))
    elif basis == 'flatten':
        for mat in matrices_np:
            vectorized_matrices.append(mat.flatten())
    else:
        raise ValueError(f"Unknown basis type: {basis}")

    # Stack vectors column-wise
    # np.vstack creates rows, so transpose is needed.
    # np.column_stack directly creates columns from 1D arrays.
    return np.column_stack(vectorized_matrices)


def frame_operator(
    povm: Sequence[MatrixInput],
    basis: BasisType = 'pauli',
    rescaling: RescalingInput = 'none'
) -> np.ndarray:
    """
    Computes the frame operator F for a given POVM.

    The frame operator is defined as F = sum_b alpha_b^2 |v_b><v_b|, where v_b
    is the vectorized version of the POVM element mu_b, and alpha_b are the
    rescaling factors. |v_b><v_b| denotes the outer product v_b @ v_b.conj().T.

    Args:
        povm: A list of POVM elements (qutip.Qobj or np.ndarray).
        basis: The vectorization basis ('pauli' or 'flatten').
        rescaling: Rescaling strategy for POVM elements before vector projection:
            'none': alpha_b = 1 for all b.
            'trace': alpha_b = Tr(mu_b) / d, where d is the dimension.
            list: A list of custom rescaling factors alpha_b. Must have the
                  same length as the POVM list.

    Returns:
        np.ndarray: The frame operator F (a d*d x d*d NumPy array).
    """
    if not isinstance(povm, list) or not povm:
        raise ValueError("POVM must be a non-empty list of matrices.")

    povm_np = _preprocess_matrices(povm)
    num_elements = len(povm_np)
    d = povm_np[0].shape[0] # space dimension; Assuming all matrices have the same dimension
    vec_dim = d * d

    # Determine rescaling factors
    alpha_b = np.ones(num_elements) # Default: 'none'
    if isinstance(rescaling, list):
        if len(rescaling) != num_elements:
            raise ValueError(f"Custom rescaling list length ({len(rescaling)}) must match POVM length ({num_elements}).")
        alpha_b = np.array(rescaling, dtype=float)
    elif rescaling == 'trace':
        traces = np.array([np.trace(mu) for mu in povm_np])
        # Ensure trace is real, as POVM elements should be positive semi-definite
        alpha_b = 1 / np.sqrt(np.real(traces) / d)
    elif rescaling != 'none':
        raise ValueError(f"Unknown rescaling option: {rescaling}")

    # Initialize frame operator
    frame_op = np.zeros((vec_dim, vec_dim), dtype=np.complex128)

    # Vectorize all POVM elements at once for potential efficiency
    # Resulting shape is (vec_dim, num_elements)
    vectorized_povm = vectorize_density_matrix(povm_np, basis=basis)

    # Compute sum of scaled outer products
    for b in range(num_elements):
        v_b = vectorized_povm[:, b:b+1] # Get b-th column as a (vec_dim, 1) array
        alpha_sq = alpha_b[b]**2
        # Outer product: (vec_dim, 1) @ (1, vec_dim) -> (vec_dim, vec_dim)
        outer_product = v_b @ v_b.conj().T
        frame_op += alpha_sq * outer_product

    return frame_op


def shadow_estimator(
    povm: List[MatrixInput],
    basis: BasisType = 'pauli',
    rescaling: RescalingInput = 'none'
) -> List[np.ndarray]:
    """
    Computes the dual POVM (shadow estimator) for a given POVM.

    The dual POVM elements E_b are constructed such that their vectorized
    versions w_b satisfy w_b = F^{-1} @ (alpha_b^2 * v_b), where v_b is the
    vectorized original POVM element mu_b, F is the frame operator computed
    with the same rescaling, and alpha_b are the rescaling factors.
    The function returns the unvectorized matrices E_b.

    Args:
        povm: A list of POVM elements (qutip.Qobj or np.ndarray).
        basis: The vectorization basis ('pauli' or 'flatten').
        rescaling: Rescaling strategy. Must be consistent with how the frame
                   operator F was (or would be) computed. Options:
                   'none', 'trace', or a list of custom factors.

    Returns:
        List[np.ndarray]: A list of the dual POVM element matrices E_b as
                          NumPy arrays.
    """
    if not isinstance(povm, list) or not povm:
        raise ValueError("POVM must be a non-empty list of matrices.")

    d, povm_np = _preprocess_matrices(povm)
    num_elements = len(povm_np)

    # 1. Compute the frame operator
    frame_op = frame_operator(povm_np, basis=basis, rescaling=rescaling)

    # 2. Compute the inverse (or pseudo-inverse) of the frame operator
    try:
        # Use pseudo-inverse for numerical stability and handling singular cases
        frame_op_inv = np.linalg.pinv(frame_op)
        # Check condition number if worried about inversion accuracy
        # cond_num = np.linalg.cond(frame_op)
        # if cond_num > 1e10: # Example threshold
        #     print(f"Warning: Frame operator might be ill-conditioned. Condition number: {cond_num}")
    except np.linalg.LinAlgError:
        raise RuntimeError("Failed to compute the inverse of the frame operator.")

    # 3. Determine rescaling factors (again, needed for the formula)
    alpha_b = np.ones(num_elements) # Default: 'none'
    if isinstance(rescaling, list):
        if len(rescaling) != num_elements:
             # This check is also done in frame_operator, but repeat for safety
            raise ValueError(f"Custom rescaling list length ({len(rescaling)}) must match POVM length ({num_elements}).")
        alpha_b = np.array(rescaling, dtype=float)
    elif rescaling == 'trace':
        traces = np.array([np.trace(mu) for mu in povm_np])
        alpha_b = np.real(traces) / d
    elif rescaling != 'none':
        raise ValueError(f"Unknown rescaling option: {rescaling}")

    # 4. Vectorize the original POVM elements
    vectorized_povm = vectorize_density_matrix(povm_np, basis=basis) # Shape (d*d, num_elements)

    # 5. Compute the vectorized dual POVM elements w_b = F^{-1} @ (alpha_b^2 * v_b)
    vectorized_dual_povm = np.zeros_like(vectorized_povm, dtype=np.complex128)
    for b in range(num_elements):
        v_b = vectorized_povm[:, b] # Get b-th column as a (d*d,) array
        scaled_v_b = (alpha_b[b]**2) * v_b
        w_b = frame_op_inv @ scaled_v_b
        vectorized_dual_povm[:, b] = w_b

    # 6. Unvectorize the dual POVM vectors w_b back into matrices E_b
    dual_povm_matrices = []
    pauli_basis = None
    if basis == 'pauli':
         # Need the basis matrices for unvectorization
        num_qubits = math.log2(d)
        # Check should have been done during vectorization, but belt-and-suspenders
        if not num_qubits.is_integer() or num_qubits <= 0:
            raise ValueError(f"Pauli basis requires d=2^n. Got d={d}")
        _, pauli_basis = _get_pauli_basis_product(int(num_qubits))

    for b in range(num_elements):
        w_b = vectorized_dual_povm[:, b] # Get b-th column as a (d*d,) array
        E_b = _unvectorize_matrix(w_b, d, basis=basis, pauli_basis=pauli_basis)
        dual_povm_matrices.append(E_b)

    return dual_povm_matrices


# --- Example Usage ---
if __name__ == '__main__':
    # --- Example 1: Single Qubit (d=2) ---
    print("--- Example 1: Single Qubit (d=2) ---")
    # Define Pauli matrices (using qutip)
    sigma_i = qutip.qeye(2)
    sigma_x = qutip.sigmax()
    sigma_y = qutip.sigmay()
    sigma_z = qutip.sigmaz()

    # Example density matrix (mixed state)
    rho = 0.7 * qutip.ket2dm(qutip.basis(2, 0)) + 0.3 * qutip.ket2dm(qutip.basis(2, 1))
    print("Original Density Matrix (qutip):\n", rho)

    # Vectorize using Pauli basis
    vec_rho_pauli = vectorize_density_matrix(rho, basis='pauli')
    print("\nVectorized Rho (Pauli basis):\n", vec_rho_pauli)
    # Expected: [Tr(rho*I), Tr(rho*X), Tr(rho*Y), Tr(rho*Z)] = [1, 0, 0, 0.4] for this diagonal rho

    # Vectorize using flatten basis
    vec_rho_flat = vectorize_density_matrix(rho, basis='flatten')
    print("\nVectorized Rho (Flatten basis):\n", vec_rho_flat)
    # Expected: [0.7, 0, 0, 0.3] reshaped

    # Unvectorize Pauli basis vector back to matrix
    d_rho, pauli_basis_2 = _get_pauli_basis_product(1)
    rho_reconstructed_pauli = _unvectorize_matrix(vec_rho_pauli.flatten(), d_rho, 'pauli', pauli_basis_2)
    print("\nReconstructed Rho (from Pauli vector):\n", rho_reconstructed_pauli)
    print("Close to original?", np.allclose(rho.full(), rho_reconstructed_pauli))

    # --- Example 2: Pauli POVM for a single qubit (SIC-POVM basis up to scaling) ---
    print("\n--- Example 2: Pauli POVM (d=2) ---")
    # A simple, informationally complete POVM (unnormalized projectors)
    povm_pauli = [
        (sigma_i + sigma_x).full(), # Using numpy arrays directly
        (sigma_i - sigma_x).full(),
        (sigma_i + sigma_y).full(),
        (sigma_i - sigma_y).full(),
        (sigma_i + sigma_z).full(),
        (sigma_i - sigma_z).full(),
    ]
    # Normalize to sum to Identity * 2 (standard for Pauli measurements)
    povm_pauli_norm = [(1/3.0) * p for p in povm_pauli]


    print("POVM Elements (first one):\n", povm_pauli_norm[0])

    # Calculate Frame Operator (Pauli basis, no rescaling)
    F_pauli_none = frame_operator(povm_pauli_norm, basis='pauli', rescaling='none')
    print("\nFrame Operator (Pauli basis, no rescaling):\n", F_pauli_none)
    # For this specific POVM, F should be proportional to identity

    # Calculate Frame Operator (Pauli basis, trace rescaling)
    F_pauli_trace = frame_operator(povm_pauli_norm, basis='pauli', rescaling='trace')
    print("\nFrame Operator (Pauli basis, trace rescaling):\n", F_pauli_trace)
    # Tr(mu_b) = Tr(I/3 +/- P/3) = 2/3. alpha_b = (2/3)/2 = 1/3. F_trace = (1/3)^2 * F_none

    # Calculate Shadow Estimator (Dual POVM)
    shadow_pauli_none = shadow_estimator(povm_pauli_norm, basis='pauli', rescaling='none')
    print("\nShadow Estimator (Dual POVM) (first element):\n", shadow_pauli_none[0])

    # --- Example 3: Two Qubits (d=4) ---
    print("\n--- Example 3: Two Qubits (d=4) ---")
    rho_2q = qutip.ket2dm(qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 1))) # |01><01| state
    vec_rho_2q_pauli = vectorize_density_matrix(rho_2q, basis='pauli')
    print("\nVectorized |01><01| (Pauli basis, shape):", vec_rho_2q_pauli.shape)
    # print("Vector:", vec_rho_2q_pauli.flatten()) # Can be long

    vec_rho_2q_flat = vectorize_density_matrix(rho_2q, basis='flatten')
    print("Vectorized |01><01| (Flatten basis, shape):", vec_rho_2q_flat.shape)
    # print("Vector:", vec_rho_2q_flat.flatten())

    # Create a simple POVM for d=4 (e.g., projectors onto computational basis)
    povm_comp_basis_4 = [qutip.ket2dm(qutip.basis(4, i)) for i in range(4)]
    print("\nPOVM: Computational basis projectors for d=4 (first element):")
    print(povm_comp_basis_4[0])

    # Calculate Frame Operator (Flatten basis, no rescaling)
    F_comp_flat = frame_operator(povm_comp_basis_4, basis='flatten', rescaling='none')
    print("\nFrame Operator (Computational POVM, Flatten basis):\n", F_comp_flat) # Should be diagonal

    # Calculate Shadow Estimator (Dual POVM)
    shadow_comp_flat = shadow_estimator(povm_comp_basis_4, basis='flatten', rescaling='none')
    print("\nShadow Estimator (Computational POVM, Flatten) (first element):\n", shadow_comp_flat[0])
    # For projectors, the dual basis should be related to the original projectors