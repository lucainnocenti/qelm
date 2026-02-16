from __future__ import annotations
import itertools
import logging
logger = logging.getLogger(__name__)

import qutip
import numpy as np

import numbers
from numpy.typing import NDArray, ArrayLike
import typing
from typing import List, Union, Iterable, Optional, Literal, Sequence, cast, ClassVar
from dataclasses import dataclass
from functools import cached_property
# --- Custom types imports ---
from src.types import BasisType, SamplingMethodType


# --------------------------------------------------------------------------- #
# Pauli basis
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PauliBasis:
    """n-qubit tensor-product Pauli operators.

    The idea is to enable generating the full set of Pauli operators via a generator
    (so not necessarily having the whole thing in memory)

    Usage:
    >>> pb = PauliBasis(num_qubits=2).matrices

    Parameters
    ----------
    num_qubits :
        Number of qubits *n* (must be positive).
    normalised :
        If *True*, scale each matrix by ``1/sqrt(2**n)`` so that
        ``Tr(P_i P_j) = δ_ij``.
    include_identity :
        If *False*, the first element (I⊗n) is omitted.
        Useful for traceless-only applications.
    """

    num_qubits: int
    normalised: bool = True
    include_identity: bool = True

    # -- single-qubit paulis ------------------------------------------------- #
    _PAULIS: ClassVar[dict[str, NDArray]] = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    # -- lazy materialisation ------------------------------------------------ #
    @cached_property
    def labels(self) -> List[str]:
        """Lexicographically ordered list of operator labels."""
        labels = [
            "".join(prod)
            for prod in itertools.product(self._PAULIS, repeat=self.num_qubits)
        ]
        if not self.include_identity:
            labels = [lab for lab in labels if lab != "I" * self.num_qubits]
        return labels

    @cached_property
    def factor(self) -> float:
        """Normalisation factor applied to every matrix."""
        return 1 / np.sqrt(2**self.num_qubits) if self.normalised else 1.0

    def _build_matrix(self, label: str) -> NDArray:
        """Materialise a single tensor product."""
        mat: NDArray = self._PAULIS[label[0]]
        for char in label[1:]:
            mat = np.kron(mat, self._PAULIS[char])
        return mat * self.factor

    @cached_property
    def matrices(self) -> List[NDArray]:
        """**Materialises** and stores all matrices (may be huge!)."""
        return [self._build_matrix(lab) for lab in self.labels]

    # -- iterator interface -------------------------------------------------- #
    def __iter__(self):
        for lab in self.labels:
            yield self._build_matrix(lab)

    def __len__(self) -> int:
        return len(self.labels)


# --------------------------------------------------------------------------- #
# Representations of quantum states
# --------------------------------------------------------------------------- #


class QuantumOperator:
    def __init__(self, matrix: ArrayLike):
        matrix = np.asarray(matrix, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Input must be a square 2D array (density matrix).")
        self.data = matrix
        self.dim: int = matrix.shape[0]

    @classmethod
    def from_ket(cls, vector: np.ndarray) -> QuantumOperator:
        """Re-create a QuantumOperator from a ket vector, via outer product."""
        return cls(np.outer(vector, vector.conj()))

    @classmethod
    def from_vector(cls, vec: np.ndarray, basis: BasisType = "pauli") -> QuantumOperator:
        """Re-create a QuantumState from a vectorised representation."""
        return cls(unvectorize_op(vec, basis=basis))

    def vectorise(self, basis: BasisType = "pauli") -> np.ndarray:
        """Return the coordinate vector of rho in the chosen basis."""
        return vectorise_op(self.data, basis=basis)

    def asbatch(self) -> QuantumOperatorsBatch:
        """Convert the quantum density matrix to a batch representation."""
        return QuantumOperatorsBatch(self.data[None, :, :])


class QuantumOperatorsBatch:
    """A batch of quantum operators (density matrices, observables, etc), represented as a 3D array (shape: n_ops x dim x dim)."""
    def __init__(self, operators: ArrayLike):
        operators = np.asarray(operators, dtype=np.complex128)
        if operators.ndim != 3 or operators.shape[1] != operators.shape[2]:
            raise ValueError("Input must be a 3D array (shape: n_ops x dim x dim).")
        self.data = operators
        self.dim: int = operators.shape[1]
        self.n_ops: int = operators.shape[0]

    def __len__(self) -> int: return self.n_ops
    def __iter__(self): return iter(self.data)
    def __getitem__(self, index): return self.data[index]

    def vectorise(self, basis: BasisType = "pauli") -> NDArray:
        """Vectorise each operator in the batch in the specified basis.
        
        Parameters:
            basis (BasisType): The basis in which to vectorise the operators.
                Can be 'pauli', 'flatten', or a list of matrices.
        Returns:
            NDArray: A 2D array where each row corresponds to a vectorised operator.
            If basis is 'flatten', the shape is (dim * dim, n_ops).
            If basis is a list of matrices, the shape is (n_basis_ops, n_ops).
        """
        if isinstance(basis, str) and basis == 'flatten':
            # return a 2D array where each row is a vectorised matrix
            return self.data.reshape(self.n_ops, -1)
        else:
            basis_ = _make_basis_into_nparray(basis, self.dim)
            matrix = np.einsum('nij,kij->nk', self.data, basis_.conj(), optimize=True).T
            if basis == 'pauli':
                return matrix.real
            return matrix

# class representing POVM (as a list of qutip.Qobj) and the associated label, for example, "SIC", "MUB", etc.
class POVM(QuantumOperatorsBatch):
    """A POVM, represented as a batch of quantum operators (the effects).
    """    
    def __init__(self, elements: ArrayLike, label: Optional[str] = None):
        """Initialize a POVM with the given elements.
        """
        QuantumOperatorsBatch.__init__(self, elements)
        # Generate default label if none provided
        if label is None:
            label = f"unknown, {len(self)} outcomes"
        self.label = label

    def __repr__(self) -> str:
        return f"POVM(label={self.label}, num_outcomes={self.n_ops})"


# --- Type Definitions ---
POVMElement = Union[qutip.Qobj, ArrayLike]
POVMType = Union[Sequence[POVMElement], POVM]  # Allow sequences or the POVM class
ObservablesType = Union[Sequence[qutip.Qobj], qutip.Qobj, Sequence[ArrayLike], ArrayLike, QuantumOperator, QuantumOperatorsBatch]



class QuantumState:
    """Base class for quantum states, providing a common interface."""
    def __init__(self, dim: int):
        if dim <= 0:
            raise ValueError("Dimension must be a positive integer.")
        self.dim = dim

    def todm(self) -> QuantumDensityMatrix:
        """Convert the quantum state to a density matrix representation."""
        raise NotImplementedError("Subclasses must implement this method.")

    def vectorise(self, basis: BasisType = "pauli") -> np.ndarray:
        """Return the coordinate vector of the state in the chosen basis."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def asbatch(self) -> QuantumStatesBatch:
        """Convert the quantum state to a batch representation."""
        raise NotImplementedError("Subclasses must implement this method.")

    def expvals(self, observables: ObservablesType) -> NDArray[np.float64]:
        return self.asbatch().expvals(observables).flatten()

    def measure_povm(self, povm: POVM, statistics: float,
                     sampling_method: SamplingMethodType = 'standard',
                     return_frequencies: bool = True) -> NDArray:
        return self.asbatch().measure_povm(
            povm=povm,
            statistics=statistics,
            sampling_method=sampling_method,
            return_frequencies=return_frequencies
        ).flatten()


class QuantumKet(QuantumState):
    """Density-operator wrapper with vectorisation helpers."""
    # --------------- construction helpers ---------------------------------- #
    def __init__(self, vec: NDArray | Sequence[complex]):
        data = np.asarray(vec, dtype=np.complex128)
        if data.ndim != 1:
            if data.ndim == 2 and (data.shape[1] == 1 or data.shape[0] == 1):
                # if it's a 2D array with one column or one row, we can treat it as a 1D vector
                data = data.flatten()
            else:
                raise ValueError("Input vector must be 1D (a ket).")
        self.data = data
        self.dim: int = data.shape[0]

    def todm(self) -> QuantumDensityMatrix:
        """Converts into a density matrix representation (as a 2D array)."""
        return QuantumDensityMatrix.from_ket(self.data)

    def vectorise(self, basis: BasisType = "pauli") -> np.ndarray:
        """Return the coordinate vector of rho in the chosen basis.
        The operation is devolved to QuantumDensityMatrix.vectorise().
        """
        return QuantumDensityMatrix.from_ket(self.data).vectorise(basis=basis)
    
    def asbatch(self) -> QuantumKetsBatch:
        """Convert the quantum ket to a batch representation."""
        return QuantumKetsBatch(self.data[None, :])

    # def expvals(self, observables: ObservablesType) -> NDArray[np.float64]:
    #     """Calculate the expectation values of the given observables for each ket in the batch.
        
    #     Parameters:
    #         observables (Union[qutip.Qobj, Sequence[qutip.Qobj], NDArray]): A single observable or a sequence of observables.
        
    #     Returns:
    #         NDArray[np.float64]: The expectation values, shape (n_observables, n_states).
    #     """
    #     # parse input to ensure observables is a 3D array
    #     observables_np = _make_observables_into_nparray(observables)
    #     # compute the expectation values
    #     return expvals_from_kets_and_observables_numpy(kets=self.data, observables=observables_np).flatten()


class QuantumDensityMatrix(QuantumOperator, QuantumState):
    """Density matrix representation of a quantum state."""
    def __init__(self, matrix: NDArray[np.complex128]):
        QuantumOperator.__init__(self, matrix)

    @classmethod
    def from_ket(cls, vector: NDArray[np.complex128]) -> QuantumDensityMatrix:
        """Create a density matrix from a ket vector."""
        return cls(np.outer(vector, vector.conj()))

    def todm(self) -> QuantumDensityMatrix:
        """Returns itself as it is already a density matrix."""
        return self

    def asbatch(self) -> QuantumDensityMatricesBatch:
        """Convert the quantum density matrix to a batch representation."""
        return QuantumDensityMatricesBatch(self.data[None, :, :])
    
    # def expvals(self, observables: ObservablesType) -> NDArray[np.float64]:
    #     # parse input to ensure observables is a 3D array
    #     observables_np = _make_observables_into_nparray(observables)
    #     # compute the expectation values
    #     return expvals_from_dms_and_observables_numpy(dms=self.data, observables=observables_np).flatten()


class QuantumStatesBatch:
    """Base class for batches of quantum states, providing a common interface.
    
    The point of these classes is to provide efficient handling of multiple states avoiding for loops.
    """
    def __init__(self):
        self.data: NDArray[np.complex128]
        self.n_states: int
        self.dim: int
        raise NotImplementedError("Subclasses must implement this method.")
    
    def vectorise(self, basis: BasisType = "pauli") -> NDArray[np.complex128]:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def measure_povm(self, povm: POVM, statistics: float, 
                     sampling_method: SamplingMethodType = 'standard',
                     return_frequencies: bool = True) -> NDArray:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def expvals(self, observables: ObservablesType) -> NDArray[np.float64]:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def __len__(self) -> int:
        raise NotImplementedError("Subclasses must implement this method.")
    

class QuantumKetsBatch(QuantumStatesBatch):
    """A batch of quantum kets, represented as a 2D array (shape: n_states x dim)."""

    def __init__(self, kets: Sequence | NDArray[np.complex128]):
        self.data = np.asarray(kets, dtype=np.complex128)
        if self.data.ndim != 2:
            raise ValueError("Input must be a 2D array (shape: n_states x dim).")
        self.dim: int = self.data.shape[1]
        self.n_states: int = self.data.shape[0]

    def todm(self) -> QuantumDensityMatricesBatch:
        """Convert each ket in the batch to a density matrix."""
        matrices = np.einsum('ni,nj->nij', self.data, self.data.conj(), optimize=True)
        return QuantumDensityMatricesBatch(matrices)
    
    def vectorise(self, basis: BasisType = "pauli") -> NDArray[np.complex128]:
        return self.todm().vectorise(basis=basis)
    
    def measure_povm(self, povm: POVM, statistics: float,
                     sampling_method: SamplingMethodType = 'standard',
                     return_frequencies: bool = True) -> NDArray:
        """Measure a POVM on each ket in the batch, with some given statistics.
        
        Parameters:
            povm (POVM): The POVM to measure.
            statistics (float): Number of measurement shots PER KET. Use np.inf for exact probabilities.
            sampling_method (str): 'standard' or 'poisson'.
            return_frequencies (bool): If True, return frequencies instead of raw outcomes.
        Returns:
            NDArray: The measurement outcomes or frequencies.
            If return_frequencies is True, shape is (n_povm_outcomes, n_states).
            If return_frequencies is False, shape is (n_states, statistics).
        """
        if not isinstance(povm, POVM):
            raise TypeError("POVM must be an instance of the POVM class.")
        probabilities = expvals_from_kets_and_observables_numpy(self.data, povm.data)
        return sample_from_probabilities(
            probabilities.T,
            statistics,
            return_frequencies=return_frequencies,
            sampling_method=sampling_method
        ).T
    
    def expvals(self, observables: ObservablesType) -> NDArray[np.float64]:
        """Calculate the expectation values of the given observables for each ket in the batch.
        
        Parameters:
            observables (Union[qutip.Qobj, Sequence[qutip.Qobj], NDArray]): A single observable or a sequence of observables.
        
        Returns:
            NDArray[np.float64]: The expectation values, shape (n_observables, n_states).
        """
        # parse input to ensure observables is a 3D array
        observables_np = _make_observables_into_nparray(observables)
        # compute the expectation values
        return expvals_from_kets_and_observables_numpy(kets=self.data, observables=observables_np)

    def __len__(self) -> int: return self.n_states




class QuantumDensityMatricesBatch(QuantumOperatorsBatch, QuantumStatesBatch):
    """A batch of quantum density matrices, represented as a 3D array (shape: n_dms x dim x dim)."""
    def __init__(self, matrices: NDArray):
        QuantumOperatorsBatch.__init__(self, matrices)
        self.n_states = self.n_ops  # n_dms is the same as n_states in this context

    def expvals(self, observables: ObservablesType) -> NDArray[np.float64]:
        """Calculate the expectation values of the given observables for each density matrix in the batch.
        
        Parameters:
            observables (ObservablesType): A single observable or a sequence of observables.
        
        Returns:
            NDArray[np.float64]: The expectation values, shape (n_observables, n_matrices).
        """
        # parse input to ensure observables is a 3D array
        observables_np = _make_observables_into_nparray(observables)
        # compute the expectation values
        return expvals_from_dms_and_observables_numpy(dms=self.data, observables=observables_np)
    
    def measure_povm(self, povm: POVM, statistics: float,
                     sampling_method: SamplingMethodType = 'standard',
                     return_frequencies: bool = True) -> NDArray:
        """Measure a POVM on each density matrix in the batch, with the specified statistics.
        
        Parameters:
            povm (POVM): The POVM to measure.
            statistics (float): Number of measurement shots PER DENSITY MATRIX. Use np.inf for exact probabilities.
            sampling_method (str): 'standard' or 'poisson'.
            return_frequencies (bool): If True, return frequencies instead of raw outcomes.
        Returns:
            NDArray: The measurement outcomes or frequencies.
            If return_frequencies is True, shape is (n_povm_outcomes, n_matrices).
            If return_frequencies is False, shape is (n_matrices, statistics).
        """
        if not isinstance(povm, POVM):
            raise TypeError("POVM must be an instance of the POVM class.")
        probabilities = expvals_from_dms_and_observables_numpy(self.data, povm.data)
        return sample_from_probabilities(
            probabilities.T,
            statistics,
            return_frequencies=return_frequencies,
            sampling_method=sampling_method
        ).T


def _make_observables_into_nparray(observables: ObservablesType) -> NDArray:
    """Helper function to parse the ObservablesType argument.
    
    Used eg by expvals_from_states_and_observables_qutip and expvals_from_kets_and_observables_numpy.
    Ensures that observables is a 3D array (shape: n_observables x dim x dim).
    """
    # print(observables)
    # print(isinstance(observables, QuantumOperatorsBatch))
    if isinstance(observables, QuantumOperator):
        # if it's a single QuantumOperator, convert it to a 3D array
        observables_np = observables.data.reshape(1, *observables.data.shape)
    elif isinstance(observables, QuantumOperatorsBatch):
        # if it's a QuantumOperatorsBatch, we can use its data directly
        observables_np = observables.data
    elif isinstance(observables, qutip.Qobj):
        observables_np = observables.full()[None, :, :]
    elif isinstance(observables, Sequence) and all(isinstance(obs, qutip.Qobj) for obs in observables):
        observables_np = np.array(
            [cast(qutip.Qobj, obs).full() for obs in observables],
            dtype=np.complex128)
    elif isinstance(observables, Sequence) and all(isinstance(obs, np.ndarray) for obs in observables):
        # assume each observable is a 2D array (a matrix)
        observables_np = np.array(observables, dtype=np.complex128)
    elif isinstance(observables, np.ndarray):
        if observables.ndim == 2:
            # if it's a 2D array, we assume it's a single observable
            observables = observables[None, :, :]  # reshape to 3D (1, dim, dim)
        elif observables.ndim != 3 or observables.shape[1] != observables.shape[2]:
            raise ValueError("Observables must be a 3D array (shape: n_observables x dim x dim).")
        observables_np = observables
    else:
        raise TypeError("Observables must be a qutip.Qobj, a sequence of qutip.Qobj, or a numpy array.")
    return observables_np


def _make_basis_into_nparray(basis: BasisType, dim: int) -> NDArray[np.complex128]:
    """Helper function to parse the BasisType argument.
    
    Used eg by vectorise_op and unvectorize_op.
    """
    if isinstance(basis, str):
        if basis == 'pauli':
            num_qubits = int(np.log2(dim))
            if 2**num_qubits != dim:
                raise ValueError(f"Pauli basis is only supported for dimensions d = 2^n (n > 0). Got d={dim}.")
            basis_ = np.asarray(PauliBasis(num_qubits=num_qubits, normalised=True).matrices)
        else:
            raise ValueError(f"Unknown basis type: {basis}. Must be 'pauli' or 'flatten'.")
    elif isinstance(basis, Sequence):
        # if basis is a sequence we assume it's a list of matrices
        if not isinstance(basis[0], np.ndarray) or basis[0].ndim != 2 or len(basis) == 0:
            raise ValueError("If basis is a sequence, it must be a list of 2D matrices.")
        basis_ = np.asarray(basis, dtype=np.complex128)
    elif isinstance(basis, np.ndarray):
        if basis.ndim != 3:
            raise ValueError(f"Basis must be a 3D array (list of operators). Got {basis.ndim} dimensions.")
        basis_ = basis
    else:
        raise ValueError(f"Unknown basis type: {basis}. Must be 'pauli', 'flatten', or a list of matrices.")
    return basis_


def sequence_of_qutip_qobj_to_states_batch(data: Sequence[qutip.Qobj]) -> QuantumStatesBatch:
    """Convert a sequence of qutip.Qobj to a QuantumStatesBatch."""
    if not all(isinstance(item, qutip.Qobj) for item in data):
        raise TypeError("All items in the sequence must be qutip.Qobj.")
    # check if all elements are kets
    if all(item.isket for item in data):
        # if all items are kets, we can create a QuantumKetsBatch
        matrices = np.array([item.full().flatten() for item in data], dtype=np.complex128)
        return QuantumKetsBatch(matrices)
    elif all(item.isoper for item in data):
        # if all items are operators (density matrices), we can create a QuantumDensityMatricesBatch
        matrices = np.array([item.full() for item in data], dtype=np.complex128)
        return QuantumDensityMatricesBatch(matrices)
    else:
        raise ValueError("All items in the sequence must be either kets or density matrices (operators).")


def unvectorize_op(vector: NDArray, basis: BasisType = 'pauli') -> NDArray:
    """
    Converts a flattened vector back into a dxd matrix using the specified basis.

    Args:
        vector: The flattened matrix as a 1D NumPy array.
        basis: Either 'pauli', 'flatten', or an iterable of basis matrices.

    Returns:
        np.ndarray: The reconstructed dxd matrix.
    """
    # make sure the vector is 1D or equivalent to 1D (ie 2D with one column)
    if not isinstance(vector, np.ndarray) or vector.ndim != 1:
        raise ValueError(f"Input vector must be 1D. Got an array with {vector.ndim} dimensions.")
    # compute the dimension if not provided, assuming the provided vector comes vectorizing a d*d matrix
    d = int(np.sqrt(vector.size))
    if d * d != vector.size:
        raise ValueError(f"Vector length {vector.size} is not a perfect square.")

    if isinstance(basis, str) and basis == 'flatten':
        return vector.reshape((d, d))
    else:
        basis_ = _make_basis_into_nparray(basis, d)  # returns a 3D array of shape (n_ops, d, d)

    # compute the op as \sum_k vector[k] * basis_[k]
    op = np.einsum('k,kij->ij', vector, basis_, optimize=True)
    return op


def vectorise_op(op: NDArray, basis: BasisType = 'pauli') -> NDArray:
    """Vectorise an operator in a given basis.
    
    Parameters:
        op (NDArray): The operator to vectorise. It should be a 2D array (a matrix, eg a density operator).
        basis (BasisType): The basis in which to vectorise the operator.
            Can be 'pauli', 'flatten', or a list of matrices.
    """
    if not isinstance(op, np.ndarray) and op.ndim != 2:
        raise ValueError(f"Operator must be a 2D array (matrix). Got {op.ndim} dimensions.")
    if isinstance(basis, str) and basis == 'flatten':
        return op.flatten()
    else:
        basis_ = _make_basis_into_nparray(basis, op.shape[0])

    if basis_.shape[1] != op.shape[0] or basis_.shape[2] != op.shape[1]:
        raise ValueError(f"Basis shape {basis_.shape} does not match operator shape {op.shape}.")

    # actual bloody computation
    vec = np.einsum('ij,kij->k', op, basis_.conj())
    # if the basis is Pauli, we return the real part only
    if basis == 'pauli':
        return vec.real
    return vec


def vectorise_op_paulis(op: ArrayLike) -> NDArray[np.float64]:
    """
    Vectorize an operator in the Pauli basis.

    Parameters:
    -----------
    op : NDArray[np.complex128]
        The operator to vectorize, expected to be a 2D array (matrix).

    Returns:
    --------
    NDArray[np.float64]
        The vectorized operator in the Pauli basis.
    """
    op = np.asarray(op, dtype=np.complex128)
    pauli_basis = PauliBasis(num_qubits=int(np.log2(op.shape[0])), normalised=True).matrices
    return vectorise_op(op, np.array(pauli_basis, dtype=np.complex128)).real


def ket2dm(ket: ArrayLike) -> NDArray[np.complex128]:
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



def random_kets(dim: int, num_kets: int, seed: Optional[int] = None) -> NDArray[np.complex128]:
    """
    Generate a list of random kets in a d-dimensional Hilbert space.
    This is much faster than generating random states using qutip.rand_ket, especially for many kets.

    Parameters:
    -----------
    dim : int
        The dimension of the Hilbert space.
    num_kets : int
        The number of random kets to generate.
    seed : Optional[int], optional
        Seed for reproducibility (default is None).

    Returns:
    --------
    List[NDArray[np.complex128]]
        A list of random kets, each represented as a NumPy array.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Step 1: Generate D complex numbers with real and imaginary parts from N(0,1)
    # For a batch of num_kets, this will be an array of shape (num_kets, dim)
    raw_complex_vectors = np.random.randn(num_kets, dim) + 1j * np.random.randn(num_kets, dim)

    # Step 2: Normalize each vector
    # Calculate the L2 norm along the dimension axis (axis=1)
    # norms will be of shape (num_kets,)
    norms = np.linalg.norm(raw_complex_vectors, axis=1)

    # Normalize. We need to reshape norms to (num_kets, 1) for broadcasting
    # (num_kets, dim) / (num_kets, 1) -> (num_kets, dim)
    haar_states = raw_complex_vectors / norms[:, np.newaxis]
    if not isinstance(haar_states, np.ndarray):
        raise TypeError("Generated states should be a NumPy array. Some weird shit happened because it's not.")

    return haar_states

def random_density_matrices(n: int, dim: int, seed=None, dtype=np.complex128) -> NDArray[np.complex128]:
    """
    Generate n random dxd density matrices from the Hilbert–Schmidt ensemble.

    Returns: array of shape (n, d, d), complex Hermitian PSD, trace 1.
    """
    rng = np.random.default_rng(seed)

    # Complex Ginibre: entries ~ N(0,1) + i N(0,1)
    G = rng.standard_normal((n, dim, dim)) + 1j * rng.standard_normal((n, dim, dim))
    G = G.astype(dtype, copy=False)

    # X = G G†  (batched)
    X = G @ np.conjugate(np.swapaxes(G, -1, -2))

    # Normalize by trace (batched)
    tr = np.real(np.trace(X, axis1=-2, axis2=-1))  # shape (n,)
    rho = X / tr[:, None, None]
    return rho

def expvals_from_states_and_observables_qutip(
    states: Union[qutip.Qobj, Sequence[qutip.Qobj]],
    observables: Union[qutip.Qobj, Sequence[qutip.Qobj]]
) -> NDArray[np.float64]:
    """Calculate the probabilities of measurement outcomes for a given set of states and a POVM."""
    if isinstance(observables, qutip.Qobj):
        observables = [observables]    
    if isinstance(states, qutip.Qobj):
        states = [states]
    
    expvals = np.array(
        [qutip.expect(obs, states).real for obs in observables]
    ).T  # shape (n_states, n_povm_outcomes)
    
    return expvals

def expvals_from_kets_and_observables_numpy(
    kets: NDArray[np.complex128],
    observables: NDArray[np.complex128]
) -> NDArray[np.float64]:
    """
    Calculate the expectation values for a given set of kets and observables, using numpy.

    Everything is taken to be a numpy array here.
    kets is a 2D array where each row is a ket (shape: n_kets x dim),
    and observables is a 3D array of observables (shape: n_observables x dim x dim).

    Returns a 2D array of expectation values (shape: n_observables x n_kets).
    """
    if kets.ndim == 1:
        # If kets is a 1D array, we assume it's a single ket
        kets = kets[None, :]
    elif kets.ndim != 2:
        raise ValueError("kets must be a 2D array (shape: n_kets x dim). (at least for now)")
    
    expvals = np.einsum('ni,kij,nj->kn',
        kets.conj(),
        observables,
        kets,
        optimize=True
    ).real
    # shape (n_observables, n_kets)    
    return expvals

def expvals_from_dms_and_observables_numpy(
    dms: NDArray[np.complex128],
    observables: NDArray[np.complex128]
) -> NDArray[np.float64]:
    """
    Calculate the expectation values for a given set of density matrices and observables, using numpy.

    dms is a 3D array (shape: n_dms x dim x dim),
    and observables is a 3D array (shape: n_observables x dim x dim).

    Returns a 2D array of expectation values (shape: n_observables x n_dms).
    """
    if dms.ndim == 2:
        # If dms is a 2D array, we assume it's a single density matrix
        dms = dms[None, :, :]
    elif dms.ndim != 3 or dms.shape[1] != dms.shape[2]:
        raise ValueError("dms must be a 3D array (shape: n_dms x dim x dim).")
    if observables.ndim != 3 or observables.shape[1] != observables.shape[2]:
        raise ValueError("observables must be a 3D array (shape: n_observables x dim x dim).")
    
    expvals = np.einsum('nij,kij->kn',
        dms,
        observables.conj(),
        optimize=True
    ).real
    # shape (n_observables, n_dms)
    return expvals


def sample_from_probabilities(
    probabilities: NDArray[np.float64],
    statistics: float,
    return_frequencies: bool = True,
    sampling_method: SamplingMethodType = 'standard'
) -> NDArray:
    """Sample outcomes from a given set of probabilities.
    
    Parameters
    ----------
    probabilities : NDArray[np.float64]
        A 2D array of shape (n_states, num_outcomes) containing the probabilities for each outcome.
    statistics : float
        Number of measurement shots PER STATE. Use np.inf for exact probabilities.
        Finite non-integer values will raise an error.
    return_frequencies : bool, optional
        - If True: Return measurement frequencies for each state.
          Shape: (n_states, n_outcomes).
        - If False (default) and sampling_method='standard': Return raw sampled
          outcome indices for each state. Shape: (n_states, statistics).
        - If False and sampling_method='poisson': Raises ValueError.
    sampling_method : SamplingMethodType, optional
        - 'standard': Sample outcomes using multinomial distribution based on
          exact probabilities for the requested number of shots (statistics).
        - 'poisson': Sample counts for each outcome using a Poisson distribution
          with mean = statistics * probability. Requires return_frequencies=True.
        (default: 'standard').
    """
    if not isinstance(probabilities, np.ndarray):
        raise TypeError("Probabilities must be a numpy.ndarray.")
    if probabilities.ndim == 1:
        # If probabilities is a 1D array, we assume it's for a single state
        probabilities = probabilities.reshape(1, -1)
    elif probabilities.ndim != 2:
        raise ValueError(f"Probabilities must be a 2D array (shape: n_states x num_outcomes). Got {probabilities.ndim} dimensions.")
    # Check statistics value (return probabilities if np.inf, round non-inf floats)
    if np.isinf(statistics):
        if not return_frequencies:
             raise ValueError("Cannot return raw outcomes for infinite statistics. Set return_frequencies=True.")
        logger.debug("Returning exact probabilities for infinite statistics.")
        return probabilities  # Shape (num_probs, num_outcomes)
    elif not isinstance(statistics, numbers.Integral):
        statistics = int(statistics)
        raise ValueError(f"Statistics must be a positive integer or np.inf. You gave {statistics}.")

    n_states = probabilities.shape[0]
    num_outcomes = probabilities.shape[1]
    if sampling_method == 'standard':
        if return_frequencies:
            statistics = int(statistics)
            # Calculate frequencies for each state
            all_frequencies = np.zeros_like(probabilities, dtype=float)
            for i in range(n_states):
                counts = np.random.multinomial(n=statistics, pvals=probabilities[i])
                all_frequencies[i, :] = counts / statistics
            return all_frequencies # Shape (n_states, num_outcomes)
        else:
            # Return raw sampled outcomes for each state
            all_sampled_outcomes = np.zeros((n_states, statistics), dtype=int)
            for i in range(n_states):
                all_sampled_outcomes[i, :] = np.random.choice(
                    a=num_outcomes, size=statistics, p=probabilities[i]
                )
            return all_sampled_outcomes # Shape (n_states, statistics)
    elif sampling_method == 'poisson':
        if not return_frequencies:
            raise ValueError("Cannot return raw outcomes when using Poisson sampling. Set return_frequencies=True.")

        # Calculate expected counts (lambda for Poisson)
        # Shape: (n_states, num_outcomes)
        expected_counts = statistics * probabilities

        # Sample from Poisson distribution
        # Shape: (n_states, num_outcomes)
        poisson_counts = np.random.poisson(lam=expected_counts)

        # Calculate frequencies
        # Avoid division by zero if statistics is 0 (although handled earlier, defensive check)
        frequencies = poisson_counts / statistics
        return frequencies # return shape (num_outcomes, n_states)

    else:
        raise ValueError(f"Unknown sampling method: '{sampling_method}'. Use 'standard' or 'poisson'.")



def measure_povm_np(
    states: ArrayLike,
    povm: ArrayLike,
    statistics: float,
    return_frequencies: bool = True,
    sampling_method: SamplingMethodType = 'standard'
) -> NDArray:
    """
    Simulates measuring a POVM on one or more quantum states.
    This function operates on the raw numpy arrays. If you want fancies stuff use QuantumOperatorsBatch.measure_povm() etc instead.

    Parameters
    ----------
    states : ArrayLike
        One or more quantum states.
        If it's a 1D array, it is treated as a single ket state.
        If it's a 2D array, each row is assumed to be a ket vector.
        If it's a 3D array, each row is assumed to be a density matrix.
    povm : ArrayLike
        A POVM represented as a list of effects as a 3D array (shape: n_povm_outcomes x dim x dim).
    statistics : float
        Number of measurement shots PER STATE. Use np.inf for exact probabilities.
        Finite non-integer values will raise an error.
    return_frequencies : bool, optional
        - If True: Return measurement frequencies for each state.
          Shape: (n_states, n_povm_outcomes).
        - If False (default) and sampling_method='standard': Return raw sampled
          outcome indices for each state. Shape: (n_states, statistics).
        - If False and sampling_method='poisson': Raises ValueError.
    sampling_method : str, optional
        - 'standard': Sample outcomes using multinomial distribution based on
          exact probabilities for the requested number of shots (statistics).
        - 'poisson': Sample counts for each outcome using a Poisson distribution
          with mean = statistics * probability. Requires return_frequencies=True.
        (default: 'standard').

    Returns
    -------
    NDArray
        Either the raw measurement outcomes or the outcome frequencies,
        depending on `return_frequencies` and `sampling_method`. See parameter
        descriptions for exact shapes.
    """
    povm = np.asarray(povm, dtype=np.complex128)
    if povm.ndim != 3 or povm.shape[1] != povm.shape[2]:
        raise ValueError(f"POVM must be a 3D array (shape: n_povm_outcomes x dim x dim). Got {povm.ndim} dimensions.")
    
    states = np.asarray(states, dtype=np.complex128)
    if states.ndim == 1:
        # If states is a 1D array, we assume it's a single ket state
        states = states.reshape(1, -1)
    if states.ndim == 2:
        # If states is a 2D array, we assume each row is a ket vector
        probabilities = expvals_from_kets_and_observables_numpy(states, povm)
    elif states.ndim == 3:
        # If states is a 3D array, we assume each row is a density matrix
        probabilities = expvals_from_dms_and_observables_numpy(states, povm)
    else:
        raise ValueError(f"States must be a 1D array (single ket), a 2D array (multiple kets), or a 3D array (density matrices). Got {states.ndim} dimensions.")

    # Check statistics value (return probabilities if np.inf, round non-inf floats)
    return sample_from_probabilities(
        probabilities.T,
        statistics,
        return_frequencies=return_frequencies,
        sampling_method=sampling_method
    ).T

def measure_povm(
    states: Union[QuantumState, QuantumStatesBatch, ArrayLike],
    povm: Union[POVM, ArrayLike],
    statistics: float,
    return_frequencies: bool = True,
    sampling_method: SamplingMethodType = 'standard'
) -> NDArray:
    """Measure a POVM on one or more quantum states.
    
    General interface that works with both QuantumStatesBatch and raw numpy arrays.
    """
    if isinstance(states, QuantumStatesBatch):
        states_ = states.data
    elif isinstance(states, QuantumState):
        states_ = states.asbatch().data
    else:
        states_ = np.asarray(states, dtype=np.complex128)

    if isinstance(povm, POVM):
        povm_ = povm.data
    else:
        povm_ = np.asarray(povm, dtype=np.complex128)
    # do the thing
    return measure_povm_np(states_, povm_, statistics, return_frequencies, sampling_method)


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