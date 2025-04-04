import qutip # type: ignore
import numpy as np
import numpy.typing as npt
import typing
from typing import List, Union, Iterable, Optional

def measure_povm(
        state: Union[qutip.Qobj, Iterable[qutip.Qobj]],
        povm: Iterable[qutip.Qobj],
        statistics: Union[int, float],
        return_frequencies: bool = False
    ) -> npt.NDArray:
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
    statistics : int
        The number of measurements to perform.
    return_frequencies : bool, optional
        If True, return the frequencies instead of the raw outcomes.
        Default is False.
    """
    # make povm a list if it is not already
    if not isinstance(povm, list):
        povm = list(povm)

    # Check if the POVM is valid
    if not all(op.isherm for op in povm):
        raise ValueError("All POVM operators must be Hermitian.")
    
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
    sampled_outcomes = np.random.choice(a=len(povm), size=int(statistics), p=probabilities)
    

    if return_frequencies:
        # return the frequencies of each outcome
        frequencies = np.bincount(sampled_outcomes, minlength=len(povm)) / statistics
        return frequencies
    # otherwise, return the sampled outcomes
    return sampled_outcomes

# define a function that returns the single-qubit SIC-POVM
def sic_povm() -> List[qutip.Qobj]:
    """
    Get the single-qubit SIC-POVM.

    Returns:
    --------
    List[qutip.Qobj]
        A list of qutip.Qobj representing the POVM operators.
    """
    # Define the SIC-POVM operators for a single qubit
    povm = [
        qutip.Qobj([[1, 0], [0, 0]]) / 2,
        qutip.ket2dm(qutip.Qobj([1/np.sqrt(3), np.sqrt(2/3)])) / 2,
        qutip.ket2dm(qutip.Qobj([1/np.sqrt(3), np.sqrt(2/3) * np.exp(2 * np.pi * 1j / 3)])) / 2,
        qutip.ket2dm(qutip.Qobj([1/np.sqrt(3), np.sqrt(2/3) * np.exp(4 * np.pi * 1j / 3)])) / 2
    ]
    return povm

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
    return [op / 3 for op in ops]

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
    
    return povm