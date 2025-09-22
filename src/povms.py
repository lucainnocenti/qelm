"""
Module defining the POVM class.
This class encapsulates a list of qutip.Qobj (POVM elements) and an associated label.

The class provides methods for initialization, string representation, and iteration over the POVM elements.
The main goal is handle labels and qutip.Qobj objects in a consistent manner, mostly for plotting purposes.
"""
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Sequence, Union, Optional, List
import numpy as np
from numpy.typing import NDArray
import qutip

from src.quantum_utils import POVM, ket2dm


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
        np.array([[1, 0], [0, 0]]) / 2,
        ket2dm([1/np.sqrt(3), np.sqrt(2/3)]) / 2,
        ket2dm([1/np.sqrt(3), np.sqrt(2/3) * np.exp(2 * np.pi * 1j / 3)]) / 2,
        ket2dm([1/np.sqrt(3), np.sqrt(2/3) * np.exp(4 * np.pi * 1j / 3)]) / 2
    ]
    return POVM(povm, label="SIC")

def mub_povm() -> POVM:
    """
    Generate a single-qubit POVM consisting of the projections over the eigenstates of the three Pauli matrices.
    The resulting POVM is a list of 6 rank-1 projectors.
    """
    ops = [
        ket2dm([1, 0]),
        ket2dm([0, 1]),
        ket2dm([1, 1]) / 2,
        ket2dm([1, -1]) / 2,
        ket2dm([1, 1j]) / 2,
        ket2dm([1, -1j]) / 2
    ]
    normalized_povm = [op / 3 for op in ops]
    return POVM(normalized_povm, label="MUB")

def random_rank1_povm(dim: int, num_outcomes: int, seed: Optional[int] = None) -> POVM:
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
        random_unitary = qutip.rand_unitary(num_outcomes, seed=seed).full()[:, :dim]
    else:
        random_unitary = qutip.rand_unitary(num_outcomes).full()[:, :dim]
    povm = [np.outer(row, row.conj()) for row in random_unitary]

    return POVM(povm, label="Random rank-1, {} outcomes".format(num_outcomes))