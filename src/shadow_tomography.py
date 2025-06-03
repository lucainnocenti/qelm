from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, Iterable, List, Literal, Optional, Sequence, Union

import numpy as np
import qutip

from src.POVM import POVM
from src.types import BasisType, RescalingInput
# from src.quantum_utils import QuantumState    

def frame_operator(
    povm: POVM,
    basis: BasisType = "pauli",
    rescaling: RescalingInput = "none",
) -> np.ndarray:
    """
    Compute the frame operator
       F = sum_b α_b^2 |v_b><v_b|,

    where each v_b is the vectorized POVM element in the chosen basis,
    and α_b are rescaling factors.

    Supports:
      - basis="pauli"   : uses the n-qubit Pauli basis (real coefficients)
      - basis="flatten" : simple row-major flattening (complex coefficients)
      - basis=custom    : any user-provided list of d² matrices

    Rescaling options:
      - "none"  : α_b = 1
      - "trace" : α_b = Tr(E_b)/d
      - [f₁,f₂…] : custom list of floats, length = number of outcomes
    """
    # 1) Extract the list of Qobj elements and check non-empty
    # elems = povm.elements
    if not povm.elements:
        raise ValueError("POVM must contain at least one element.")

    # 2) Figure out the dimension `d` and the vector dimension `d²`
    d = QuantumState(povm.elements[0]).dimension
    vec_dim = d * d
    n_outcomes = len(povm.elements)  # number of POVM outcomes

    # 3) Compute rescaling factors α_b
    if isinstance(rescaling, list):
        # custom list must match the number of outcomes
        if len(rescaling) != n_outcomes:
            raise ValueError("Length of custom rescaling must equal POVM size.")
        alpha = np.array(rescaling, dtype=float)

    elif rescaling == "trace":
        # α_b = Tr(E_b)/d  (use real part of the trace)
        alpha = np.array([E.full().trace().real for E in povm.elements]) / d

    elif rescaling == "none":
        # α_b = 1 for all b
        alpha = np.ones(n_outcomes, dtype=float)

    else:
        raise ValueError(f"Unknown rescaling option {rescaling!r}.")

    # 5) Vectorize each POVM element into columns of V (shape = d²×n_outcomes)
    #    We allocate a complex buffer and let QuantumState.vectorise
    #    cast to real if needed.
    V = np.empty((vec_dim, n_outcomes), dtype=complex)
    for i, E in enumerate(povm.elements):
        V[:, i] = QuantumState(E).vectorise(basis=basis)

    # 6) Build the frame operator F = ∑ α_b² · v_b·v_b†
    #    Start from zeros of the correct dtype
    F = np.zeros((vec_dim, vec_dim), dtype=complex if np.iscomplexobj(V) else float)

    #    Loop over each outcome, form the outer product
    for b in range(n_outcomes):
        v = V[:, b : b+1]       # column vector shape (d²,1)
        F += (alpha[b] ** 2) * (v @ v.conj().T)

    return F


def shadow_estimator(
    povm: POVM,
    basis: BasisType = "pauli",
    rescaling: RescalingInput = "none",
) -> List[np.ndarray]:
    """
    Compute the dual (shadow) POVM elements Ẽ_b so that

        vec(Ẽ_b) = F^{-1}[ α_b² · vec(E_b) ].

    Returns a list of NumPy arrays for the dual POVM elements.
    """
    if not povm.elements:
        raise ValueError("POVM must contain at least one element.")

    d = QuantumState(povm.elements[0]).dimension
    n_outcomes = len(povm.elements)

    # 1) Compute frame operator F and its pseudo-inverse F⁻¹
    F     = frame_operator(povm, basis=basis, rescaling=rescaling)
    F_inv = np.linalg.pinv(F)

    # 2) Recompute α_b exactly as in frame_operator
    if isinstance(rescaling, list):
        alpha = np.array(rescaling, dtype=float)
    elif rescaling == "trace":
        alpha = np.array([E.full().trace().real for E in povm.elements]) / d
    elif rescaling == "none":
        alpha = np.ones(n_outcomes, dtype=float)
    else:
        raise ValueError(f"Unknown rescaling option {rescaling!r}.")


    # 4) Vectorise original POVM into V (d²×k)
    V = np.empty((d*d, n_outcomes), dtype=complex)
    for i, E in enumerate(povm.elements):
        V[:, i] = QuantumState(E).vectorise(basis=basis)

    # 5) Compute dual vectors W[:,b] = F⁻¹ [ α_b² · V[:,b] ]
    W = np.empty_like(V, dtype=complex)
    for b in range(n_outcomes):
        W[:, b] = F_inv @ (alpha[b] ** 2 * V[:, b])

    # 6) Un-vectorise each dual vector back into a d×d matrix
    duals: List[np.ndarray] = []
    for b in range(n_outcomes):
        qs = QuantumState.from_vector(
            W[:, b],
            dimension=d,
            basis=basis
        )
        duals.append(qs.matrix)

    return duals