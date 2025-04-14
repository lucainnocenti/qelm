"""
Module defining the POVM class.
This class encapsulates a list of qutip.Qobj (POVM elements) and an associated label.

The class provides methods for initialization, string representation, and iteration over the POVM elements.
The main goal is handle labels and qutip.Qobj objects in a consistent manner, mostly for plotting purposes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Union, Optional
import numpy as np
from numpy.typing import NDArray
import qutip




# a dataclass that contains a POVM (as a list of qutip.Qobj) and the associated label, for example, "SIC", "MUB", etc.
@dataclass
class POVM:
    """
    Class representing a POVM.

    This class encapsulates a list of POVM elements (qutip.Qobj or numpy.ndarray) and an associated label.
    It ensures that the POVM elements are a list of qutip.Qobj objects, and provides a default label if none is provided.
    
    Attributes:
        elements (POVMType): A sequence of POVM elements (qutip.Qobj or numpy.ndarray).
        label (Optional[str]): An optional label for the POVM.
        num_outcomes (Optional[int]): Number of outcomes. If known, otherwise inferred from the length of the POVM.
    """
    elements: POVMType
    label: Optional[str] = None
    num_outcomes: Optional[int] = None  # Number of outcomes, if known
    
    def __post_init__(self):
        """Process the raw POVM input after initialization."""
        # Generate default label if none provided
        if self.label is None:
            self.label = f"unknown, {len(self.elements)} outcomes"
        if self.num_outcomes is None:
            self.num_outcomes = len(self.elements)
        
        # Process the POVM elements
        processed_povm = []
        for op in self.elements:
            if not isinstance(op, (qutip.Qobj, np.ndarray)):
                raise ValueError("POVM elements must be qutip.Qobj or numpy.ndarray.")
            if isinstance(op, np.ndarray):
                op = qutip.Qobj(op)
            processed_povm.append(op)
        
        self.elements = processed_povm  # Replace with processed version
    
    # make it easier to work with the POVM in a for loop making it seqeuence-like
    def __iter__(self):
        return iter(self.elements)
    def __len__(self):
        return len(self.elements)
    def __getitem__(self, index):
        return self.elements[index]



# --- Type Definitions ---
POVMElement = Union[qutip.Qobj, NDArray[np.complex128]]
POVMType = Union[Sequence[POVMElement], POVM]  # Allow sequences or the POVM class