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


# class representing POVM (as a list of qutip.Qobj) and the associated label, for example, "SIC", "MUB", etc.
class POVM:
    """
    Class representing a POVM.

    Accepts various input types for POVM elements (sequences of qutip.Qobj or numpy arrays),
    but internally converts and stores them as a list of qutip.Qobj objects.
    It also stores an associated label, useful for plotting or identification purposes.
    
    Attributes:
        elements (List[qutip.Qobj]): A list of POVM elements as qutip.Qobj objects.
        label (str): An optional label for the POVM.
        num_outcomes (int): Number of outcomes. Inferred from the length of the POVM.
    """
    elements: List[qutip.Qobj]  # The actual stored type
    label: str
    num_outcomes: int
    
    def __init__(self, elements: POVMType, label: Optional[str] = None):
        """
        Initialize a POVM with the given elements.

        Parameters:
            elements (POVMType): Input POVM elements, which can be:
                - A sequence of qutip.Qobj objects
                - A sequence of numpy.ndarray objects (will be converted to qutip.Qobj)
                - Another POVM object
            label (Optional[str], optional): A label for the POVM. Defaults to a generated label.

        Note:
            Although the input can be of various types, after initialization `self.elements` 
            will always be a list of qutip.Qobj objects.
        
        Raises:
            ValueError: If any element is neither a qutip.Qobj nor a numpy.ndarray.
        """
        # Generate default label if none provided
        self.num_outcomes = len(elements)
        if label is None:
            label = f"unknown, {len(elements)} outcomes"
        
        # Process the POVM elements
        processed_povm = []
        for op in elements:
            if not isinstance(op, (qutip.Qobj, np.ndarray)):
                raise ValueError("POVM elements must be qutip.Qobj or numpy.ndarray.")
            if isinstance(op, np.ndarray):
                op = qutip.Qobj(op)
            processed_povm.append(op)
        
        self.elements = processed_povm
        self.label = label
    
    def __iter__(self):
        return iter(self.elements)
    
    def __len__(self):
        return len(self.elements)
    
    def __getitem__(self, index):
        return self.elements[index]


# --- Type Definitions ---
POVMElement = Union[qutip.Qobj, NDArray[np.complex128]]
POVMType = Union[Sequence[POVMElement], POVM]  # Allow sequences or the POVM class