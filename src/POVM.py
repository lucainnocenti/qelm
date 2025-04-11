"""
Module defining the POVM class.
This class encapsulates a list of qutip.Qobj (POVM elements) and an associated label.

The class provides methods for initialization, string representation, and iteration over the POVM elements.
The main goal is handle labels and qutip.Qobj objects in a consistent manner, mostly for plotting purposes.
"""

from typing import Iterable, Union, Optional
import numpy as np
from numpy.typing import NDArray
import qutip

# a class that contains a POVM (as a list of qutip.Qobj) and the associated label, for example, "SIC", "MUB", etc.
# when converted to a string, it should return the label of the POVM
class POVM:
    def __init__(self, povm: Iterable[Union[qutip.Qobj, NDArray]], label: Optional[str] = None):
        """Initialize the POVM with a list of operators and a label.
        
        We process the input to ensure we have a list of qutip.Qobj.
        """
        if not isinstance(povm, list):
            povm = list(povm)
        if label is None:
            num_outcomes = len(povm)
            label = f"unknown, {num_outcomes} outcomes"
        # ensure that the POVM is a list of qutip.Qobj
        new_povm = []
        for op in povm:
            if not isinstance(op, (qutip.Qobj, np.ndarray)):
                raise ValueError("POVM elements must be qutip.Qobj or numpy.ndarray.")
            if isinstance(op, np.ndarray):
                op = qutip.Qobj(op)
            new_povm.append(op)
        self.povm = new_povm
        self.label = label

    def __str__(self):
        return self.label

    def __repr__(self):
        return repr(self.povm)
    
    # if one tries to convert the POVM to a list, it should return the list of operators
    def __iter__(self):
        return iter(self.povm)
    # doing len(povm) should return the number of outcomes
    def __len__(self):
        return len(self.povm)
    
    def __getitem__(self, index):
        # enable subscriptable access to the POVM elements
        return self.povm[index]
