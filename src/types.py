import numpy as np
import qutip
from dataclasses import dataclass
import logging

# --- Type Hinting Setup ---
from numpy.typing import NDArray
from typing import Optional, Union, Dict, Any, Iterable, List, Sequence, Tuple



# --- Data Classes ---
@dataclass
class TrainingData:
    """Holds the training frequencies and corresponding labels.
    
    Attributes
    ----------
    frequencies : NDArray[np.float64]
        Measurement frequencies for each training state and POVM outcome.
    labels : NDArray[np.float64]
        Exact expectation values of target observables for each training state.
    """
    frequencies: Optional[NDArray[np.float64]] = None
    labels: Optional[NDArray[np.float64]] = None
    states: Optional[List[NDArray[np.complex128]]] = None

    @property
    def can_train(self) -> bool:
        """Check if the training data is complete and ready for training.
        
        Logs information about what data is missing if the data is incomplete.
        """
        missing = []
        if self.frequencies is None:
            missing.append("frequencies")
        if self.labels is None:
            missing.append("labels")
            
        if missing:
            logging.info(f"Cannot train: missing {' and '.join(missing)} data.")
            return False
        return True