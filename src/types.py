from __future__ import annotations
import numpy as np
import qutip
from dataclasses import dataclass

# Logger setup
import logging
# --- Type Hinting Setup ---
from numpy.typing import NDArray
from typing import Optional, Union, Dict, Any, Iterable, List, Sequence, Tuple

from src.POVM import POVMType
from src.quantum_utils import measure_povm

# --- Data Classes ---
@dataclass
class TrainingData:
    """Holds the training frequencies and corresponding labels."""
    frequencies: Optional[NDArray[np.float64]] = None
    labels: Optional[NDArray[np.float64]] = None
    states: Optional[Sequence[qutip.Qobj]] = None
    observables: Optional[Sequence[qutip.Qobj]] = None
    statistics: Optional[float] = None
    sampling_method: Optional[str] = None
    povm: Optional[POVMType] = None

    @property
    def can_train(self) -> bool:
        """Check if the necessary data (frequencies, labels) is present."""
        missing = []
        if self.frequencies is None:
            missing.append("frequencies")
        if self.labels is None:
            missing.append("labels")
            
        if missing:
            logging.debug(f"Cannot train: missing {' and '.join(missing)} data.")
            return False
        return True
    
    @classmethod
    def from_states_and_observables(
        cls,
        states: Sequence[qutip.Qobj],
        observables: Sequence[qutip.Qobj],
        povm: POVMType,
        statistics: float,
        sampling_method: str = 'standard'
        # Add save_states: bool = False if you want to store states_qobj
    ) -> TrainingData:
        """
        Factory method to generate TrainingData from quantum states and observables.

        Computes expectation values from given states and target observables,
        and generates measurement frequencies using the provided POVM.

        Parameters
        ----------
        states : Sequence[qutip.Qobj]
            Quantum states (density matrices) to use.
        observables : Sequence[qutip.Qobj]
            Observables whose expectation values are the target labels.
        povm : POVMType
            POVM operators for measurement.
        statistics : float
            Measurement shots PER STATE (np.inf for exact probabilities).
        sampling_method : str
             Method for sampling measurement outcomes ('standard' or 'poisson').

        Returns
        -------
        TrainingData
            An instance containing generated 'frequencies' and 'labels'.
        """
        n_states = len(states)
        n_observables = len(observables)

        if n_states == 0:
            raise ValueError("Input states list cannot be empty.")
        if n_observables == 0:
            raise ValueError("Target observables list cannot be empty.")
        if statistics <= 0 and statistics != np.inf :
            raise ValueError("Statistics must be positive or np.inf.")

        logging.debug(f"Generating training data for {n_states} states, {n_observables} observables, {statistics} shots/state.")

        # Measure POVM -> frequencies (n_states, n_outcomes)
        frequencies: NDArray[np.float64] = measure_povm(
            states=states,
            povm=povm,
            statistics=statistics,
            return_frequencies=True,
            sampling_method=sampling_method
        )

        # Calculate exact expectation values -> labels (n_observables, n_states)
        labels = np.zeros((n_observables, n_states), dtype=np.float64)
        for idx, observable in enumerate(observables):
            expvals = qutip.expect(observable, states)
            # make sure expvals is a numpy array for typing purposes
            if not isinstance(expvals, np.ndarray):
                expvals = np.array(expvals)
            # Qutip expect can return complex near zero for Hermitian op/state due to numerics
            if np.any(np.abs(expvals.imag) > 1e-10): 
                logging.warning(f"Non-negligible imaginary part found in expectation values for observable {idx}. Taking real part.")
            labels[idx] = expvals.real
            
        # Create and return the TrainingData instance
        trainingdata = cls(
            frequencies=frequencies, 
            labels=labels
        )
        # Store additional information about the training data
        trainingdata.povm = povm
        trainingdata.statistics = statistics
        trainingdata.sampling_method = sampling_method
        trainingdata.states = states
        trainingdata.observables = observables
        # Return the instance
        return trainingdata