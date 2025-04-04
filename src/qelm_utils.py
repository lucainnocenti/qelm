from typing import Optional, Union, Dict, Any

import qutip
import src.QELM as QELM
from src.quantum_utils import measure_povm
import numpy as np


def train_qelm_from_states_for_observables(
    training_states: list[qutip.Qobj],
    target_observables: list[qutip.Qobj],
    povm: list[qutip.Qobj],
    statistics: int,
    method: str = 'standard',
    train_options: Optional[dict] = None
) -> QELM.QELM:
    """
    Train a QELM model from given states and labels.

    Parameters
    ----------
    training_states : np.ndarray
        Array of quantum states.
    labels : np.ndarray
        Corresponding labels for the states.
    method : str, optional
        Method to be used for training. Default is 'standard'.
    train_options : dict, optional
        Additional options for training.
        These are passed to the QELM class.

    Returns
    --------
    QELM.QELM
        Trained QELM model.
    """

    # measure the POVM on the training states
    frequencies = measure_povm(
        state=training_states,
        povm=povm,
        statistics=statistics,
        return_frequencies=True
    )

    target_expvals = np.zeros((len(target_observables), len(training_states)), dtype=np.float64)
    for idx, observable in enumerate(target_observables):
        expvals = np.array([qutip.expect(observable, state) for state in training_states]).real
        target_expvals[idx] = expvals

    train_dict = {
        'frequencies': frequencies,
        'labels': target_expvals
    }
    # print(train_dict['frequencies'].shape)
    # print(train_dict['labels'])

    # Initialize and train the QELM model
    qelm_model = QELM.QELM(train_dict=train_dict, method=method, train_options=train_options)
    
    return qelm_model