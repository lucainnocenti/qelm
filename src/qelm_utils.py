import numpy as np
import qutip

import src.QELM as QELM
from src.quantum_utils import measure_povm

from numpy.typing import NDArray
from typing import Optional, Union, Dict, Any, Iterable


def make_train_dict_from_states(
    training_states: Iterable[qutip.Qobj],
    target_observables: Iterable[qutip.Qobj],
    povm: Iterable[qutip.Qobj],
    statistics: int
) -> Dict[str, NDArray[np.float64]]:
    """
    Create a training dictionary from given states and labels.

    Parameters
    ----------
    training_states : np.ndarray
        Array of quantum states.
    target_observables : np.ndarray
        Corresponding labels for the states.
    povm : np.ndarray
        POVM operators to be used for measurement.
    statistics : int
        Number of measurements to perform PER STATE.
        Meaning the number of samples to be drawn from the POVM, FOR EACH STATE.
    
    Returns
    --------
    Dict[str, NDArray[np.float64]]
        Dictionary containing the training data.
        The keys are 'frequencies' and 'labels'.
    """
    # ensure iterables are lists
    training_states = list(training_states)
    target_observables = list(target_observables)
    povm = list(povm)

    # measure the POVM on the training states
    frequencies = measure_povm(
        state=training_states,
        povm=povm,
        statistics=statistics,
        return_frequencies=True
    )

    train_expvals = np.zeros((len(target_observables), len(training_states)), dtype=np.float64)
    for idx, observable in enumerate(target_observables):
        expvals = np.array([qutip.expect(observable, state) for state in training_states]).real
        train_expvals[idx] = expvals

    train_dict = {
        'frequencies': frequencies,
        'labels': train_expvals
    }

    return train_dict



def train_qelm_from_states_for_observables(
    training_states: Iterable[qutip.Qobj],
    target_observables: Iterable[qutip.Qobj],
    povm: Iterable[qutip.Qobj],
    statistics: Union[int, list[int]],
    method: str = 'standard',
    train_options: Optional[dict] = None,
    test_states: Optional[Iterable[qutip.Qobj]] = None
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
    # ensure iterables are lists
    training_states = list(training_states)
    target_observables = list(target_observables)
    povm = list(povm)
    if test_states is not None:
        test_states = list(test_states)
    if isinstance(statistics, list):
        # if statistics is a list, it must have two elements
        if len(statistics) != 2:
            raise ValueError("If statistics is a list, it must have two elements (for train and for test).")
        # test states must be provided in this case
        if test_states is None:
            raise ValueError("If statistics is a list, test states must be provided.")
        train_stat = statistics[0]
        test_stat = statistics[1]
    else:
        train_stat = statistics
        test_stat = statistics


    train_dict = make_train_dict_from_states(
        training_states=training_states,
        target_observables=target_observables,
        povm=povm,
        statistics=train_stat
    )
    # Initialize and train the QELM model

    if test_states is not None:
        test_dict = make_train_dict_from_states(
            training_states=test_states,
            target_observables=target_observables,
            povm=povm,
            statistics=test_stat
        )
        qelm_model = QELM.QELM(train_dict=train_dict, test_dict=test_dict, method=method,
                               train_options=train_options)
    else:
        qelm_model = QELM.QELM(train_dict=train_dict, method=method, train_options=train_options)
    
    return qelm_model