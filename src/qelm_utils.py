import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import qutip

import src.QELM as QELM
from src.quantum_utils import measure_povm, average_rho, average_rho_tensor_rho
from src.POVM import POVM

from numpy.typing import NDArray
from typing import Optional, Union, Dict, Any, Iterable, List


def make_train_dict_from_states(
    training_states: Iterable[qutip.Qobj],
    target_observables: Iterable[qutip.Qobj],
    povm: Union[Iterable[qutip.Qobj], POVM],
    statistics: float
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
    statistics : int or float
        Number of measurements to perform PER STATE.
        Meaning the number of samples to be drawn from the POVM, FOR EACH STATE.
        The float is to accept the case of infinite statistics via np.inf.
    
    Returns
    --------
    Dict[str, NDArray[np.float64]]
        Dictionary containing the training data.
        The keys are 'frequencies' and 'labels'.
    """
    # ensure iterables are lists
    training_states = list(training_states)
    target_observables = list(target_observables)

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
    povm: Union[Iterable[qutip.Qobj], POVM],
    statistics: Union[float, list[float]],
    method: str = 'standard',
    train_options: dict = {},
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


# =================================================
# FUNCTIONS FOR ESTIMATOR ANALYSIS
# =================================================


def exact_average_estimator_variance(
    estimator: Iterable[float],
    povm: Union[Iterable[Union[qutip.Qobj, NDArray]], POVM]
) -> float:
    r"""
    Computes the *single-shot* variance of the given estimator, averaged over all pure test states.

    This implements the formula
    Var[\hat o] = E_\rho[\sum-a Tr[\mu_a \rho] \hat o(a)^2 - (\sum_a Tr[\mu_a \rho] \hat o(a))^2]
    with the average taken over all pure states.

    Parameters:
    estimator (Iterable[float]): The estimator values.
    povm (Iterable[qutip.Qobj]): The POVM elements.

    Returns:
    float: The variance of the average estimator.
    """
    # ensure iterables are lists
    estimator = list(estimator)
    povm = list(povm)
    povm_np: List[np.ndarray] = [
        p.full() if isinstance(p, qutip.Qobj) else p
        for p in povm
    ]


    dim = povm_np[0].shape[0]
    num_outcomes = len(povm_np)

    avg_rho1 = average_rho(d=dim)
    avg_rho2 = average_rho_tensor_rho(d=dim)

    # compute first term of the variance
    # this is the sum over b of Tr[mu_b \rho] o(b)^2, with \rho replaced by its average I/dim
    term1 = 0
    for b in range(num_outcomes):
        term1 += estimator[b]**2 * np.trace(np.dot(povm_np[b], avg_rho1)).real

    # compute second term of the variance
    # this is the sum over b of Tr[(mu_a\otimes\mu_b) (\rho\otimes\rho)] o(a) o(b), with \rho\otimes\rho replaced by its average
    term2 = 0
    for a in range(num_outcomes):
        for b in range(num_outcomes):
            term2 += estimator[a] * estimator[b] * np.trace(np.dot(np.kron(povm_np[a], povm_np[b]), avg_rho2)).real
    
    return term1 - term2

def exact_average_bias2(
    estimator: Iterable[float],
    povm: Union[Iterable[Union[qutip.Qobj, NDArray]], POVM],
    target_observable: Union[qutip.Qobj, NDArray],
) -> float:
    r"""
    Computes the *single-shot* bias squared, averaged over all pure test states.

    This implements the formula
    Bias[\hat o]^2 = (\sum_a Tr[\mu_a \rho] \hat o(a) - Tr[O \rho])^2
    with the average taken over all pure states.

    Parameters:
        estimator (Iterable[float]): The estimator values.
        povm (Iterable[qutip.Qobj]): The POVM elements.

    Returns:
        float: The average bias2 of the estimator.
    """
    # ensure iterables are lists and all objects are numpy arrays
    estimator = list(estimator)
    povm = list(povm)
    povm_np: List[np.ndarray] = [
        p.full() if isinstance(p, qutip.Qobj) else p
        for p in povm
    ]
    obs_np = target_observable.full() if isinstance(target_observable, qutip.Qobj) else target_observable

    dim = povm_np[0].shape[0]
    num_outcomes = len(povm_np)

    avg_rho1 = average_rho(d=dim)
    avg_rho2 = average_rho_tensor_rho(d=dim)

    # compute first term of the variance
    # this is the sum over b of Tr[mu_b \rho] o(b), with \rho replaced by its average I/dim
    term1 = 0
    for a in range(num_outcomes):
        for b in range(num_outcomes):
            term1 += estimator[a] * estimator[b] * np.trace(np.dot(np.kron(povm_np[a], povm_np[b]), avg_rho2)).real

    term2 = 0
    for a in range(num_outcomes):
        term2 += estimator[a] * np.trace(np.dot(np.kron(povm_np[a], obs_np), avg_rho2)).real
    
    term3 = np.trace(np.dot(np.kron(obs_np, obs_np), avg_rho2)).real
    
    return term1 - 2 * term2 + term3


# =================================================
# PLOTTING FUNCTIONS
# =================================================

def biasvar_vs_nstates(
    povm: POVM,
    target_observable: Union[qutip.Qobj, NDArray],
    n_states_list: Iterable[int],
    n_realizations : int,
    statistics: float,
    test_statistics: int = 1,
    quantiles: List[float] = [0.25, 0.75],
    fix_total_statistics : bool = True,
    train_options: Dict[str, Any] = {}
) -> dict:
    r"""
    Computes the bias^2 and variance of the estimator for different numbers of states.

    Produces a plot of computed quantities against the number of states, with median and quantiles computed
    over multiple realizations.
    Each realization resamples the training states and the samples from the POVM, but the POVM is fixed beforehand.
    
    Args:
        povm (Iterable[Union[qutip.Qobj, NDArray]]): The POVM elements.
        target_observable (Union[qutip.Qobj, NDArray]): The target observable.
        n_states_list (Iterable[int]): List of numbers of states to test.
        n_realizations (int): Number of realizations to average over.
        quantiles (Iterable[float]): Quantiles to compute when plotting results.
        statistics (int): Total TRAINING statistics.
        fix_total_statistics (bool): If True, the total statistics is fixed and the number of samples per state is adjusted accordingly. If False, each training state uses the provided statistics regardless of n_states.

    Returns:
        A dictionary containing the computed bias^2, variance, and MSE for each number of states.
    """
    # ensure iterables are lists and all objects are qutip objects
    obs = qutip.Qobj(target_observable) if not isinstance(target_observable, qutip.Qobj) else target_observable
    n_states_list = list(n_states_list)
    # extract state dimension
    dim = povm[0].shape[0]

    var_results = {}
    bias2_results = {}
    mse_results = {}

    # ---- compute the bias and variance for each number of states ----
    for n_states in tqdm(n_states_list, desc='n_states loop'):
        var_for_states = []
        bias2_for_states = []
        mse_for_states = []

        for _ in tqdm(range(n_realizations), desc='Realization loop', leave=False):
            # sample random training states
            train_states = [qutip.ket2dm(qutip.rand_ket(dim, 1)) for _ in range(n_states)]
            # compute traing statistics, dividing by the number of states if so required
            if fix_total_statistics:
                train_stat = statistics / n_states
            else:
                train_stat = statistics
            # train the qelm
            qelm = train_qelm_from_states_for_observables(
                training_states=train_states,
                target_observables=[obs],
                povm=povm,
                statistics=train_stat,
                train_options=train_options
            )
            if qelm.w is None:
                raise ValueError('Model is not trained, or some fuckery is going on (`w` is not set).')
            estimator = qelm.w[0]

            bias2 = exact_average_bias2(estimator=estimator, povm=povm, target_observable=obs)
            var = exact_average_estimator_variance(estimator=estimator, povm=povm)
            # the given test statistics is only used to divide the variance
            var = var / test_statistics
            # build the list of results for this specific n_states
            var_for_states.append(var)
            bias2_for_states.append(bias2)
            mse_for_states.append(var + bias2)
        # build the dictionaries containing all computed results
        var_results[n_states] = var_for_states
        bias2_results[n_states] = bias2_for_states
        mse_results[n_states] = mse_for_states
    # ---- compute the stuff for the plots ----

    # compute the median and quantiles for each number of states
    median_var = [np.median(var_results[n_states]) for n_states in n_states_list]
    lower_var = [np.quantile(var_results[n_states], quantiles[0]) for n_states in n_states_list]
    upper_var = [np.quantile(var_results[n_states], quantiles[1]) for n_states in n_states_list]

    median_bias2 = [np.median(bias2_results[n_states]) for n_states in n_states_list]
    lower_bias2 = [np.quantile(bias2_results[n_states], quantiles[0]) for n_states in n_states_list]
    upper_bias2 = [np.quantile(bias2_results[n_states], quantiles[1]) for n_states in n_states_list]

    median_mse = [np.median(mse_results[n_states]) for n_states in n_states_list]
    lower_mse = [np.quantile(mse_results[n_states], quantiles[0]) for n_states in n_states_list]
    upper_mse = [np.quantile(mse_results[n_states], quantiles[1]) for n_states in n_states_list]

    # ---- plot the results ----
    plt.figure(figsize=(10, 6))


    # Plot Variance with quantile error shading
    plt.fill_between(n_states_list, lower_var, upper_var, color='blue', alpha=0.3, label='Variance {}-{} quantiles'.format(quantiles[0], quantiles[1]))
    plt.plot(n_states_list, median_var, marker='o', color='blue', label='Median variance')

    # Plot Bias² with quantile error shading
    plt.fill_between(n_states_list, lower_bias2, upper_bias2, color='green', alpha=0.3, label='Bias² {}-{} quantiles'.format(quantiles[0], quantiles[1]))
    plt.plot(n_states_list, median_bias2, marker='o', color='green', label='Median bias²')

    # Plot MSE with quantile error shading
    plt.fill_between(n_states_list, lower_mse, upper_mse, color='red', alpha=0.3, label='MSE {}-{} quantiles'.format(quantiles[0], quantiles[1]))
    plt.plot(n_states_list, median_mse, marker='o', color='red', label='Median MSE')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of training states')
    # plt.ylabel('Metric Value')
    plt.title('MSE, Var, Bias² vs number of training states\nTotal training statistics: {}; {} realizations; POVM: {}'.format(statistics, n_realizations, str(povm)))
    plt.grid(True)
    plt.legend()
    plt.show()

    return {'var': var_results, 'bias2': bias2_results, 'mse': mse_results}


def biasvar_vs_statistics(
    povm: POVM,
    train_states: Iterable[qutip.Qobj],
    target_observable: Union[qutip.Qobj, NDArray],
    stats_list: Iterable[float],
    n_realizations : int,
    test_statistics: int = 1,
    quantiles: List[float] = [0.25, 0.75],
    train_options: Dict[str, Any] = {}
) -> dict:
    r"""
    Computes the bias^2 and variance of the estimator for different numbers of states.

    Produces a plot of computed quantities against the number of states, with median and quantiles computed
    over multiple realizations.
    Each realization resamples from the POVM. POVM and training states are fixed beforehand.
    
    Args:
        povm (Iterable[Union[qutip.Qobj, NDArray]]): The POVM elements.
        target_observable (Union[qutip.Qobj, NDArray]): The target observable.
        train_states (Iterable[qutip.Qobj]): The training states.
        stats_list (Iterable[float]): List of statistics to test.
        n_realizations (int): Number of realizations to average over.
        test_statistics (int): Total TEST statistics. This is used to divide the variance.
        quantiles (Iterable[float]): Quantiles to compute when plotting results.

    Returns:
        A dictionary containing the computed bias^2, variance, and MSE for each number of states.
    """
    # ensure iterables are lists and all objects are qutip objects
    obs = qutip.Qobj(target_observable) if not isinstance(target_observable, qutip.Qobj) else target_observable
    stats_list = list(stats_list)
    train_states_list = list(train_states)
    # extract state dimension
    var_results = {}
    bias2_results = {}
    mse_results = {}
    for stat in tqdm(stats_list, desc="Total Statistics"):
        var_for_states = []
        bias2_for_states = []
        mse_for_states = []
        # print(round(stat / len(train_states)))
        for _ in tqdm(range(n_realizations), desc="Realizations", leave=False):
            qelm = train_qelm_from_states_for_observables(
                training_states=train_states_list,
                target_observables=[obs],
                povm=povm,
                statistics=stat // len(train_states_list),
                train_options=train_options
            )
            if qelm.w is None:
                raise ValueError('Model is not trained, or some fuckery is going on (`w` is not set).')
            estimator = qelm.w[0]

            var = exact_average_estimator_variance(estimator=estimator, povm=povm) / test_statistics
            var_for_states.append(var)

            bias2 = exact_average_bias2(estimator=estimator, povm=povm, target_observable=obs)
            bias2_for_states.append(bias2)

            mse_for_states.append(var + bias2)

        var_results[stat] = var_for_states
        bias2_results[stat] = bias2_for_states
        mse_results[stat] = mse_for_states

    # compute the median and quantiles for each number of states
    median_var = [np.median(var_results[stat]) for stat in stats_list]
    lower_var = [np.quantile(var_results[stat], quantiles[0]) for stat in stats_list]
    upper_var = [np.quantile(var_results[stat], quantiles[1]) for stat in stats_list]

    median_bias2 = [np.median(bias2_results[stat]) for stat in stats_list]
    lower_bias2 = [np.quantile(bias2_results[stat], quantiles[0]) for stat in stats_list]
    upper_bias2 = [np.quantile(bias2_results[stat], quantiles[1]) for stat in stats_list]

    median_mse = [np.median(mse_results[stat]) for stat in stats_list]
    lower_mse = [np.quantile(mse_results[stat], quantiles[0]) for stat in stats_list]
    upper_mse = [np.quantile(mse_results[stat], quantiles[1]) for stat in stats_list]

    # ---- plot the results ----
    plt.figure(figsize=(10, 6))


    # Plot Variance with quantile error shading
    plt.fill_between(stats_list, lower_var, upper_var, color='blue', alpha=0.3, label='Variance {}-{} quantiles'.format(quantiles[0], quantiles[1]))
    plt.plot(stats_list, median_var, marker='o', color='blue', label='Median variance')

    # Plot Bias² with quantile error shading
    plt.fill_between(stats_list, lower_bias2, upper_bias2, color='green', alpha=0.3, label='Bias² {}-{} quantiles'.format(quantiles[0], quantiles[1]))
    plt.plot(stats_list, median_bias2, marker='o', color='green', label='Median bias²')

    # Plot MSE with quantile error shading
    plt.fill_between(stats_list, lower_mse, upper_mse, color='red', alpha=0.3, label='MSE {}-{} quantiles'.format(quantiles[0], quantiles[1]))
    plt.plot(stats_list, median_mse, marker='o', color='red', label='Median MSE')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Total statistics')
    # plt.ylabel('Metric Value')
    plt.title('MSE, Var, bias² vs total stat\nnum_train_states: {}; num_realizations: {}; POVM: {}'.format(
        len(train_states_list), n_realizations, str(povm)))
    plt.grid(True)
    plt.legend()
    plt.show()

    return {'var': var_results, 'bias2': bias2_results, 'mse': mse_results}


def biasvar_vs_nstates_fixedstatperstate(
    povm: POVM,
    target_observable: Union[qutip.Qobj, NDArray],
    nstates_list: Iterable[int],
    stat_per_state: int,
    n_realizations : int,
    test_statistics: int = 1,
    quantiles: List[float] = [0.25, 0.75],
    train_options: Dict[str, Any] = {}
) -> dict:
    r"""
    """
    # ensure iterables are lists and all objects are qutip objects
    obs = qutip.Qobj(target_observable) if not isinstance(target_observable, qutip.Qobj) else target_observable
    nstates_list = list(nstates_list)
    # extract state dimension
    var_results = {}
    bias2_results = {}
    mse_results = {}
    for nstates in tqdm(nstates_list, desc="Total Statistics"):
        var_for_states = []
        bias2_for_states = []
        mse_for_states = []
        # print(round(stat / len(train_states)))
        for _ in tqdm(range(n_realizations), desc="Realizations", leave=False):
            train_states = [qutip.ket2dm(qutip.rand_ket(2, 1)) for _ in range(nstates)]
            qelm = train_qelm_from_states_for_observables(
                training_states=train_states,
                target_observables=[obs],
                povm=povm,
                statistics=stat_per_state,
                train_options=train_options
            )
            if qelm.w is None:
                raise ValueError('Model is not trained, or some fuckery is going on (`w` is not set).')
            estimator = qelm.w[0]

            var = exact_average_estimator_variance(estimator=estimator, povm=povm) / test_statistics
            var_for_states.append(var)

            bias2 = exact_average_bias2(estimator=estimator, povm=povm, target_observable=obs)
            bias2_for_states.append(bias2)

            mse_for_states.append(var + bias2)

        var_results[nstates] = var_for_states
        bias2_results[nstates] = bias2_for_states
        mse_results[nstates] = mse_for_states

    # compute the median and quantiles for each number of states
    median_var = [np.median(var_results[stat]) for stat in nstates_list]
    lower_var = [np.quantile(var_results[stat], quantiles[0]) for stat in nstates_list]
    upper_var = [np.quantile(var_results[stat], quantiles[1]) for stat in nstates_list]

    median_bias2 = [np.median(bias2_results[stat]) for stat in nstates_list]
    lower_bias2 = [np.quantile(bias2_results[stat], quantiles[0]) for stat in nstates_list]
    upper_bias2 = [np.quantile(bias2_results[stat], quantiles[1]) for stat in nstates_list]

    median_mse = [np.median(mse_results[stat]) for stat in nstates_list]
    lower_mse = [np.quantile(mse_results[stat], quantiles[0]) for stat in nstates_list]
    upper_mse = [np.quantile(mse_results[stat], quantiles[1]) for stat in nstates_list]

    # ---- plot the results ----
    plt.figure(figsize=(10, 6))


    # Plot Variance with quantile error shading
    plt.fill_between(nstates_list, lower_var, upper_var, color='blue', alpha=0.3, label='Variance {}-{} quantiles'.format(quantiles[0], quantiles[1]))
    plt.plot(nstates_list, median_var, marker='o', color='blue', label='Median variance')

    # Plot Bias² with quantile error shading
    plt.fill_between(nstates_list, lower_bias2, upper_bias2, color='green', alpha=0.3, label='Bias² {}-{} quantiles'.format(quantiles[0], quantiles[1]))
    plt.plot(nstates_list, median_bias2, marker='o', color='green', label='Median bias²')

    # Plot MSE with quantile error shading
    plt.fill_between(nstates_list, lower_mse, upper_mse, color='red', alpha=0.3, label='MSE {}-{} quantiles'.format(quantiles[0], quantiles[1]))
    plt.plot(nstates_list, median_mse, marker='o', color='red', label='Median MSE')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Total statistics')
    # plt.ylabel('Metric Value')
    plt.title('MSE, Var, bias² vs total stat\nnum_realizations: {}; POVM: {}'.format(
        n_realizations, str(povm)))
    plt.grid(True)
    plt.legend()
    plt.show()

    return {'var': var_results, 'bias2': bias2_results, 'mse': mse_results}