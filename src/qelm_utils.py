import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import qutip
from dataclasses import dataclass, field
import logging
logger = logging.getLogger(__name__)

# --- Type Hinting Setup ---
from numpy.typing import NDArray
from typing import Optional, Union, Dict, Any, Iterable, List, Sequence, Tuple, cast

# --- Custom Imports ---
from src.QELM import QELM
from src.quantum_utils import measure_povm, average_rho, average_rho_tensor_rho
from src.POVM import POVM, POVMType, POVMElement
# --- Custom types imports ---
from src.types import TrainingData

# --- Dataclass Definitions ---

@dataclass
class EstimatorMetrics:
    """Holds computed metrics for estimator performance analysis.
    
    Attributes
    ----------
    variance : List[float]
        Variance of the estimator.
    bias2 : List[float]
        Bias squared of the estimator.
    mse : List[float]
        Mean squared error of the estimator.
    """
    variance: List[float]
    bias2: List[float]
    mse: List[float]

@dataclass
class MetricResults:
    """Holds the raw results from multiple realizations for different parameter values."""
    param_values: List[Any] # The values of the parameter being varied (e.g., n_states, statistics)
    variance_results: Dict[Any, List[float]] = field(default_factory=dict)
    bias2: Dict[Any, List[float]] = field(default_factory=dict)
    mse_results: Dict[Any, List[float]] = field(default_factory=dict)

    def add_realization_results(self, param_value: Any, variance: float, bias2: float, mse: float):
        """Adds results from a single realization."""
        if param_value not in self.variance_results:
            self.variance_results[param_value] = []
            self.bias2[param_value] = []
            self.mse_results[param_value] = []
        self.variance_results[param_value].append(variance)
        self.bias2[param_value].append(bias2)
        self.mse_results[param_value].append(mse)

# --- Core Functions ---

# def make_training_data(
#     training_states: Sequence[qutip.Qobj],
#     target_observables: Sequence[qutip.Qobj],
#     povm: POVMType,
#     statistics: float,
#     sampling_method: str = 'standard'
# ) -> TrainingData:
#     """
#     Generates training data (measurement frequencies and target expectation values).

#     Measures the provided POVM on each training state to get frequencies and
#     calculates the exact expectation values of target observables for labels.

#     Parameters
#     ----------
#     training_states : Sequence[qutip.Qobj]
#         A sequence of quantum states (density matrices) used for training.
#     target_observables : Sequence[qutip.Qobj]
#         A sequence of observables whose expectation values are the target labels.
#     povm : POVMType
#         The POVM operators used for measurement. Can be a sequence of operators
#         or an instance of the POVM class.
#     statistics : float
#         Number of measurement shots PER STATE. Use np.inf for exact probabilities, otherwise it should be an int.

#     Returns
#     -------
#     TrainingData
#         A dataclass instance containing 'frequencies' and 'labels'.
#         - frequencies: Shape (n_states, n_povm_outcomes)
#         - labels: Shape (n_observables, n_states)
#     """
#     n_states = len(training_states)
#     n_observables = len(target_observables)

#     # Input validation (optional but good practice)
#     if n_states == 0:
#         raise ValueError("Training states list cannot be empty.")
#     if n_observables == 0:
#         raise ValueError("Target observables list cannot be empty.")
#     if statistics <= 0:
#         raise ValueError("Statistics must be positive.")

#     # Measure the POVM on the training states to get frequencies
#     # this should return shape (n_states, n_outcomes)
#     frequencies: NDArray[np.float64] = measure_povm(
#         states=training_states,
#         povm=povm,
#         statistics=statistics,
#         return_frequencies=True,
#         sampling_method=sampling_method
#     )

#     # Calculate exact expectation values for labels
#     train_expvals = np.zeros((n_observables, n_states), dtype=np.float64)
#     for idx, observable in enumerate(target_observables):
#         # Use a list comprehension for conciseness
#         expvals = np.array([qutip.expect(observable, state) for state in training_states]).real
#         train_expvals[idx] = expvals

#     return TrainingData(frequencies=frequencies, labels=train_expvals)


# def train_qelm_with_observables(
#     training_states: Sequence[qutip.Qobj],
#     target_observables: Sequence[qutip.Qobj],
#     povm: POVMType,
#     statistics: Union[float, Tuple[float, float]],
#     method: str = 'standard',
#     train_options: Dict[str, Any] = {},
#     test_states: Optional[Sequence[qutip.Qobj]] = None,
#     sampling_method: str = 'standard'
# ) -> QELM:
#     """
#     Trains a Quantum Extreme Learning Machine (QELM) model.

#     Generates training (and optionally testing) data from quantum states,
#     target observables, and POVM measurements, then trains a QELM model.

#     Parameters
#     ----------
#     training_states : Sequence[qutip.Qobj]
#         Quantum states for the training set.
#     target_observables : Sequence[qutip.Qobj]
#         Observables defining the target labels for training.
#     povm : POVMType
#         POVM used for measurements.
#     statistics : Union[float, Tuple[float, float]]
#         Measurement statistics.
#         - If float: Used for both training and testing (if test_states provided).
#         - If Tuple[float, float]: Specifies (train_statistics, test_statistics).
#           Requires `test_states` to be provided.
#     method : str, optional
#         Training method for the QELM model (default: 'standard').
#     train_options : Dict[str, Any], optional
#         Additional options passed directly to the QELM class constructor.
#     test_states : Optional[Sequence[qutip.Qobj]], optional
#         Quantum states for the test set (default: None).
#     sampling_method : str, optional
#         Method for sampling training states (default: 'standard').
#         Accepts 'standard' or 'poisson'.

#     Returns
#     -------
#     QELM
#         The trained QELM model instance.

#     Raises
#     ------
#     ValueError
#         If `statistics` is a tuple but `test_states` is None, or if the tuple
#         does not contain exactly two elements.
#     """
#     # Process statistics argument
#     if isinstance(statistics, (list, tuple)):
#         if len(statistics) != 2:
#             raise ValueError("If statistics is a list/tuple, it must have two elements (train_stat, test_stat).")
#         if test_states is None:
#             raise ValueError("If separate train/test statistics are given via a list/tuple, test_states must be provided.")
#         train_stat, test_stat = statistics
#     else:
#         # statistics is a float (or int)
#         train_stat = statistics
#         test_stat = statistics # Use same for test

#     # Generate training data
#     train_data: TrainingData = make_training_data(
#         training_states=training_states,
#         target_observables=target_observables,
#         povm=povm,
#         statistics=train_stat,
#         sampling_method=sampling_method
#     )
#     # Generate test data if provided
#     test_data: Optional[TrainingData]
#     if test_states is not None:
#         test_data = make_training_data(
#             training_states=test_states,
#             target_observables=target_observables,
#             povm=povm,
#             statistics=test_stat,
#             sampling_method=sampling_method
#         )
#     else:
#         test_data = None

#     # Initialize and train the QELM model
#     qelm_model = QELM(
#         train=train_data,
#         test=test_data, # Will be None if test_states not provided
#         method=method,
#         train_options=train_options
#     )

#     return qelm_model


# =================================================
# FUNCTIONS FOR ESTIMATOR ANALYSIS
# =================================================

def _ensure_numpy_array(op: POVMElement) -> NDArray[np.complex128]:
    """Helper to convert qutip.Qobj or ndarray to ndarray."""
    if isinstance(op, qutip.Qobj):
        return op.full()
    elif isinstance(op, np.ndarray):
        return op
    else:
        raise TypeError(f"Unsupported operator type: {type(op)}")

def exact_average_estimator_variance(estimator: Sequence[float], povm: POVMType) -> float:
    r"""
    Computes the single-shot estimator variance, averaged over Haar-random pure states.

    Calculates Var[\hat{o}] = E_\rho[ E_a[ \hat{o}(a)^2 | \rho ] - (E_a[ \hat{o}(a) | \rho ])^2 ],
    where E_\rho denotes the average over pure states sampled uniformly from the
    Haar measure, E_a[ f(a) | \rho ] = \sum_a Tr[\mu_a \rho] f(a) denotes the expectation
    value over POVM outcomes 'a' for a fixed state rho, \mu_a are POVM elements,
    and \hat{o}(a) is the estimator value for outcome 'a'.

    The formula implemented is derived by averaging over rho:
    Var[\hat o] = \sum_a E_\rho[Tr[\mu_a \rho]] \hat{o}(a)^2
                 - \sum_{a,b} E_\rho[Tr[\mu_a \rho] Tr[\mu_b \rho]] \hat{o}(a) \hat{o}(b)
    Using E_\rho[\rho] = I/d and E_\rho[\rho \otimes \rho] = (I \otimes I + S) / (d(d+1)),
    where S is the swap operator, leading to the implemented trace formulas involving
    average_rho (I/d) and average_rho_tensor_rho ((I \otimes I + S)/(d(d+1))).

    Parameters
    ----------
    estimator : Sequence[float]
        The estimator values \hat{o}(a) corresponding to each POVM outcome 'a'.
    povm : POVMType
        The POVM elements \mu_a.

    Returns
    -------
    float
        The average single-shot variance of the estimator.
    """
    povm_np: List[NDArray[np.complex128]] = [_ensure_numpy_array(p) for p in povm]
    estimator_np = np.asarray(estimator, dtype=np.float64) # Ensure numpy array

    if not povm_np:
        raise ValueError("POVM list cannot be empty.")
    if len(estimator_np) != len(povm_np):
         raise ValueError("Estimator length must match the number of POVM elements.")

    dim = povm_np[0].shape[0]
    num_outcomes = len(povm_np)

    # Get the averaged density matrices (assumed pre-computed or computed by these functions)
    avg_rho1 = average_rho(d=dim) # Proportional to Identity: I / d
    avg_rho2 = average_rho_tensor_rho(d=dim) # Proportional to (I \otimes I + SWAP) / (d*(d+1))

    # Term 1: E[ E_a[ o(a)^2 | rho ] ] = Sum_a o(a)^2 * Tr[mu_a * E[rho]]
    # E[rho] = avg_rho1 = I/d
    term1 = np.sum([
        estimator_np[a]**2 * np.trace(np.dot(povm_np[a], avg_rho1)).real
        for a in range(num_outcomes)
    ])

    # Term 2: E[ (E_a[ o(a) | rho ])^2 ] = Sum_{a,b} o(a)o(b) * E[ Tr[mu_a rho] Tr[mu_b rho] ]
    # E[ Tr[mu_a rho] Tr[mu_b rho] ] = Tr[ (mu_a \otimes mu_b) * E[rho \otimes rho] ]
    # E[rho \otimes rho] = avg_rho2
    term2 = np.sum([
        estimator_np[a] * estimator_np[b] * np.trace(np.kron(povm_np[a], povm_np[b]) @ avg_rho2).real
        for a in range(num_outcomes)
        for b in range(num_outcomes)
    ])

    variance = term1 - term2
    # Numerical precision can sometimes lead to slightly negative variance
    # return max(0.0, variance)
    return variance


def exact_average_bias2(
    estimator: Sequence[float],
    povm: POVMType,
    target_observable: POVMElement,
) -> float:
    r"""
    Computes the single-shot bias squared, averaged over Haar-random pure states.

    Calculates Bias[\hat o]^2 = E_\rho[ (E_a[ \hat{o}(a) | \rho ] - Tr[O \rho])^2 ],
    where E_\rho denotes the average over pure states sampled uniformly from the
    Haar measure, E_a[ . | \rho ] = \sum_a Tr[\mu_a \rho] . denotes the expectation
    value over POVM outcomes 'a' for a fixed state rho, \mu_a are POVM elements,
    \hat{o}(a) is the estimator value for outcome 'a', and O is the target observable.

    The formula implemented is derived by expanding the square and averaging over rho:
    Bias[\hat o]^2 = E_\rho[ (E_a[ \hat{o}(a) | \rho ])^2 ]
                    - 2 * E_\rho[ E_a[ \hat{o}(a) | \rho ] * Tr[O \rho] ]
                    + E_\rho[ (Tr[O \rho])^2 ]
    This involves terms like E[\rho \otimes \rho], leading to the trace formulas
    implemented using average_rho_tensor_rho.

    Parameters
    ----------
    estimator : Sequence[float]
        The estimator values \hat{o}(a) corresponding to each POVM outcome 'a'.
    povm : POVMType
        The POVM elements \mu_a.
    target_observable : PovmElement
        The target observable O.

    Returns
    -------
    float
        The average single-shot bias squared of the estimator.
    """
    povm_np: List[NDArray[np.complex128]] = [_ensure_numpy_array(p) for p in povm]
    obs_np: NDArray[np.complex128] = _ensure_numpy_array(target_observable)
    estimator_np = np.asarray(estimator, dtype=np.float64)

    if not povm_np:
        raise ValueError("POVM list cannot be empty.")
    if len(estimator_np) != len(povm_np):
        raise ValueError("Estimator length must match the number of POVM elements.")
    if obs_np.shape != povm_np[0].shape:
        raise ValueError("Observable dimensions must match POVM element dimensions.")


    dim = povm_np[0].shape[0]
    num_outcomes = len(povm_np)

    # Get the averaged tensor product density matrix
    # avg_rho1 = average_rho(d=dim) # Not needed here
    avg_rho2 = average_rho_tensor_rho(d=dim) # Proportional to (I \otimes I + SWAP)/(d*(d+1))

    # Term 1: E[ (E_a[ o(a) | rho ])^2 ] = Sum_{a,b} o(a)o(b) * E[ Tr[mu_a rho] Tr[mu_b rho] ]
    # E[ Tr[mu_a rho] Tr[mu_b rho] ] = Tr[ (mu_a \otimes mu_b) * E[rho \otimes rho] ]
    term1 = np.sum([
        estimator_np[a] * estimator_np[b] * np.trace(np.kron(povm_np[a], povm_np[b]) @ avg_rho2).real
        for a in range(num_outcomes)
        for b in range(num_outcomes)
    ])

    # Term 2: -2 * E[ E_a[ o(a) | rho ] * Tr[O rho] ] = -2 * Sum_a o(a) E[ Tr[mu_a rho] Tr[O rho] ]
    # E[ Tr[mu_a rho] Tr[O rho] ] = Tr[ (mu_a \otimes O) * E[rho \otimes rho] ]
    term2 = -2 * np.sum([
        estimator_np[a] * np.trace(np.kron(povm_np[a], obs_np) @ avg_rho2).real
        for a in range(num_outcomes)
    ])

    # Term 3: E[ (Tr[O rho])^2 ] = E[ Tr[O rho] Tr[O rho] ]
    # E[ Tr[O rho] Tr[O rho] ] = Tr[ (O \otimes O) * E[rho \otimes rho] ]
    term3 = np.trace(np.kron(obs_np, obs_np) @ avg_rho2).real

    bias_sq = term1 + term2 + term3
    # Numerical precision can sometimes lead to slightly negative bias^2
    # return max(0.0, bias_sq)
    return bias_sq


# =================================================
# PLOTTING FUNCTION
# =================================================

def _plot_bias_variance_results(
    results: MetricResults,
    x_label: str,
    title: str,
    quantiles: Tuple[float, float] = (0.25, 0.75),
    plot_variance: bool = True,
    plot_bias2: bool = True,
    plot_mse: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Internal helper function to plot bias, variance, and MSE results.

    Creates a log-log plot showing the median and quantile range for
    variance, bias², and MSE against a varying parameter.

    Parameters
    ----------
    results : MetricResults
        A dataclass containing the parameter values and the computed metrics
        (variance, bias2, mse) for each realization at each parameter value.
    x_label : str
        Label for the x-axis.
    title : str
        Title for the plot.
    quantiles : Tuple[float, float], optional
        The lower and upper quantiles to display as shaded regions (default: (0.25, 0.75)).
    plot_variance : bool, optional
        Whether to plot the variance (default: True).
    plot_bias2 : bool, optional
        Whether to plot the bias squared (default: True).
    plot_mse : bool, optional
        Whether to plot the MSE (default: True).
    figsize : Tuple[int, int], optional
        Figure size (default: (10, 6)).
    """
    x_values = results.param_values
    q_low, q_high = quantiles

    # Compute medians and quantiles for plotting
    def compute_stats(metric_results_dict):
        medians = [np.median(metric_results_dict[x]) for x in x_values]
        lower_q = [np.quantile(metric_results_dict[x], q_low) for x in x_values]
        upper_q = [np.quantile(metric_results_dict[x], q_high) for x in x_values]
        return medians, lower_q, upper_q

    median_var, lower_var, upper_var = compute_stats(results.variance_results)
    median_bias2, lower_bias2, upper_bias2 = compute_stats(results.bias2)
    median_mse, lower_mse, upper_mse = compute_stats(results.mse_results)

    # Create the plot
    plt.figure(figsize=figsize)

    quantile_label = f'{q_low*100:.0f}-{q_high*100:.0f}% quantiles'

    # Plot Variance
    if plot_variance:
        plt.fill_between(x_values, lower_var, upper_var, color='blue', alpha=0.2, label=f'Variance ({quantile_label})')
        plt.plot(x_values, median_var, marker='o', linestyle='-', color='blue', label='Median Variance')

    # Plot Bias²
    if plot_bias2:
        plt.fill_between(x_values, lower_bias2, upper_bias2, color='green', alpha=0.2, label=f'Bias² ({quantile_label})')
        plt.plot(x_values, median_bias2, marker='s', linestyle='--', color='green', label='Median Bias²')

    # Plot MSE
    if plot_mse:
        plt.fill_between(x_values, lower_mse, upper_mse, color='red', alpha=0.2, label=f'MSE ({quantile_label})')
        plt.plot(x_values, median_mse, marker='^', linestyle=':', color='red', label='Median MSE')

    # Configure plot appearance
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(x_label)
    # plt.ylabel('Metric Value (log scale)')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()


# =================================================
# BIAS-VARIANCE ANALYSIS FUNCTIONS (using plotter)
# =================================================

def analyze_biasvar_vs_nstates(
    povm: POVMType,
    target_observable: Union[qutip.Qobj, NDArray],
    n_states_list: Sequence[int],
    n_realizations: int,
    train_statistics: float,
    test_statistics: int = 1,
    fix_total_statistics: bool = True,
    quantiles: Tuple[float, float] = (0.25, 0.75),
    train_options: Dict[str, Any] = {},
    plot_options: Dict[str, Any] = {'var': True, 'bias2': True, 'mse': True},
    generate_states: str = 'everytime',
    sampling_method: str = 'standard'
) -> MetricResults:
    """
    Analyzes estimator bias² and variance vs. the number of training states.

    For each number of states in `n_states_list`, it performs `n_realizations`.
    In each realization, new random training states are generated, a QELM model
    is trained, and its average bias² and variance are computed. The results
    are aggregated and plotted.

    Parameters
    ----------
    povm : POVMType
        The POVM elements for measurement.
    target_observable : Union[qutip.Qobj, NDArray]
        The target observable for estimation.
    n_states_list : Sequence[int]
        List of different numbers of training states to evaluate.
    n_realizations : int
        Number of independent trials for each number of states.
    train_statistics : float
        Total number of measurement shots available for training across all states.
    test_statistics : float, optional
        Number of shots assumed for the test phase, used to scale the variance
        (Var_N = Var_1 / N) (default: 1.0).
    fix_total_statistics : bool, optional
        If True (default), `train_statistics` is divided equally among
        the `n_states` training states (stat_per_state = total / n_states).
        If False, each state gets `train_statistics` shots.
    quantiles : Tuple[float, float], optional
        Quantiles for the shaded region in the plot (default: (0.25, 0.75)).
    train_options : Dict[str, Any], optional
        Additional options passed to the QELM training function.
    plot_options : Dict[str, Any], optional
        Options for plotting (default: {'var': True, 'bias2': True, 'mse': True}).
    generate_states : str, optional
        Determines how training states are generated:
        - 'everytime': Generate new random states for each realization.
        - 'once_per_realization': Generate random states once for each n_states value.
        - 'once': Generate random states once and reuse them for all realizations.
        (default: 'everytime').
    sampling_method : str, optional
        Method for sampling training states (default: 'standard').
        Accepts 'standard' or 'poisson'.

    Returns
    -------
    MetricResults
        A dataclass containing the raw results (variance, bias², mse lists for
        each n_states value across all realizations).
    """
    # Input type handling and validation
    obs_qobj = qutip.Qobj(target_observable) if not isinstance(target_observable, qutip.Qobj) else target_observable
    n_states_list = sorted(list(n_states_list)) # Ensure sorted list
    if not isinstance(povm, POVM):
        povm = POVM(povm)
    # Extract state dimension (assuming all POVM elements have same shape)
    dim = povm.elements[0].shape[0]

    # Initialize results storage
    results = MetricResults(param_values=n_states_list)
    
    # prepare training states once if so requested
    if generate_states == 'once':
        all_train_states = [qutip.rand_ket(dim) for _ in range(max(n_states_list))]
    train_states = None # Placeholder for when we generate them in the loop


    # Main computation loop
    for n_states in tqdm(n_states_list, desc='Number of States (n_states)', unit='states'):
        if n_states <= 0:
            logger.warning(f"Skipping non-positive n_states value: {n_states}")
            continue

        # Determine statistics per state for this iteration
        if fix_total_statistics:
            # Ensure at least 1 sample per state if possible, avoid division by zero
            train_stat_per_state = max(1.0, train_statistics / n_states)
        else:
            train_stat_per_state = train_statistics # Use the total for each state

        if train_stat_per_state < 1:
             logger.warning(f"Training statistics per state ({train_stat_per_state:.2f}) is less than 1 for n_states={n_states}. Results may be unreliable.")

        if generate_states == 'once_per_realization':
            # Generate random training states only once for this n_states
            train_states = [qutip.rand_ket(dim) for _ in range(n_states)]
        elif generate_states == 'once':
            # Use the pre-generated states, slice to get the right number of states
            train_states = cast(List[qutip.Qobj], all_train_states)[:n_states] # type: ignore[assignment]
 
        for _ in tqdm(range(n_realizations), desc=f'Realizations (n={n_states})', leave=False, unit='realization'):
            # 1. Sample random training states (using pure states)
            if generate_states == 'everytime':
                # Generate random training states for each realization
                train_states = [qutip.rand_ket(dim) for _ in range(n_states)]
            # 2. Train the QELM model
            try:
                qelm = QELM.train_from_observables_and_states(
                    training_states=cast(List[qutip.Qobj], train_states), # States generated above
                    target_observables=[obs_qobj], # The single observable
                    povm=povm,                     # The POVM
                    statistics=train_stat_per_state, # Statistics per state
                    method='standard',             # Assuming standard method
                    train_options=train_options,   # Pass through options
                    test_states=None,              # No test set needed for bias/var calc
                    sampling_method=sampling_method # Pass sampling method
                )
                # Access weights; handle case where training might fail internally
                if qelm.w is None:
                    raise RuntimeError(f"QELM training did not produce weights for n_states={n_states}.")
                # Estimator is the first (and only) row of weights for the single observable
                estimator = qelm.w[0] # Shape (n_povm_outcomes,)
            except Exception as e:
                logger.error(f"QELM training failed for n_states={n_states}: {e}")
                raise e # Raise the exception to stop execution

            # 3. Compute average bias² and variance for this estimator
            # Variance is single-shot, scale by test_statistics
            avg_var_single_shot = exact_average_estimator_variance(estimator=estimator, povm=povm)
            avg_var = avg_var_single_shot / test_statistics

            avg_bias2 = exact_average_bias2(estimator=estimator, povm=povm, target_observable=obs_qobj)
            avg_mse = avg_var + avg_bias2

            # 4. Store results
            results.add_realization_results(n_states, avg_var, avg_bias2, avg_mse)

    # ---- Plotting ----
    title = (
        f'Estimator Performance vs. Number of Training States\n'
        f'Train stat: {train_statistics}, test stat: {test_statistics}, '
        f'{n_realizations} Realizations/pt, Fixed total stat: {fix_total_statistics}\n'
        f'POVM: {povm.label}'
    )
    title = (
        r'$\mathbf{Estimator\ Performance\ vs.\ Number\ of\ Training\ States}$' + '\n' +
        r'$\mathit{Train\ stat:' + f'{train_statistics}' + r',\ test\ stat:' + f'{test_statistics}' +
        r',\ realizations/pt:' + f'{n_realizations}, ' + r'fixed\ total\ stat:}$' + f'{fix_total_statistics}' +
        '\n' +
        r'$\mathit{POVM:}$' + f'{povm.label}' +
        r', $\mathit{state\ generation:}$' + f'{generate_states}'
        r', $\mathit{sampling\ method:}$' + f'{sampling_method}'
    )
    _plot_bias_variance_results(
        results=results,
        x_label='Number of Training States (log scale)',
        title=title,
        quantiles=quantiles,
        plot_variance=plot_options.get('var', True),
        plot_bias2=plot_options.get('bias2', True),
        plot_mse=plot_options.get('mse', True)
    )

    return results


def analyze_biasvar_vs_statistics(
    povm: POVMType,
    target_observable: Union[qutip.Qobj, NDArray],
    train_states: Sequence[qutip.Qobj],
    train_stats_list: Sequence[int],
    n_realizations: int,
    test_statistics: int = 1,
    divide_stat_per_state: bool = True,
    quantiles: Tuple[float, float] = (0.25, 0.75),
    train_options: Dict[str, Any] = {},
    plot_options: Dict[str, Any] = {'var': True, 'bias2': True, 'mse': True},
    sampling_method: str = 'standard'
) -> MetricResults:
    """
    Analyzes estimator bias² and variance vs. total training statistics.

    Uses a *fixed* set of training states. For each value in `total_stats_list`,
    it performs `n_realizations`. In each realization, new measurement samples
    are drawn based on the current statistics budget (divided equally among the
    fixed training states), a QELM model is trained, and its average bias²
    and variance are computed. Results are aggregated and plotted.

    Parameters
    ----------
    povm : POVMType
        The POVM elements for measurement.
    train_states : Sequence[qutip.Qobj]
        The *fixed* set of quantum states used for training in all realizations.
    target_observable : Union[qutip.Qobj, NDArray]
        The target observable for estimation.
    train_stats_list : Sequence[int]
        List of different total training statistics budgets to evaluate.
        This budget is divided equally among the `train_states`, iff divide_stat_per_state is True.
    n_realizations : int
        Number of independent trials (resampling measurement outcomes) for each
        statistics budget.
    test_statistics : int, optional
        Number of shots assumed for the test phase, used to scale the variance
        (Var_N = Var_1 / N) (default: 1).
    divide_stat_per_state : bool, optional
        If True (default), `train_statistics` is divided equally among
        the `train_states` training states (stat_per_state = train_stats / n_states).
        If False, each state gets `train_statistics` shots (stat_per_state = train_stats).
    quantiles : Tuple[float, float], optional
        Quantiles for the shaded region in the plot (default: (0.25, 0.75)).
    train_options : Dict[str, Any], optional
        Additional options passed to the QELM training function.
    plot_options : Dict[str, Any], optional
        Options for plotting (default: {'var': True, 'bias2': True, 'mse': True}).
    sampling_method : str, optional
        Method for sampling training states (default: 'standard').
        Accepts 'standard' or 'poisson'.


    Returns
    -------
    MetricResults
        A dataclass containing the raw results (variance, bias², mse lists for
        each total_stats value across all realizations).
    """
    # Input type handling and validation
    obs_qobj = qutip.Qobj(target_observable) if not isinstance(target_observable, qutip.Qobj) else target_observable
    train_stats_list = sorted(list(train_stats_list)) # Ensure sorted list
    train_states_list = list(train_states) # Ensure list
    n_train_states = len(train_states_list)
    if n_train_states == 0:
        raise ValueError("train_states list cannot be empty.")
    
    if not isinstance(povm, POVM):
        povm = POVM(povm)

    # Initialize results storage
    results = MetricResults(param_values=train_stats_list)

    # Main computation loop
    for train_stat in tqdm(train_stats_list, desc='Total Training Statistics', unit='shots'):
        # Statistics per state for this iteration
        # NOTE: we are rounding train_stat / n_train_states here. This might lead to slightly misleading plots, be careful
        train_stat_per_state = round(train_stat / n_train_states) if divide_stat_per_state else train_stat

        if train_stat_per_state < 1:
             logger.warning(f"Training statistics per state ({train_stat_per_state:.2f}) is less than 1 for total_stat={train_stat}. Results may be unreliable.")


        for _ in tqdm(range(n_realizations), desc=f'Realizations (Stat={train_stat:.1e})', leave=False, unit='realization'):
            # 1. Train the QELM model (states fixed, only measurements resampled)
            try:
                qelm = QELM.train_from_observables_and_states(
                    training_states=train_states_list, # Fixed states
                    target_observables=[obs_qobj],
                    povm=povm,
                    statistics=train_stat_per_state, # Calculated stat per state
                    method='standard',
                    train_options=train_options,
                    test_states=None,
                    sampling_method=sampling_method
                )
                
                estimator = qelm.w[0] # type: ignore[assignment]
            except Exception as e:
                logger.error(f"QELM training failed for total_stat={train_stat}: {e}")
                raise e # Raise the exception to stop execution            

            # 2. Compute average bias² and variance for this estimator
            avg_var_single_shot = exact_average_estimator_variance(estimator=estimator, povm=povm)
            avg_var = avg_var_single_shot / test_statistics

            avg_bias2 = exact_average_bias2(estimator=estimator, povm=povm, target_observable=obs_qobj)
            avg_mse = avg_var + avg_bias2

            # 3. Store results
            results.add_realization_results(train_stat, avg_var, avg_bias2, avg_mse)

    # ---- Plotting ----
    title = (
        f'Estimator Performance vs. Total Training Statistics\n'
        f'Num Train States: {n_train_states} (fixed), Test Stat: {test_statistics}, '
        f'{n_realizations} Realizations/pt\n'
        f'POVM: {povm.label}'
    )
    _plot_bias_variance_results(
        results=results,
        x_label='Total Training Statistics (log scale)',
        title=title,
        quantiles=quantiles,
        plot_variance=plot_options.get('var', True),
        plot_bias2=plot_options.get('bias2', True),
        plot_mse=plot_options.get('mse', True)
    )

    return results


# def analyze_biasvar_vs_nstates_fixedstatperstate(
#     povm: POVMType,
#     target_observable: POVMElement,
#     n_states_list: Sequence[int],
#     stat_per_state: float,
#     n_realizations: int,
#     test_statistics: float = 1.0,
#     quantiles: Tuple[float, float] = (0.25, 0.75),
#     train_options: Dict[str, Any] = {}
# ) -> MetricResults:
#     """
#     Analyzes estimator bias² and variance vs. number of training states,
#     using a *fixed* number of measurement shots *per state*.

#     This is similar to `analyze_biasvar_vs_nstates`, but instead of fixing the
#     *total* statistics budget, it fixes the budget allocated to *each*
#     training state. The total number of shots therefore scales linearly with
#     the number of training states.

#     Parameters
#     ----------
#     povm : POVMType
#         The POVM elements for measurement.
#     target_observable : PovmElement
#         The target observable for estimation.
#     n_states_list : Sequence[int]
#         List of different numbers of training states to evaluate.
#     stat_per_state : float
#         The *fixed* number of measurement shots used for *each* training state. Must be >= 1.
#     n_realizations : int
#         Number of independent trials for each number of states.
#     test_statistics : float, optional
#         Number of shots assumed for the test phase, used to scale the variance
#         (Var_N = Var_1 / N) (default: 1.0).
#     quantiles : Tuple[float, float], optional
#         Quantiles for the shaded region in the plot (default: (0.25, 0.75)).
#     train_options : Dict[str, Any], optional
#         Additional options passed to the QELM training function.

#     Returns
#     -------
#     MetricResults
#         A dataclass containing the raw results (variance, bias², mse lists for
#         each n_states value across all realizations).
#     """
#     # Input type handling and validation
#     if stat_per_state < 1:
#         raise ValueError("stat_per_state must be >= 1.")
#     obs_qobj = qutip.Qobj(target_observable) if not isinstance(target_observable, qutip.Qobj) else target_observable
#     n_states_list = sorted(list(n_states_list))

#     # Extract state dimension
#     if isinstance(povm, POVM):
#         dim = povm.elements[0].shape[0]
#     elif povm:
#         dim = povm[0].shape[0]
#     else:
#         raise ValueError("POVM cannot be empty.")

#     # Initialize results storage
#     results = MetricResults(param_values=n_states_list)

#     # Main computation loop (essentially identical to analyze_biasvar_vs_nstates,
#     # but using `stat_per_state` directly instead of calculating it)
#     for n_states in tqdm(n_states_list, desc='Number of States (n_states)', unit='states'):
#         if n_states <= 0:
#             logging.warning(f"Skipping non-positive n_states value: {n_states}")
#             continue

#         # stat_per_state is fixed by the function argument

#         for _ in tqdm(range(n_realizations), desc=f'Realizations (n={n_states})', leave=False, unit='realization'):
#             # 1. Sample random training states
#             train_states = [qutip.ket2dm(qutip.rand_ket(dim)) for _ in range(n_states)]

#             # 2. Train the QELM model
#             try:
#                 qelm = train_qelm_with_observables(
#                     training_states=train_states,
#                     target_observables=[obs_qobj],
#                     povm=povm,
#                     statistics=stat_per_state, # Use the fixed stat_per_state
#                     train_options=train_options
#                 )
#             except Exception as e:
#                 logging.error(f"QELM training failed for n_states={n_states}: {e}")
#                 continue

#             if not hasattr(qelm, 'w') or qelm.w is None or len(qelm.w) == 0:
#                 logging.error(f"QELM training did not produce weights 'w' for n_states={n_states}.")
#                 continue

#             estimator = qelm.w[0]

#             # 3. Compute average bias² and variance
#             avg_var_single_shot = exact_average_estimator_variance(estimator=estimator, povm=povm)
#             avg_var = avg_var_single_shot / test_statistics if test_statistics > 0 else np.inf

#             avg_bias2 = exact_average_bias2(estimator=estimator, povm=povm, target_observable=obs_qobj)
#             avg_mse = avg_var + avg_bias2

#             # 4. Store results
#             results.add_realization_results(n_states, avg_var, avg_bias2, avg_mse)

#     # ---- Plotting ----
#     povm_str = str(povm) if len(str(povm)) < 50 else type(povm).__name__
#     title = (
#         f'Estimator Performance vs. Number of Training States\n'
#         f'Fixed Stat/State: {stat_per_state}, Test Stat: {test_statistics}, '
#         f'{n_realizations} Realizations/pt\n'
#         f'POVM: {povm_str}'
#     )
#     # Note: X-axis label changed slightly for clarity
#     _plot_bias_variance_results(
#         results=results,
#         x_label='Number of Training States (Fixed Stat/State) (log scale)',
#         title=title,
#         quantiles=quantiles
#     )

#     return results