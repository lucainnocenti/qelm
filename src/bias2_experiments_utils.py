from tqdm.auto import tqdm
from dataclasses import dataclass, field, replace
from datetime import datetime
import pickle
import os

import numbers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, List, Optional, Tuple, Union, Iterable, Literal, Any, Sequence, cast

from src.quantum_utils import measure_povm, random_kets, ObservablesType, _make_observables_into_nparray
from src.quantum_utils import QuantumStatesBatch, QuantumKetsBatch, QuantumOperatorsBatch, QuantumState, QuantumKet, QuantumOperator, POVM
from src.povms import random_rank1_povm
from src.QELM import QELM
from src.utils import ensure_unique_filename


BIAS2_DATA_DIR = "../data/bias2/"

def truncate_svdvals(matrix, num_vals: int):
    """Truncate singular values beyond a given number."""
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # 2) Zero out all but the largest k singular values
    s_trunc = np.zeros_like(s)
    s_trunc[:num_vals] = s[:num_vals]
    
    # 3) Reconstruct the matrix with truncated singular values
    return U @ np.diag(s_trunc) @ Vt

def truncated_pinv(matrix, num_vals: int):
    """Compute the pseudo-inverse of a matrix using truncated SVD."""
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # 2) Zero out all but the largest k singular values
    s_trunc = np.zeros_like(s)
    s_trunc[:num_vals] = 1 / s[:num_vals]
    
    # 3) Compute the pseudo-inverse with truncated singular values
    return Vt.T @ np.diag(s_trunc) @ U.T

class MatrixBlocksExtractor:
    def __init__(self, true_P: np.ndarray, dirty_P: np.ndarray, train_stat: int):
        self.true_P = true_P
        self.dirty_P = dirty_P
        self.train_stat = train_stat

        self.X = (dirty_P - true_P) * np.sqrt(train_stat)

        rank = np.linalg.matrix_rank(true_P)
        self.true_Pinv = truncated_pinv(true_P, num_vals=rank)
        # compute projectors
        self.pi_supp = self.true_Pinv @ true_P
        self.pi_im = true_P @ self.true_Pinv
        self.I_supp = np.eye(true_P.shape[1])
        self.I_im = np.eye(true_P.shape[0])
        # compute matrix blocks
        self.X11 = self.pi_im @ self.X @ self.pi_supp
        self.X12 = self.pi_im @ self.X @ (self.I_supp - self.pi_supp)
        self.X21 = (self.I_im - self.pi_im) @ self.X @ self.pi_supp
        self.X22 = (self.I_im - self.pi_im) @ self.X @ (self.I_supp - self.pi_supp)
        # compute Schur complement
        X22_pinv = np.linalg.pinv(self.X22)
        self.scX = self.X11 - self.X12 @ X22_pinv @ self.X21
        self.Pi2 = self.I_supp - X22_pinv @ self.X22
        self.Q_pinv = np.linalg.pinv(self.X22 @ self.X22.T)
        self.Y = self.I_supp + self.X21.T @ self.Q_pinv @ self.X21


@dataclass
class QELMExperimentContext:
    """Represents a QELM experiment configuration."""
    dim: int

    # User-providable quantum objects
    test_state: QuantumStatesBatch | Literal['random']
    povm: POVM | Tuple[Literal['random'], int]
    target_obs: QuantumOperatorsBatch | Literal['random']

    train_states: Optional[QuantumStatesBatch] = None  # training states
    train_stat: Optional[int] = None  # statistics used FOR EACH training state
    test_stat: Optional[int] = None  # statistics used for the test state


class Bias2ExperimentsManager:

    def __init__(self, dim: int, n_realizations: int,
                 test_state: QuantumState | Literal['random'],
                 target_obs: ObservablesType | Literal['random'],
                 povm: POVM | Tuple[Literal['random'], int],
                 train_states: Union[QuantumStatesBatch, int, Sequence[int]],
                 train_stat: Union[int, Sequence[int]],
                 seed: Optional[int] = None,
                 savefile: Optional[str]=None,
                 options: Optional[Dict[str, Any]] = None):
        # save starting timestamp for timing purposes
        self.start_time = datetime.now()
        # handle seed
        if seed is None:
            self.seed = int.from_bytes(os.urandom(4), byteorder="big", signed=False)
        else:
            self.seed = seed
        np.random.seed(self.seed)
        # parse savefile
        if savefile is not None:
            # if savefile is not in the folder ./data/bias2, put it there
            if not savefile.startswith(BIAS2_DATA_DIR):
                savefile = f"{BIAS2_DATA_DIR}{savefile}"
            # check that the directory makes sense as a file path
            if not os.path.exists(os.path.dirname(savefile)):
                raise FileNotFoundError(f"Directory {os.path.dirname(savefile)} does not exist.")
            # if savefile already exists, change it to avoid overwriting
            savefile = ensure_unique_filename(savefile)
        # partial initialization
        self.n_realizations = n_realizations
        self.options = options if options is not None else {}
        if isinstance(target_obs, str) and target_obs == 'random':
            target_obs_ = 'random'
        else:  # assume target_obs is a list of observables or a single observable
            target_obs_ = QuantumOperatorsBatch(_make_observables_into_nparray(target_obs))
        if isinstance(test_state, str) and test_state == 'random':
            test_state_ = 'random'
        else:
            test_state_ = test_state.asbatch()
        # parse train_states and train_stat to figure out which set of parameters to use for the experiment
        # if one of them is a single value, it will be used for all realizations
        # if both are sequences, all combinations will be used
        base_context = QELMExperimentContext(dim=dim, test_state=test_state_,
                                             povm=povm, target_obs=target_obs_)
        self.base_context = base_context
        
        results = []
        if isinstance(train_states, int) and isinstance(train_stat, int):
            for _ in tqdm(range(n_realizations), desc="Running realizations"):
                context = replace(base_context,
                                  train_states=QuantumKetsBatch(random_kets(dim=dim, num_kets=train_states)),
                                  train_stat=train_stat)
                results.append(self.run_experiment(context))
        elif isinstance(train_states, int) and isinstance(train_stat, Sequence):
            # in this case we loop over train_stat
            train_states_ = QuantumKetsBatch(random_kets(dim=dim, num_kets=train_states))
            for train_stat_ in tqdm(train_stat, desc="Train_stat loop"):
                for _ in tqdm(range(n_realizations), desc="Running realizations", leave=False):
                    context = replace(base_context,
                                      train_states=train_states_,
                                      train_stat=train_stat_)
                    results.append(self.run_experiment(context))
        elif isinstance(train_states, Sequence) and isinstance(train_stat, int):
            # in this case we loop over train_states
            for train_states_ in tqdm(train_states, desc="Train_states loop"):
                # here train_states_ is an int specifying the number of training states
                for _ in tqdm(range(n_realizations), desc="Running realizations", leave=False):
                    context = replace(base_context,
                                      train_states=QuantumKetsBatch(random_kets(dim=dim, num_kets=train_states_)),
                                      train_stat=train_stat)
                    results.append(self.run_experiment(context))
        elif isinstance(train_states, Sequence) and isinstance(train_stat, Sequence):
            # in this case we loop over both train_states and train_stat
            for train_states_ in tqdm(train_states, desc="Train_states loop"):
                for train_stat_ in tqdm(train_stat, desc="Train_stat loop", leave=False):
                    for _ in tqdm(range(n_realizations), desc="Running realizations", leave=False):
                        context = replace(base_context,
                                          train_states=QuantumKetsBatch(random_kets(dim=dim, num_kets=train_states_)),
                                          train_stat=train_stat_)
                        results.append(self.run_experiment(context))
        else:
            raise ValueError("train_states and train_stat must be either int or Sequence[int]")
        # store the results
        self.data = pd.DataFrame(results)
        # save final time
        self.end_time = datetime.now()
        # save the summary if savefile is provided
        if savefile is not None:
            self.save_results(savefile)


    def run_experiment(self, context: QELMExperimentContext) -> Dict[str, Any]:
        # train the QELM model with finite and infinite statistics
        assert isinstance(context.train_states, QuantumStatesBatch), "train_states is a numpy array"
        assert isinstance(context.train_stat, numbers.Integral), "train_stat is an integer"
        # generate random stuff if needed
        if context.test_state == 'random':
            test_state = QuantumKetsBatch(random_kets(dim=context.dim, num_kets=1))
        else:
            test_state = context.test_state
        if context.target_obs == 'random':
            target_obs = QuantumKet(random_kets(dim=context.dim, num_kets=1)).todm().asbatch()
        else:
            target_obs = context.target_obs
        if isinstance(context.povm, tuple) and context.povm[0] == 'random':
            povm = random_rank1_povm(dim=context.dim, num_outcomes=context.povm[1])
        elif isinstance(context.povm, POVM):
            povm = context.povm
        else:
            raise ValueError("povm must be a POVM object or a tuple of ('random', num_outcomes)")
        # do the training with finite and infinite statistics
        qelm_inf = QELM.train_from_observables_and_states(
            training_states=context.train_states, test_states=test_state,
            target_observables=target_obs, povm=povm, statistics=np.inf)
        true_w = cast(NDArray, qelm_inf.w)

        true_P = qelm_inf.train.frequencies # type: ignore[assignment]
        true_test_P = qelm_inf.test.frequencies # type: ignore[assignment]

        qelm_fin = QELM.train_from_observables_and_states(
            training_states=context.train_states, test_states=test_state,
            target_observables=target_obs, povm=povm, statistics=(context.train_stat, np.inf)
        )
        dirty_w = qelm_fin.w
        dirty_P = qelm_fin.train.frequencies # type: ignore[assignment]

        # extract the matrix blocks, schur complement, and other quantities
        assert true_P is not None, "true_P should not be None"
        assert true_test_P is not None, "true_test_P should not be None"
        assert dirty_P is not None, "dirty_P should not be None"

        # compute the bias approximations
        bias2_real = ((dirty_w - true_w) @ true_test_P.flatten())**2
        bias2_real = bias2_real[0]

        if self.options.get('onlybias2real', False):
            # if we only want the bias2_real, return it immediately
            # this can speed things up considerably if we only care about the real bias (eg for the sic povm)
            return {
                'n_train_states': context.train_states.n_states,
                'train_stat': context.train_stat,
                'bias2_real': bias2_real
            }
        # compute the needed vectorised objects
        vO = target_obs.vectorise().flatten()
        vsigma = test_state.vectorise().flatten()
        Mmu = povm.vectorise()
        Mrho = context.train_states.vectorise()
        Mrhoinv = np.linalg.pinv(Mrho)
        Mmuinv = np.linalg.pinv(Mmu)
        # extract relevant matrix blocks and Schur complement etc
        mb = MatrixBlocksExtractor(true_P=true_P, dirty_P=dirty_P, train_stat=context.train_stat)
        train_stat = context.train_stat

        bias2_leading_core_1 = vO.T @ Mmuinv.T @ mb.scX @ Mrhoinv @ vsigma
        bias2_leading_core_2 = mb.scX @ mb.true_Pinv @ mb.scX - mb.X12 @ mb.Pi2 @ mb.X12.T @ mb.true_Pinv.T @ mb.Y
        bias2_leading_core_2 = vO.T @ Mmuinv.T @ bias2_leading_core_2 @ Mrhoinv @ vsigma

        bias2_leading_1 = bias2_leading_core_1 ** 2 / train_stat
        bias2_leading_1_onlyX11 = (vO.T @ Mmuinv.T @ mb.X11 @ Mrhoinv @ vsigma) ** 2 / train_stat
        
        bias2_leading_2 = - 2 * bias2_leading_core_1 * bias2_leading_core_2 / train_stat**(3/2)
        bias2_leading_2 =+ bias2_leading_1

        bias2_leading_3 = bias2_leading_core_2 ** 2 / (train_stat ** 2)
        bias2_leading_3 += bias2_leading_2

        # if bias2_leading_3 > 10**2:
        #     print(f"Yo wtf. I'm dumping the current state to a pickle for future debugging.")
        #     debug_filename = f"{BIAS2_DATA_DIR}debug_dump_{self.seed}.pkl"
        #     base, ext = os.path.splitext(debug_filename)
        #     i = 1
        #     while os.path.exists(debug_filename):
        #         debug_filename = f"{base}_{i}{ext}"
        #         i += 1
        #     with open(debug_filename, 'wb') as f:
        #         pickle.dump({
        #             'context': context,
        #             'true_P': true_P,
        #             'dirty_P': dirty_P,
        #             'true_test_P': true_test_P,
        #             'vO': vO,
        #             'vsigma': vsigma,
        #             'Mmuinv': Mmuinv,
        #             'Mrhoinv': Mrhoinv,
        #             'mb': mb,
        #             'train_stat': train_stat,
        #             'bias2_leading_core_1': bias2_leading_core_1,
        #             'bias2_leading_core_2': bias2_leading_core_2,
        #             'bias2_leading_1': bias2_leading_1,
        #             'bias2_leading_2': bias2_leading_2,
        #             'bias2_leading_3': bias2_leading_3
        #         }, f)

        bias2_leading_3_onlyX11 = (vO.T @ Mmuinv.T @ (mb.X11 @ mb.true_Pinv @ mb.X11) @ Mrhoinv @ vsigma) ** 2 / (train_stat ** 2)
        bias2_leading_3_onlyX11 += bias2_leading_1_onlyX11
        bias2_leading_3_onlyX11 = bias2_leading_3_onlyX11

        bias2_leading_3_onlyX11withPi = (vO.T @ Mmuinv.T @ (mb.X11 @ mb.true_Pinv @ mb.X11 - mb.X12 @ mb.Pi2 @ mb.X12.T @ mb.true_Pinv.T @ mb.Y) @ Mrhoinv @ vsigma) ** 2 / (train_stat ** 2)
        bias2_leading_3_onlyX11withPi += bias2_leading_1_onlyX11
        bias2_leading_3_onlyX11withPi = bias2_leading_3_onlyX11withPi

        # this is the one with only the "important" bits
        bias2_leading_3_true = (vO.T @ Mmuinv.T @ (mb.X12 @ np.eye(mb.X22.shape[1]) @ mb.X12.T @ mb.true_Pinv.T) @ Mrhoinv @ vsigma) ** 2 / (train_stat ** 2)
        bias2_leading_3_true = bias2_leading_3_true
        
        # collect all results in a dictionary
        return {
            'n_train_states': context.train_states.n_states,
            'train_stat': context.train_stat,
            'bias2_real': bias2_real,
            'bias2_leading_1': bias2_leading_1,
            'bias2_leading_1_onlyX11': bias2_leading_1_onlyX11,
            'bias2_leading_2': bias2_leading_2,
            'bias2_leading_3': bias2_leading_3,
            'bias2_leading_3_onlyX11': bias2_leading_3_onlyX11,
            'bias2_leading_3_onlyX11withPi': bias2_leading_3_onlyX11withPi,
            'bias2_leading_3_true': bias2_leading_3_true
        }

    def save_results(self, savefile: str) -> None:
        if savefile is None:
            raise ValueError("Please provide a savefile name to save the summary data.")
        # if savefile already exists, change it to avoid overwriting
        # we already do this in the constructor, but we do it again in case something else created the same filename during the run
        savefile = ensure_unique_filename(savefile)
        # this is probably useless after the ensure_unique_filename call, but let's keep it for safety
        if os.path.exists(savefile):
            raise FileExistsError(f"File {savefile} already exists. Please choose a different name or delete the existing file.")
        # ensure random stuff is properly saved
        if self.base_context.test_state == 'random':
            test_state = 'random'
        else:
            test_state = self.base_context.test_state.data
        if isinstance(self.base_context.povm, tuple):
            povm = self.base_context.povm
            povm_label = f'random ({povm[1]} outcomes)'
        else:
            povm = self.base_context.povm.data
            povm_label = self.base_context.povm.label
        if self.base_context.target_obs == 'random':
            target_obs = 'random'
        else:
            target_obs = self.base_context.target_obs.data
        # create metadata dictionary
        metadata = {
            'n_realizations': self.n_realizations,
            'dim': self.base_context.dim,
            'test_state': test_state,
            'povm': povm,
            'povm_label': povm_label,
            'target_obs': target_obs,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'seed': self.seed
        }
        # dump to a pickle file a tuple with metadata and data
        stuff_to_save = (metadata, self.data)
        with open(savefile, 'wb') as f:
            pickle.dump(stuff_to_save, f)

        print(f"Summary data saved to {savefile}")

    @classmethod
    def load_data(cls, savefile: str) -> Tuple[Dict[str, Any], pd.DataFrame] | pd.DataFrame:
        """Load data from a saved file."""
        # unless an absolute path is provided, assume the file is in the data/bias2 folder
        if not savefile.startswith(BIAS2_DATA_DIR):
            savefile = f"{BIAS2_DATA_DIR}{savefile}"
        if not os.path.exists(savefile):
            raise FileNotFoundError(f"File {savefile} does not exist.")
        if not os.path.isfile(savefile):
            raise ValueError(f"Expected a file at {savefile}, but it does not exist or is not a file.")
        with open(savefile, 'rb') as f:
            data_from_file = pickle.load(f)
        if not isinstance(data_from_file, tuple) or len(data_from_file) != 2:
            print('This was saved with an old version of the code, no metadata is available.')
            if not isinstance(data_from_file, pd.DataFrame):
                raise ValueError(f"Expected a tuple of (metadata, data) or just data, but got {type(data_from_file)}.")
            metadata = {}
            data = data_from_file
        else:
            metadata, data = data_from_file
        return metadata, data
    
    @classmethod
    def load_df(cls, savefile: str) -> pd.DataFrame:
        """Load the DataFrame from a saved file."""
        data_from_file = cls.load_data(savefile)
        if isinstance(data_from_file, pd.DataFrame):
            return data_from_file
        else:
            # if the data is a tuple, return the DataFrame part
            return data_from_file[1]
    
    @classmethod
    def load_metadata(cls, savefile: str) -> Dict[str, Any]:
        """Load the metadata from a saved file."""
        data_from_file = cls.load_data(savefile)
        if isinstance(data_from_file, pd.DataFrame):
            raise ValueError("This file does not contain metadata, only a DataFrame.")
        return data_from_file[0]  # return the metadata part of the tuple


def plot_aggregated_stats(
    data: pd.DataFrame,
    base_cols: list[str],
    x_col: str,
    figure_split_col: str,
    stats_to_plot: Optional[list[str]] = None,
    quantiles: Optional[tuple[float, float]] = None,
    group_by_cols: Optional[list[str]] = None,
    base_labels: Optional[dict[str, str]] = None,
    y_value_filter_max: Optional[float] = None,
    x_scale: str = 'log', y_scale: str = 'log',
    custom_xlabel: Optional[str] = None, custom_ylabel: Optional[str] = None,
    title_prefix: str = 'Bias terms',
    cmap_name: str = 'tab10', legend_fontsize: int = 8, show_grid: bool = True,
    figsize: tuple[int, int] = (8, 6)
):
    """ Plot aggregated statistics from a DataFrame.
    This function aggregates statistics from a DataFrame and plots them against a specified x-column.
    It allows for grouping by multiple columns, plotting various statistics (mean, median, max, etc.),
    and optionally shading quantile bands.
    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data to plot.
    base_cols : list[str]
        List of columns to aggregate and plot. Each column will be plotted with the specified statistics.
    x_col : str
        The column to use as the x-axis in the plot.
    figure_split_col : str
        The column to split the figures by. Each unique value in this column will generate a separate figure.
    stats_to_plot : Optional[list[str]], default=None
        List of statistics to plot for each base column. Options include 'mean', 'median', 'max', 'min'.
        If None, defaults to ['mean', 'median', 'max'].
    quantiles : Optional[tuple[float, float]], default=None
        If provided, will plot shaded bands for the specified quantiles (e.g., (0.25, 0.75) for the 25th and 75th percentiles).
    group_by_cols : Optional[list[str]], default=None
        List of columns to group by before aggregation. If None, defaults to [x_col, figure_split_col].
    base_labels : Optional[dict[str, str]], default=None
        A dictionary mapping base column names to custom labels for the plot legend. If None, uses raw names.
    y_value_filter_max : Optional[float], default=None
        If provided, will filter y-values to be less than or equal to this value before plotting.
    x_scale : str, default='log'
        Scale for the x-axis. Can be 'log' or 'linear'.
    y_scale : str, default='log'
        Scale for the y-axis. Can be 'log' or 'linear'.
    custom_xlabel : Optional[str], default=None
        Custom label for the x-axis. If None, uses the x_col name with LaTeX formatting.
    custom_ylabel : Optional[str], default=None
        Custom label for the y-axis. If None, uses a generic label "Aggregated Value".
    title_prefix : str, default='Bias terms'
        Prefix for the plot title. Will be followed by the statistics and x_col.
    cmap_name : str, default='tab10'
        Name of the colormap to use for plotting. Defaults to 'tab10'.
    legend_fontsize : int, default=8
        Font size for the legend text.
    show_grid : bool, default=True
        Whether to show a grid in the plots.
    figsize : tuple[int, int], default=(8, 6)
        Size of the figure to create for each plot.
    Returns
    -------
    None
    This function generates one or more plots based on the aggregated statistics from the DataFrame.
    """
    if stats_to_plot is None:
        stats_to_plot = ['mean', 'median', 'max']
    if quantiles is not None and 'median' not in stats_to_plot:
        stats_to_plot.append('median')

    if base_labels is None:
        base_labels = {col: col for col in base_cols} # Default to raw names if not provided
    if group_by_cols is None:
        group_by_cols = [x_col, figure_split_col]
    elif x_col not in group_by_cols or figure_split_col not in group_by_cols:
        # Ensure the essential columns for plotting structure are in group_by_cols
        print(f"Warning: {x_col} and {figure_split_col} must be in group_by_cols. Adding them.")
        group_by_cols = list(set(group_by_cols + [x_col, figure_split_col]))

    stat_linestyles = {
        "mean": "-",      # solid
        "median": "--",   # dashed
        "max": "-.",      # dash‐dot
        "min": ":"        # dotted
    }
    linestyle_map = {
        f"{base}_{stat}": stat_linestyles.get(stat, "-")
        for base in base_cols
        for stat in stats_to_plot
    }

    agg_kwargs = {
        f"{base}_{stat}": pd.NamedAgg(column=base, aggfunc=stat)
        for base in base_cols
        for stat in stats_to_plot
    }
    if quantiles is not None:
        for base in base_cols:
            agg_kwargs[f'{base}_qlow']  = pd.NamedAgg(
                column=base, aggfunc=lambda a, q=quantiles[0] : a.quantile(q))
            agg_kwargs[f'{base}_qhigh'] = pd.NamedAgg(
                column=base, aggfunc=lambda a, q=quantiles[1]: a.quantile(q))

    # Perform the groupby‐aggregate:
    df_copy = data.copy()
    grouped = df_copy.groupby(group_by_cols).agg(**agg_kwargs).reset_index()  # type: ignore[arg-type]

    # Build the list of columns to plot:
    # cols_to_plot_aggregated = list(agg_kwargs.keys())

    # Plotting loop (one figure per unique value in figure_split_col):
    split_values = sorted(grouped[figure_split_col].unique())
    cmap = plt.colormaps[cmap_name].resampled(len(base_cols)) # type: ignore[call-arg]

    for split_val in split_values:  # loop producing one figure per split value
        plt.figure(figsize=figsize)
        subset = grouped[grouped[figure_split_col] == split_val]

        for j, base in enumerate(base_cols):  # loop over columns to plot

            # ---- optional shaded band ----
            if quantiles is not None:
                q_low, q_high = quantiles
                y_lo = subset[f'{base}_qlow']
                y_hi = subset[f'{base}_qhigh']
                plt.fill_between(subset[x_col], y_lo, y_hi,
                                 color=cmap(j), alpha=0.20,
                                 label=f"{int(100*q_low)}–{int(100*q_high)}% band")
                # plot the median
                y_med = subset[f'{base}_median']
                plt.plot(subset[x_col], y_med, marker='o', linewidth=1, color=cmap(j),
                         label=f"{base_labels.get(base, base)} (median)",
                         linestyle=linestyle_map[f'{base}_median'])
            # plot the rest of the shitty stats
            for stat in stats_to_plot:
                thing_to_plot = f"{base}_{stat}"
                # skip median if quantiles are provided
                if stat == 'median' and quantiles is not None:
                    continue
                # Apply y-value filter if specified
                if y_value_filter_max is not None:
                    mask = subset[thing_to_plot] <= y_value_filter_max
                else:
                    mask = pd.Series([True] * len(subset), index=subset.index) # All true if no filter
                x_vals = subset[x_col][mask]
                y_vals = subset[thing_to_plot][mask]
                # Handle cases where filtering might leave no data
                if x_vals.empty or y_vals.empty:
                    raise ValueError(f"No data to plot for {thing_to_plot} with split value {split_val}.")
                # do the plotting
                plt.plot(x_vals, y_vals, marker='o',
                    color=cmap(j), linestyle=linestyle_map.get(thing_to_plot, ':'),
                    label=f"{base_labels.get(base, base)} ({stat})" if base_labels else f"{base} ({stat})")
                
        plt.xscale(x_scale) # type: ignore
        plt.yscale(y_scale) # type: ignore

        xlabel_text = custom_xlabel if custom_xlabel else x_col.replace("_", r"\_")
        if r"$" not in xlabel_text and r"{" not in xlabel_text: # Basic check if it's already LaTeX
             xlabel_text = f"${xlabel_text}$" # Default to LaTeX math mode if simple string

        ylabel_text = custom_ylabel if custom_ylabel else "Aggregated Value" # More generic default
        if custom_ylabel and r"$" not in custom_ylabel and r"{" not in custom_ylabel:
            ylabel_text = f"${ylabel_text}$"


        plt.xlabel(xlabel_text, fontsize=12)
        plt.ylabel(ylabel_text, fontsize=12)

        title_parts = [title_prefix]
        if stats_to_plot:
            title_parts.append(f'({", ".join(stats_to_plot)})')
        title_parts.append(f'vs. {xlabel_text}')
        title_parts.append(f'({figure_split_col} = {split_val})')
        plt.title(" ".join(title_parts))

        plt.legend(fontsize=legend_fontsize)
        if show_grid:
            plt.grid(True, which='both', ls='--', lw=0.4, alpha=0.5)
        plt.tight_layout()
        plt.show()