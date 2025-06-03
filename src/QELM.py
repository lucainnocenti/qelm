from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import numpy as np
import qutip
from dataclasses import dataclass
from IPython.display import Markdown, display
from numpy.typing import NDArray
from typing import Optional, TypedDict, List, Any, Union, Dict, cast, Sequence, Tuple, Literal

from src.utils import truncate_svd
from src.quantum_utils import measure_povm, QuantumStatesBatch, POVM, _make_observables_into_nparray, POVMType, ObservablesType
from src.types import SamplingMethodType


@dataclass
class TrainingData:
    """Holds the training frequencies and corresponding labels."""
    frequencies: Optional[NDArray[np.float64]] = None
    labels: Optional[NDArray[np.float64]] = None
    states: Optional[QuantumStatesBatch] = None
    observables: Optional[ObservablesType] = None
    statistics: Optional[float] = None
    sampling_method: Optional[SamplingMethodType] = None
    povm: Optional[POVM] = None

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
    def from_states_and_observables(cls,
        states: QuantumStatesBatch,
        observables: ObservablesType,
        povm: POVM,
        statistics: float,
        sampling_method: SamplingMethodType = 'standard'
    ) -> TrainingData:
        """
        Factory method to generate TrainingData from quantum states and observables.

        Computes expectation values from given states and target observables,
        and generates measurement frequencies using the provided POVM.

        Parameters
        ----------
        states : QuantumStatesBatch
            A batch of quantum states as a single numpy array
        observables : ObservablesType
            A sequence of observables (qutip.Qobj or numpy.ndarray) for which to compute expectation values.
        povm : POVM
        statistics : float
            Number of measurement shots per state. If statistics is np.inf, it will use exact expectation values.
        sampling_method : SamplingMethodType, optional
            Method for sampling measurement outcomes ('standard' or 'poisson'). Default is 'standard'.

        Returns
        -------
        TrainingData
            An instance containing generated 'frequencies' and 'labels'.
        """
        n_states = len(states)
        observables = _make_observables_into_nparray(observables)
        n_observables = len(observables)

        logging.debug(f"Generating training data for {n_states} states, {n_observables} observables, {statistics} shots/state.")

        frequencies = states.measure_povm(povm=povm, statistics=statistics,
                                          return_frequencies=True, sampling_method=sampling_method)

        expvals = states.expvals(observables=observables)
        trainingdata = cls(frequencies=frequencies, labels=expvals)
        # Store additional information about the training data
        trainingdata.povm = povm
        trainingdata.statistics = statistics
        trainingdata.sampling_method = sampling_method
        trainingdata.states = states
        trainingdata.observables = observables
        # Return the instance
        return trainingdata


class QELM:
    """
    Quantum Extreme Learning Machine (QELM) class for training and testing.
    """
    
    def __init__(self,
        train: Optional[TrainingData] = None,
        test: Optional[TrainingData] = None,
        method: str = 'standard',
        train_options: Dict[str, Any] = {},
        w: Optional[NDArray] = None
    ) -> None:
        """
        Initialize the QELM class.

        Parameters
        -----------
        train : Optional[TrainingData]
            TrainingData object containing training data.
        test : Optional[TrainingData]
            TrainingData object containing test data.
        method : str
            Method to be used for training. Default is 'standard'.
        train_options : dict, optional
            Additional options for training.
        w : np.ndarray, optional
            Precomputed weight matrix. If provided, training is skipped.
        """
        self.train = train
        self.test = test
        self.method: str = method
        self.train_options = train_options
        
        # These will be set during training or prediction.
        self.w = None
        self.train_predictions = None
        self.test_predictions = None
        self.state_shadow = None

        if method not in ['standard']:
            raise ValueError(f'Currently only "standard" is implemented.')

        if w is None:
            self._train_standard(**self.train_options)
        else:
            self.w = w


    def _train_standard(
        self,
        truncate_singular_values: Optional[int] = None,
        rcond: Optional[Union[float, NDArray[np.float64]]] = None
    ) -> None:
        """
        Train using the standard method.

        Parameters
        -----------
        truncate_singular_values : int
            Whether to truncate the pseudo-inverse calculation at a finite value.
            Default is False (i.e., no truncation).
        rcond : array_like or float, optional
            Parameter passed to numpy.linalg.pinv (currently unused).
        """
        if self.train is None:
            raise ValueError('Train data not provided.')
        if self.train.can_train is False:
            raise ValueError('Training data is incomplete.')
        frequencies = cast(NDArray[np.float64], self.train.frequencies)  # frequencies is not None b/c of can_train
        self.train.labels = cast(NDArray[np.float64], self.train.labels)  # labels is not None b/c of can_train

        if truncate_singular_values is not None:
            # Assume truncate_svd returns an NDArray[np.float64]
            frequencies = truncate_svd(frequencies, truncate_singular_values)

        # Compute weight matrix W using the pseudoinverse of frequencies.
        self.w = np.dot(
            self.train.labels,
            np.linalg.pinv(frequencies)
        )
    
    @classmethod
    def train_from_observables_and_states(cls,
        training_states: QuantumStatesBatch,
        target_observables: ObservablesType,
        povm: POVM,
        statistics: Union[float, Tuple[float, float]],
        method: str = 'standard',
        train_options: Dict[str, Any] = {},
        test_states: Optional[QuantumStatesBatch] = None,
        sampling_method: SamplingMethodType = 'standard'
    ) -> QELM:
        """
        Factory method to create and train a QELM directly from input states and observables.

        Handles data generation using TrainingData.from_quantum_inputs and then initializes and trains the QELM instance.

        Parameters
        ----------
        training_states : QuantumStatesBatch
            Quantum states for the training set.
        target_observables : Sequence[qutip.Qobj]
            Observables defining the target labels.
        povm : POVMType
            POVM used for measurements.
        statistics : Union[float, Tuple[float, float]]
            Measurement statistics.
            - float: Used for both training and testing (if test_states provided).
            - Tuple[float, float]: (train_statistics, test_statistics). Requires test_states.
        method : str, optional
            Training method for the QELM (default: 'standard').
        train_options : Dict[str, Any], optional
            Options passed to the QELM training method.
        test_states : Optional[QuantumStatesBatch], optional
            Quantum states for the test set (default: None).
        sampling_method : SamplingMethodType, optional
            Method for sampling measurement outcomes ('standard' or 'poisson').

        Returns
        -------
        QELM
            The trained QELM model instance.
        """
        # --- Statistics Handling ---
        if isinstance(statistics, (list, tuple)):
            if len(statistics) != 2:
                raise ValueError("If statistics is a list/tuple, it must have two elements (train_stat, test_stat).")
            if test_states is None:
                raise ValueError("If separate train/test statistics are given via list/tuple, test_states must be provided.")
            train_stat, test_stat = statistics
            logger.info(f"Using separate statistics: Train={train_stat}, Test={test_stat}")
        else:
            train_stat = statistics
            test_stat = statistics # Use same for test if only one value provided
            logger.info(f"Using single statistics value for train/test: {train_stat}")

        # --- Generate Training Data ---
        logger.info("Generating training data...")
        train_data = TrainingData.from_states_and_observables(
            states=training_states,
            observables=target_observables,
            povm=povm,
            statistics=train_stat,
            sampling_method=sampling_method
        )

        # --- Generate Test Data (Optional) ---
        test_data: Optional[TrainingData] = None
        if test_states is not None:
            logger.info("Generating test data...")
            test_data = TrainingData.from_states_and_observables(
                states=test_states,
                observables=target_observables, # Use same observables
                povm=povm,
                statistics=test_stat,
                sampling_method=sampling_method # Usually same method for consistency
            )
        
        # --- Initialize and Train QELM ---
        logger.info("Initializing QELM instance with generated data.")
        # Pass generated data and other options to the standard constructor
        qelm_instance = cls(
            train=train_data,
            test=test_data,
            method=method,
            train_options=train_options
        )

        return qelm_instance


    def compute_state_shadow(self, truncate_singular_values: Union[bool, int] = False) -> QELM:
        """
        Compute the state shadow of the QELM.

        This computes a matrix that, when applied to any estimated probability distribution,
        returns an estimated full tomographic reconstruction of the measured state.

        Parameters
        -----------
        truncate_singular_values : bool or int
            Whether to truncate the pseudoinverse calculation. Default is False.

        Returns
        --------
        QELM
            The instance with computed state shadow stored in self.state_shadow.
        """
        if self.train is None:
            raise ValueError('Train data not provided.')
        if self.train.frequencies is None:
            raise ValueError('Train frequencies not provided.')
        if self.train.states is None:
            raise ValueError('No train states provided. Did you use `save_states=True` when creating the dataset?')

        frequencies = self.train.frequencies
        if frequencies.ndim != 2:
            raise ValueError('This should be a 2D array.')

        if truncate_singular_values is not False:
            frequencies = truncate_svd(frequencies, truncate_singular_values)

        
        # Convert the states (usually stored as kets) to vectorized density matrices.\
        raise NotImplementedError("This shit needs rethinking.")
        states_matrix = kets_to_vectorized_states_matrix(self.train.states, basis='pauli')
        self.state_shadow = np.dot(states_matrix, np.linalg.pinv(frequencies))
        return self

    def predict(self, probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict the expectation values for given probabilities.

        Parameters
        -----------
        probabilities : np.ndarray
            Probabilities for which to predict the expectation values.

        Returns
        --------
        np.ndarray
            Predicted expectation values.
        """
        if self.w is None:
            raise ValueError('Model is not trained (W is not set).')
        return np.dot(self.w, probabilities)

    def compute_MSE(self, train: bool = True, test: bool = True, display_results: bool = True) -> QELM:
        """
        Compute the mean squared error (MSE) for the train and/or test data.

        Parameters
        -----------
        train : bool
            Whether to compute the MSE on the training data.
        test : bool
            Whether to compute the MSE on the test data.
        display_results : bool
            Whether to display the results using Markdown.

        Returns
        --------
        QELM
            The instance with computed train_MSE and test_MSE.
        """
        if train:
            # Check if train data is provided and can be used to compute MSE
            if self.train is None or self.train.can_train is False:
                raise ValueError('Train data not provided or not sufficient.')
            self.train.frequencies = cast(NDArray[np.float64], self.train.frequencies) # frequencies is not None b/c of can_train
            self.train.labels = cast(NDArray[np.float64], self.train.labels) # labels is not None b/c of can_train
            # Compute the train predictions using the trained model
            self.train_predictions = self.predict(self.train.frequencies)
            self.train_MSE: NDArray[np.float64] = np.mean((self.train_predictions - self.train.labels) ** 2, axis=1)
        if test:
            # Check if test data is provided and can be used to compute MSE
            if self.test is None or self.test.can_train is False:
                raise ValueError('Test data not provided or not sufficient.')
            self.test.frequencies = cast(NDArray[np.float64], self.test.frequencies) # frequencies is not None b/c of can_train
            self.test.labels = cast(NDArray[np.float64], self.test.labels) # labels is not None b/c of can_train
            # Compute the test predictions using the trained model
            self.test_predictions = self.predict(self.test.frequencies)
            self.test_MSE: NDArray[np.float64] = np.mean((self.test_predictions - self.test.labels) ** 2, axis=1)

        if display_results:
            if train and not test:
                merged_md = Markdown(
                    f"***Train MSE***: {self.train_MSE}"
                )
            elif not train and test:
                merged_md = Markdown(
                    f"***Test MSE***: {self.test_MSE}"
                )
            else:
                # Both train and test MSE are computed
                merged_md = Markdown(
                    f"***Train MSE***: {self.train_MSE}\n\n***Test MSE***: {self.test_MSE}"
                )
            display(merged_md)
        
        return self

    # a function to compute the squared bias from the trained model w
    def bias2(self) -> NDArray[np.float64]:
        """
        Compute the squared bias from the trained model.

        Returns
        --------
        np.ndarray
            The squared bias.
        """
        if self.w is None:
            raise ValueError('Model is not trained (W is not set).')
        # both train and test data must have been provided
        if self.train is None or self.test is None:
            raise ValueError('Train and test data must be provided to compute bias.')
        if self.train.labels is None or self.test.labels is None:
            raise ValueError('Train and test labels must be provided to compute bias.')
        if self.train.frequencies is None or self.test.frequencies is None:
            raise ValueError('Train and test frequencies must be provided to compute bias.')
        # multiply the train w by the test frequencies to get the bias
        bias = np.dot(self.w, self.test.frequencies) - self.test.labels
        return bias ** 2