from __future__ import annotations
import numpy as np
from IPython.display import Markdown, display
from numpy.typing import NDArray
from typing import Optional, TypedDict, List, Any, Union, Dict, cast

from src.utils import truncate_svd
from src.quantum_utils import kets_to_vectorized_states_matrix
from src.types import TrainingData


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
    
    def compute_state_shadow(
        self,
        truncate_singular_values: Union[bool, int] = False
    ) -> QELM:
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

        
        # Convert the states (usually stored as kets) to vectorized density matrices.
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

    def compute_MSE(
        self,
        train: bool = True,
        test: bool = True,
        display_results: bool = True
    ) -> QELM:
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
