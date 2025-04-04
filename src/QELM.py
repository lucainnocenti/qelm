from __future__ import annotations
import numpy as np
from IPython.display import Markdown, display
from numpy.typing import NDArray
from typing import Optional, TypedDict, List, Any, Union, Dict

from src.utils import truncate_svd, kets_to_vectorized_states_matrix


class QELM:
    """
    Quantum Extreme Learning Machine (QELM) class for training and testing.
    """
    
    def __init__(self,
        train_dict: Optional[dict] = None,
        test_dict: Optional[dict] = None,
        method: str = 'standard',
        train_options: Optional[dict] = None,
        w: Optional[NDArray] = None
    ) -> None:
        """
        Initialize the QELM class.

        Parameters
        -----------
        train_dict : Optional[dict]
            Dictionary containing training data.
        test_dict : Optional[dict]
            Dictionary containing test data.
        method : str
            Method to be used for training. Default is 'standard'.
        train_options : dict, optional
            Additional options for training.
        w : np.ndarray, optional
            Precomputed weight matrix. If provided, training is skipped.
        """
        self.train_dict = train_dict
        self.test_dict = test_dict
        self.method: str = method
        self.train_options: Dict[str, Any] = train_options or {}
        
        # These will be set during training or prediction.
        self.w = None
        self.train_predictions = None
        self.test_predictions = None
        self.state_shadow = None

        if w is None:
            self._train_standard(**self.train_options)
        else:
            self.w = w


    def _train_standard(
        self,
        truncate_singular_values: Union[bool, int] = False,
        rcond: Optional[Union[float, NDArray[np.float64]]] = None
    ) -> None:
        """
        Train using the standard method.

        Parameters
        -----------
        truncate_singular_values : bool or int
            Whether to truncate the pseudo-inverse calculation at a finite value.
            Default is False (i.e., no truncation).
        rcond : array_like or float, optional
            Parameter passed to numpy.linalg.pinv (currently unused).
        """
        if self.train_dict is None:
            raise ValueError('Train data not provided.')

        frequencies = self.train_dict['frequencies']
        if truncate_singular_values is not False:
            # Assume truncate_svd returns an NDArray[np.float64]
            frequencies = truncate_svd(frequencies, truncate_singular_values)

        # Compute weight matrix W using the pseudoinverse of frequencies.
        self.w = np.dot(
            self.train_dict['labels'],
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
        if self.train_dict is None:
            raise ValueError('Train data not provided.')

        frequencies = self.train_dict['frequencies']
        if frequencies.ndim != 2:
            raise ValueError('This should be a 2D array.')

        if truncate_singular_values is not False:
            frequencies = truncate_svd(frequencies, truncate_singular_values)

        states = self.train_dict.get('states')
        if states is None:
            raise ValueError('No states provided in train_dict. Did you use `save_states=True` when creating the dataset?')
        
        # Convert the states (usually stored as kets) to vectorized density matrices.
        states_matrix = kets_to_vectorized_states_matrix(states, basis='paulis')
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
        if self.test_dict is None:
            raise ValueError('Test data not provided.')

        if train:
            if self.train_dict is None:
                raise ValueError('Train data not provided.')
            self.train_predictions = self.predict(self.train_dict['frequencies'])
            # Compute mean squared error along axis 1
            self.train_MSE: NDArray[np.float64] = np.mean((self.train_predictions - self.train_dict['labels']) ** 2, axis=1)
        if test:
            self.test_predictions = self.predict(self.test_dict['frequencies'])
            self.test_MSE: NDArray[np.float64] = np.mean((self.test_predictions - self.test_dict['labels']) ** 2, axis=1)

        if display_results:
            display(Markdown(f"***Train MSE***: {self.train_MSE}"))
            display(Markdown(f"***Test MSE***: {self.test_MSE}"))
        
        return self
