import unittest
import numpy as np

import os
import sys
# Get the directory of the current file (tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Compute the project root (one level above tests/)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import QELM

class TestQuantumFunctions(unittest.TestCase):

    def test_qelm_training_and_prediction(self) -> None:
        # Create simple training data where counts and expvals are identity matrices.
        # In this trivial case, the weight matrix W should become the identity.
        train_dict = {
            'frequencies': np.eye(2),
            'labels': np.eye(2)
        }
        test_dict = {
            'frequencies': np.eye(2),
            'labels': np.eye(2)
        }
        model = QELM.QELM(train_dict=train_dict, test_dict=test_dict)
        # Prediction on the identity should return the identity.
        predictions = model.predict(np.eye(2))
        self.assertTrue(np.allclose(predictions, np.eye(2), atol=1e-5))

    def test_compute_MSE(self) -> None:
        # Create trivial train and test dictionaries where predictions equal the targets.
        train_dict = {
            'frequencies': np.eye(2),
            'labels': np.eye(2)
        }
        test_dict = {
            'frequencies': np.eye(2),
            'labels': np.eye(2)
        }
        model = QELM.QELM(train_dict=train_dict, test_dict=test_dict)
        model.compute_MSE(display_results=False)
        # MSE should be zero for perfect predictions.
        self.assertTrue(np.allclose(model.train_MSE, np.zeros(2), atol=1e-5))
        self.assertTrue(np.allclose(model.test_MSE, np.zeros(2), atol=1e-5))

    def test_compute_state_shadow_error(self) -> None:
        # Create a train_dict without 'states' to ensure proper error is raised.
        train_dict = {
            'frequencies': np.eye(2),
            'labels': np.eye(2)
        }
        model = QELM.QELM(train_dict=train_dict)
        with self.assertRaises(ValueError):
            model.compute_state_shadow()

if __name__ == "__main__":
    unittest.main()