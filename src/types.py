from __future__ import annotations
import numpy as np
import qutip
from dataclasses import dataclass

# Logger setup
import logging
# --- Type Hinting Setup ---
from numpy.typing import NDArray
from typing import Optional, Union, Dict, Any, Iterable, List, Sequence, Tuple, Literal



# Define Type Aliases for clarity
BasisType = Union[Literal['pauli', 'flatten'], Sequence[NDArray], NDArray]
RescalingType = Literal['none', 'trace']
RescalingInput = Union[RescalingType, Sequence[float]]

SamplingMethodType = Literal['standard', 'poisson']