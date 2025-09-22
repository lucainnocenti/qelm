import os
import sys
parent_path = os.path.abspath("..")
if parent_path not in sys.path:
    sys.path.append(parent_path)
import numpy as np
from src.plotting_utils import logspace
from src.bias2_experiments_utils import Bias2ExperimentsManager
from src.povms import sic_povm
from src.quantum_utils import QuantumKet
import qutip



seed = int.from_bytes(os.urandom(4), byteorder="big", signed=False)
data = Bias2ExperimentsManager(dim=2, n_realizations=300, povm=sic_povm(),
    test_state='random',
    target_obs='random',
    train_states=list(logspace(20, 10**4, num=20).astype(int)),
    train_stat=list(logspace(20, 10**4, num=20).astype(int)),
    savefile='fullscan_sicpovm_randomall_20250604.pkl',
    seed=seed,
    options={'onlybias2real': True}
)
