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



seed = np.random.randint(1, 10**6); np.random.seed(seed)
data = Bias2ExperimentsManager(dim=2, n_realizations=300, povm=sic_povm(),
    test_state=QuantumKet(qutip.rand_ket(2, seed=seed).full().flatten()),
    target_obs=qutip.sigmax().full(),
    train_states=list(logspace(20, 10**4, num=20).astype(int)),
    train_stat=20, savefile='stat20_sicpovm_20250604.pkl', seed=seed
)

seed = np.random.randint(1, 10**6); np.random.seed(seed)
data = Bias2ExperimentsManager(dim=2, n_realizations=300, povm=sic_povm(),
    test_state=QuantumKet(qutip.rand_ket(2, seed=seed).full().flatten()),
    target_obs=qutip.sigmax().full(),
    train_states=list(logspace(20, 10**4, num=20).astype(int)),
    train_stat=40, savefile='stat40_sicpovm_20250604.pkl', seed=seed
)

seed = np.random.randint(1, 10**6); np.random.seed(seed)
data = Bias2ExperimentsManager(dim=2, n_realizations=300, povm=sic_povm(),
    test_state=QuantumKet(qutip.rand_ket(2, seed=seed).full().flatten()),
    target_obs=qutip.sigmax().full(),
    train_states=list(logspace(20, 10**4, num=20).astype(int)),
    train_stat=60, savefile='stat60_sicpovm_20250604.pkl', seed=seed
)

seed = np.random.randint(1, 10**6); np.random.seed(seed)
data = Bias2ExperimentsManager(dim=2, n_realizations=300, povm=sic_povm(),
    test_state=QuantumKet(qutip.rand_ket(2, seed=seed).full().flatten()),
    target_obs=qutip.sigmax().full(),
    train_states=list(logspace(20, 10**4, num=20).astype(int)),
    train_stat=80, savefile='stat80_sicpovm_20250604.pkl', seed=seed
)

seed = np.random.randint(1, 10**6); np.random.seed(seed)
data = Bias2ExperimentsManager(dim=2, n_realizations=300, povm=sic_povm(),
    test_state=QuantumKet(qutip.rand_ket(2, seed=seed).full().flatten()),
    target_obs=qutip.sigmax().full(),
    train_states=list(logspace(20, 10**4, num=20).astype(int)),
    train_stat=100, savefile='stat100_sicpovm_20250604.pkl', seed=seed
)