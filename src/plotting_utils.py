import numpy as np
from numpy.typing import NDArray
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt


def logspace(start: float, stop: float, num: int) -> NDArray[np.float64]:
    """Generate logarithmically spaced values between start and stop."""
    return np.logspace(np.log10(start), np.log10(stop), num=num, base=10.0)


def plot_data_with_error_bars(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    y_err_col: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: bool = True,
    xscale: str = "log",   # "log" or "linear"
    yscale: str = "log"
) -> None:
    """Plot data with asymmetric error bars, ensuring they lie within the axes."""
    # Extract data and errors
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    yerr_lower = np.array([e[0] for e in df[y_err_col]])
    yerr_upper = np.array([e[1] for e in df[y_err_col]])
    yerr = np.vstack((yerr_lower, yerr_upper))
    
    # Plot
    _, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, y, linestyle='-', linewidth=1, color='C1', label='_nolegend_')
    ax.errorbar(
        x, y, yerr=yerr,
        marker='s', linestyle='none', markersize=6,
        markeredgewidth=1, markeredgecolor='black', markerfacecolor='C1',
        ecolor='gray', elinewidth=0.5, capsize=2, capthick=0.5,
        alpha=1.0,
        label=r'Empirical $\mathrm{bias}^2 \pm \mathrm{maxerr}$'
    )
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title or 'Scaling of Bias² with Training Statistics')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    if legend:
        ax.legend()
    
    
    plt.tight_layout()
    plt.show()