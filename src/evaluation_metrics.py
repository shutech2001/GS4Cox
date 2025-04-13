import numpy as np
from statsmodels.tsa import stattools  # type: ignore


def compute_sum_acf(one_column_chain: np.ndarray) -> float:
    """Compute summation of autocorrelation for computing Effective Sample Size (ESS)

    Args:
        one_column_chain (np.ndarray): sampling result of one parameter

    Returns:
        float: sum of autocorrelation (sum_{k=1}^infty rho(k))
    """
    acf_vals = stattools.acf(one_column_chain)
    # sum of autocorrelation excluding lag 0
    return np.sum(acf_vals[1:])


def compute_ess(chain: np.ndarray) -> np.ndarray:
    """Compute Effective Sample Size (ESS) from sampling chain

    Args:
        chain (np.ndarray): sampling result

    Returns:
        np.ndarray: ESS (M/(1 + 2{sum_of_autocorrelation}))
    """
    chain = np.atleast_2d(chain)
    n, d = chain.shape
    # store sum of autocorrelation by parameter
    sum_of_acf = np.empty(d)
    for i in range(d):
        sum_of_acf[i] = compute_sum_acf(chain[:, i])
    return n / (1 + 2 * sum_of_acf)


def compute_esr(chain: np.ndarray, runtime: float) -> np.ndarray:
    """Compute Effective Sampling Rate (ESR) from sampling chain

    Args:
        chain (np.ndarray): sampling result
        runtime (float): execution time

    Returns:
        np.ndarray: ESR (normalized ESS by execution time)
    """
    ess: np.ndarray = compute_ess(chain)
    return ess / runtime
