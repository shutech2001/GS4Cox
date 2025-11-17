import numpy as np
from numpy.typing import NDArray
from statsmodels.tsa import stattools  # type: ignore


def compute_sum_acf(one_column_chain: NDArray) -> float:
    """Compute summation of autocorrelation for computing Effective Sample Size (ESS)

    Args:
        one_column_chain (NDArray): sampling result of one parameter

    Returns:
        float: sum of autocorrelation (sum_{k=1}^infty rho(k))
    """
    acf_vals = stattools.acf(one_column_chain)
    # sum of autocorrelation excluding lag 0
    return np.sum(acf_vals[1:])


def compute_ess(chain: NDArray) -> NDArray:
    """Compute Effective Sample Size (ESS) from sampling chain

    Args:
        chain (NDArray): sampling result

    Returns:
        NDArray: ESS (M/(1 + 2{sum_of_autocorrelation}))
    """
    _chain: NDArray = np.atleast_2d(chain)
    n: int
    d: int
    n, d = _chain.shape
    # store sum of autocorrelation by parameter
    sum_of_acf: NDArray = np.empty(d)
    for i in range(d):
        sum_of_acf[i] = compute_sum_acf(_chain[:, i])
    return n / (1 + 2 * sum_of_acf)


def compute_esr(chain: NDArray, runtime: float) -> NDArray:
    """Compute Effective Sampling Rate (ESR) from sampling chain

    Args:
        chain (NDArray): sampling result
        runtime (float): execution time

    Returns:
        NDArray: ESR (normalized ESS by execution time)
    """
    ess: NDArray = compute_ess(chain)
    return ess / runtime


def compute_iact(chain: NDArray) -> NDArray:
    """Compute Integrated Autocorrelation Time (IACT) from sampling chain

    Args:
        chain (NDArray): sampling result

    Returns:
        NDArray: IACT (1 + 2{sum_of_autocorrelation}) = N / ESS
    """
    _chain: NDArray = np.atleast_2d(chain)
    n: int
    d: int
    n, d = _chain.shape
    # store sum of autocorrelation by parameter
    sum_of_acf: NDArray = np.empty(d)
    for i in range(d):
        sum_of_acf[i] = compute_sum_acf(_chain[:, i])
    return 1 + 2 * sum_of_acf


def compute_mcse(chain: NDArray) -> NDArray:
    """Compute Monte Carlo Standard Error (MCSE) from sampling chain

    Args:
        chain (NDArray): sampling result

    Returns:
        NDArray: MCSE (standard_deviation * sqrt(IACT / N))
    """
    _chain: NDArray = np.atleast_2d(chain)
    n: int
    n, _ = _chain.shape
    # compute standard deviation for each parameter
    std: NDArray = np.std(_chain, axis=0, ddof=1)
    # compute IACT for each parameter
    iact: NDArray = compute_iact(_chain)
    # MCSE = std * sqrt(IACT / N)
    return std * np.sqrt(iact / n)
