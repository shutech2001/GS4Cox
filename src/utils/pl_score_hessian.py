from __future__ import annotations

from typing import Tuple
import numpy as np
from numpy.typing import NDArray


def cox_score_and_hess(
    beta: NDArray,
    covariates: NDArray,
    time: NDArray,
    event: NDArray,
) -> Tuple[NDArray, NDArray]:
    """Compute score vector and observed (negative) Hessian of the
    Cox partial log likelihood at beta.

    Args:
        beta (NDArray): current parameter
        covariates (NDArray): design matrix
        event (NDArray): event indicator

    Returns:
        Tuple[NDArray, NDArray]:
            - score vector
            - observed (negative) Hessian
    """
    n, p = covariates.shape
    idx = np.argsort(-time)
    X = covariates[idx]
    d = event[idx]

    eta = X @ beta
    e_eta = np.exp(eta)

    # cumulative sums over risk sets
    cum_e_eta = np.cumsum(e_eta)
    cum_Xe = np.cumsum((X * e_eta[:, None]), axis=0)
    # cumulative second moment \sum \exp(\eta) x x^\top - compute via outer products
    cum_S2 = np.zeros((n, p, p))
    outer = np.einsum("ni,nj->nij", X, X)
    cum_S2[0] = e_eta[0] * outer[0]
    for k in range(1, n):
        cum_S2[k] = cum_S2[k-1] + e_eta[k] * outer[k]

    score = np.zeros(p)
    hess = np.zeros((p, p))

    for k in range(n):
        if d[k] == 0:
            continue
        S0 = cum_e_eta[k]
        S1 = cum_Xe[k]
        S2 = cum_S2[k]
        weight = 1.0 / S0
        mean = S1 * weight
        score += X[k] - mean
        hess += (S2 / S0) - np.outer(mean, mean)

    return score, hess