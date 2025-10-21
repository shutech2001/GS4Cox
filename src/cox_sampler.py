from __future__ import annotations

from collections import deque
import cvxpy as cp
import random
from typing import Deque, Dict, Optional, Tuple
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve  # type: ignore
from scipy.optimize import Bounds, minimize  # type: ignore
from scipy.special import logsumexp  # type: ignore
from scipy.stats import beta as beta_dist, gamma as gamma_dist  # type: ignore
from polyagamma import random_polyagamma  # type: ignore

import rpy2.robjects as ro  # type: ignore
from rpy2.robjects import numpy2ri  # type: ignore
from rpy2.robjects.packages import importr  # type: ignore

numpy2ri.activate()


class CoxSampler:
    """Parent Class of Cox Sampler"""

    @classmethod
    def set_global_seeds(cls, seed: int = 2025) -> None:
        """Set global seeds

        Args:
            seed (int): seed
        """

        # global RNG (Python/NumPy)
        random.seed(seed)
        np.random.seed(seed)

        # global RNG (rpy2)
        try:
            ro.r('RNGkind(kind = "L\'Ecuyer-CMRG")')
            ro.r(f"set.seed({seed})")
        except Exception:
            pass  # R not initialized

    def __init__(self, covariates: NDArray, random_state: int = 2025):
        """
        Args:
            covariates (NDArray): covariates
            random_state (int): random state
        """
        self.covariates: NDArray = covariates
        self.data_num: int
        self.dim_covariates: int
        self.data_num, self.dim_covariates = covariates.shape
        self.rng: np.random.Generator = np.random.default_rng(random_state)

    def build_risk_sets(self, time: NDArray, event: NDArray) -> Tuple[Dict[float, NDArray], NDArray, Deque[int]]:
        """ "Construct the risk set at each event occurrence time.

        Args:
            time (NDArray): observed time
            event (NDArray): event indicator

        Returns:
            Tuple[Dict[float, NDArray], NDArray, Deque[int]]:
                - dictionary of event time and index
                    e.g., {1.0: np.array([2, 4, 5, 0, 1, 3])}
                - time of occurring event
                - the number of events at that time
        """
        event_times: NDArray = np.sort(np.unique(time[event == 1]))
        risk_sets: Dict[float, NDArray] = {}
        event_nums: Deque[int] = deque()
        for t in event_times:
            risk_set_idxs: NDArray = np.where(time >= t)[0]
            event_idxs: NDArray = np.where((time == t) & (event == 1))[0]
            event_nums.append(event_idxs.size)
            if event_idxs.size > 0:
                no_event_idxs: NDArray = np.setdiff1d(risk_set_idxs, event_idxs, assume_unique=True)
                sorted_risk_sets: NDArray = np.concatenate([event_idxs, np.sort(no_event_idxs)])
            else:
                sorted_risk_sets = np.sort(risk_set_idxs)
            risk_sets[t] = sorted_risk_sets
        return risk_sets, event_times, event_nums

    def log_partial_likelihood(
        self,
        beta: NDArray,
        time: NDArray,
        event: NDArray,
    ) -> float:
        """Compute log partial likelihood

        Args:
            beta (NDArray): parameters
            time (NDArray): time of occurring event
            event (NDArray): event flg (1: occurred)

        Returns:
            float: value of log partial likelihood
        """
        linpred = self.covariates.dot(beta)
        # grouping by event time
        mask = event == 1
        t_evt, idx_evt = np.unique(time[mask], return_inverse=True)
        # sum of linpred for each unique event time
        sum_evt = np.bincount(idx_evt, weights=linpred[mask])
        # count of events at each unique event time
        cnt_evt = np.bincount(idx_evt)
        # sum of linpred for each unique event time
        order = np.argsort(time)
        sorted_time = time[order]
        sorted_lp = linpred[order]
        # find the first position of each unique event time
        first_pos = np.searchsorted(sorted_time, t_evt, side="left")
        log_risk_sums = np.array([logsumexp(sorted_lp[pos:]) for pos in np.atleast_1d(first_pos)])

        return np.sum(sum_evt - cnt_evt * log_risk_sums)


class GS4Cox(CoxSampler):
    """Class of Gibbs Sampler for Cox regression Model
    - generalized Bayesian framework
    - composite partial likelihood
    - P\'olya-Gamma augmentation
    - PL-target open-faced sandwich (center+shape)
    """

    def __init__(self, covariates: NDArray, random_state: int = 2025) -> None:
        super().__init__(covariates, random_state)

    def _pl_score_hess(
        self,
        beta: NDArray,
        time: NDArray,
        event: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Compute score and Hessian matrix of the partial log likelihood

        Args:
            beta (NDArray): parameters
            time (NDArray): time of occurring event
            event (NDArray): event flg (1: occurred)

        Returns:
            Tuple[NDArray, NDArray]: score and Hessian matrix
        """
        # sort by time ascending, then compute reverse cumulative sums (risk sets are suffixes)
        order = np.argsort(time)
        X = self.covariates[order]
        t = time[order]
        d = event[order].astype(int)

        eta = X @ beta
        e_eta = np.exp(eta)

        # reverse cumulative sums over risk sets
        S0 = np.cumsum(e_eta[::-1])[::-1]  # (n,)
        S1 = np.cumsum((X * e_eta[:, None])[::-1], axis=0)[::-1]  # (n,d)
        # second moment cumulative: sum e^eta * x x^T
        XX = np.einsum("ni,nj->nij", X, X)
        S2 = np.cumsum((XX * e_eta[:, None, None])[::-1], axis=0)[::-1]  # (n,d,d)

        # unique event times (index of the first position of each unique event time)
        mask_evt = d == 1
        t_evt, inv = np.unique(t[mask_evt], return_inverse=True)
        cnt_evt = np.bincount(inv)  # number of events at each unique time

        first_pos = np.searchsorted(t, t_evt, side="left")  # index into cumulative arrays

        # risk-set moments at each unique event time
        S0_u = S0[first_pos]  # (u,)
        S1_u = S1[first_pos]  # (u,d)
        S2_u = S2[first_pos]  # (u,d,d)

        # sum of x over tied events at each unique time
        X_evt_sum = np.zeros((len(t_evt), self.dim_covariates))
        for k, tt in enumerate(t_evt):
            X_evt_sum[k] = X[(t == tt) & (d == 1)].sum(axis=0)

        # score = \sum_t [ \sum_{events at t} x_i  - cnt(t) * (S1/S0) ]
        score = X_evt_sum.sum(axis=0) - (cnt_evt[:, None] * (S1_u / S0_u[:, None])).sum(axis=0)

        # bread (observed information): Q_PL_obs = \sum_t cnt(t) * (E_xx - E_x E_x^T)
        E_x = S1_u / S0_u[:, None]  # (u,d)
        E_xx = S2_u / S0_u[:, None, None]  # (u,d,d)
        Hbread = np.zeros((self.dim_covariates, self.dim_covariates))
        for k in range(len(t_evt)):
            Hbread += cnt_evt[k] * (E_xx[k] - np.outer(E_x[k], E_x[k]))

        return score, Hbread  # Hbread is positive definite (Q_PL_obs)

    def _cpl_bread_from_pairs(
        self,
        beta: NDArray,
        D: NDArray,
        pairs_i: NDArray,
        event: NDArray,
    ) -> NDArray:
        """Compute bread (observed information) of the composite partial likelihood

        Args:
            beta (NDArray): parameters
            D (NDArray): stacked Δx matrix matching (pairs_i, pairs_j)
            pairs_i (NDArray): event indices
            event (NDArray): event flg (1: occurred)

        Returns:
            NDArray: bread (observed information)
        """
        if D.size == 0:
            return np.eye(self.dim_covariates)

        eta = D @ beta
        # logistic with mild stabilization
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1.0 - p)
        # event weight δ_i (should be 1 for all pairs here, but keep general for future extension)
        wi = event[pairs_i].astype(float)
        w *= wi
        Q_cpl = D.T @ (w[:, None] * D)  # (d,d), symmetric psd
        # symmetrize & ridge for numerical stability
        Q_cpl = 0.5 * (Q_cpl + Q_cpl.T)
        lam = 1e-10
        Q_cpl += lam * np.eye(self.dim_covariates)
        return Q_cpl

    @staticmethod
    def _sym_invsqrt(A: NDArray, eps: float = 1e-8) -> NDArray:
        """Compute symmetric (inverse) square root of a matrix

        Args:
            A (NDArray): symmetric positive definite matrix
            eps (float): epsilon

        Returns:
            NDArray: symmetric (inverse) square root of A
        """
        w, V = np.linalg.eigh(0.5 * (A + A.T))
        w = np.clip(w, eps, None)
        return (V / np.sqrt(w)) @ V.T

    @staticmethod
    def _sym_sqrt(A: NDArray, eps: float = 1e-8) -> NDArray:
        """Compute symmetric square root of a matrix

        Args:
            A (NDArray): symmetric positive definite matrix
            eps (float): epsilon

        Returns:
            NDArray: symmetric square root of A
        """
        w, V = np.linalg.eigh(0.5 * (A + A.T))
        w = np.clip(w, eps, None)
        return (V * np.sqrt(w)) @ V.T

    def _build_pairs(
        self,
        risk_sets: Dict[float, NDArray],
        event_times: NDArray,
        event_nums: Deque[int],
    ) -> Tuple[NDArray, NDArray]:
        """Vectorized construction of all (i,j) pairs for the composite partial likelihood

        Args:
            risk_sets (Dict[float, NDArray]): risk sets
            event_times (NDArray): event times
            event_nums (Deque[int]): event numbers

        Returns:
            Tuple[NDArray, NDArray]: event indices and risk-set indices
        """
        pairs_i_deq: Deque[NDArray] = deque()
        pairs_j_deq: Deque[NDArray] = deque()
        for n_evt, t_evt in zip(event_nums, event_times):
            idxs: NDArray = risk_sets[t_evt]
            if idxs.size < 2:
                continue
            event_idxs: NDArray = idxs[:n_evt]
            risk_idxs: NDArray = idxs
            # repeat and tile to create all combinations
            pairs_i_deq.append(np.repeat(event_idxs, risk_idxs.size))
            pairs_j_deq.append(np.tile(risk_idxs, event_idxs.size))
        if pairs_i_deq:
            pairs_i: NDArray = np.concatenate(pairs_i_deq)
            pairs_j: NDArray = np.concatenate(pairs_j_deq)
        else:
            pairs_i = np.array([], dtype=int)
            pairs_j = np.array([], dtype=int)
        return pairs_i, pairs_j

    def _loss_composite_partial_likelihood(
        self,
        beta: NDArray,
        time: NDArray,
        event: NDArray,
        omega: NDArray,
    ) -> float:
        """Compute PG-augmented complete-data 'gain' (negative loss) of the composite partial likelihood

        Args:
            beta (NDArray): parameters
            time (NDArray): time of occurring event
            event (NDArray): event flg (1: occurred)
            omega (NDArray): PG-augmented complete-data 'gain' (negative loss)

        Returns:
            float: PG-augmented complete-data 'gain' (negative loss)
        """
        risk_sets, event_times, event_nums = self.build_risk_sets(time, event)
        pairs_i, pairs_j = self._build_pairs(risk_sets, event_times, event_nums)
        if len(pairs_i) == 0:
            return 0.0
        D = self.covariates[pairs_i] - self.covariates[pairs_j]
        eta = D.dot(beta)
        kappa = 0.5
        return float(np.sum(kappa * eta - (omega * eta**2) / 2))

    def _cpl_meat_from_pairs(
        self,
        beta: NDArray,
        D: NDArray,
        pairs_i: NDArray,
        event: NDArray,
    ) -> NDArray:
        """Compute empirical meat of the composite partial likelihood

        Args:
            beta (NDArray): parameters
            D (NDArray): stacked Δx matrix matching (pairs_i, pairs_j)
            pairs_i (NDArray): event indices
            event (NDArray): event flg (1: occurred)

        Returns:
            NDArray: empirical meat of the composite partial likelihood
        """
        d = self.dim_covariates
        if D.size == 0:
            return np.eye(d)
        eta = D @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        r = 1.0 - p  # weights for U_i
        Z = r[:, None] * D  # (P,d)
        n = event.shape[0]
        U = np.zeros((n, d))
        # accumulate per event-anchor i
        np.add.at(U, pairs_i, Z)
        evt_mask = event == 1
        Ue = U[evt_mask]
        if Ue.shape[0] == 0:
            return np.eye(d)
        P_hat = (Ue.T @ Ue) / float(Ue.shape[0])
        P_hat = 0.5 * (P_hat + P_hat.T) + 1e-10 * np.eye(d)
        return P_hat

    @staticmethod
    def _cov_invsqrt(A: NDArray, eps: float = 1e-12) -> NDArray:
        """Compute covariance inverse square root of a matrix

        Args:
            A (NDArray): covariance matrix
            eps (float, optional): epsilon. Defaults to 1e-12.

        Returns:
            NDArray: covariance inverse square root of A
        """
        w, V = np.linalg.eigh(0.5 * (A + A.T))
        w = np.clip(w, eps, None)
        return (V / np.sqrt(w)) @ V.T

    @staticmethod
    def _cov_sqrt(A: NDArray, eps: float = 1e-12) -> NDArray:
        """Compute covariance square root of a matrix

        Args:
            A (NDArray): covariance matrix
            eps (float, optional): epsilon. Defaults to 1e-12.

        Returns:
            NDArray: covariance square root of A
        """
        w, V = np.linalg.eigh(0.5 * (A + A.T))
        w = np.clip(w, eps, None)
        return (V * np.sqrt(w)) @ V.T

    def sample(
        self,
        time: NDArray,
        event: NDArray,
        n_iter: int = 1000,
        lr: float = 1.0,
        beta_init: Optional[NDArray] = None,
        prior_mean: Optional[NDArray] = None,
        prior_cov: Optional[NDArray] = None,
        trim_burn_in: bool = False,
        burn_in: int = 500,
    ) -> NDArray:
        """PG-augmented generalized Bayes under CPL, then PL-target OFS:
          (i) center shift: beta_dagger = beta + Q_PL(beta)^{-1} S_PL(beta)
          (ii) shape map: Omega = Q_PL(beta_dagger)^{-1/2} ( lr * Q_CPL(beta_dagger) )^{1/2}
                           beta_{pl-ofs} left_arrow beta_dagger + Omega (beta - beta_dagger)

        Args:
            time (NDArray): time of occurring event
            event (NDArray): event flg (1: occurred)
            n_iter (int, optional): iteration. Defaults to 1000.
            lr (float, optional): learning rate. Defaults to 1.0.
            beta_init (Optional[NDArray], optional): initial value of parameters. Defaults to None.
            prior_mean (Optional[NDArray], optional): prior mean. Defaults to None.
            prior_cov (Optional[NDArray], optional): prior covariance. Defaults to None.
            trim_burn_in (bool, optional): trim burn-in. Defaults to False.
            burn_in (int, optional): burn-in. Defaults to 500.

        Returns:
            NDArray: sampling result
        """
        # set priors
        mean0: NDArray = prior_mean if prior_mean is not None else np.zeros(self.dim_covariates)
        cov0: NDArray = prior_cov if prior_cov is not None else np.eye(self.dim_covariates) * 100
        inv_cov0: NDArray = np.linalg.inv(cov0)
        beta: NDArray = beta_init.copy() if beta_init is not None else np.zeros(self.dim_covariates)
        beta_samples: Deque[NDArray] = deque()

        # risk set and pair pre-computation
        risk_sets, event_times, event_nums = self.build_risk_sets(time, event)
        pairs_i, pairs_j = self._build_pairs(risk_sets, event_times, event_nums)
        D: NDArray = self.covariates[pairs_i] - self.covariates[pairs_j]  # (P,d)

        for _ in range(n_iter):
            # vectorized dot products
            psi: NDArray = D.dot(beta) if D.size else np.zeros(0)
            # PG sampling
            omega: NDArray
            omega = random_polyagamma(1, psi, random_state=self.rng) if psi.size else np.zeros(0)
            kappa: float = 0.5

            # compute weighted covariance and mean for Gaussian conditional of β
            if D.size:
                W: NDArray = omega[:, None]
                add_cov: NDArray = D.T.dot(D * W)  # Σ δ ω ΔxΔx^T
                add_mean: NDArray = kappa * D.sum(axis=0)  # Σ κ Δx
            else:
                add_cov = np.zeros((self.dim_covariates, self.dim_covariates))
                add_mean = np.zeros(self.dim_covariates)

            # posterior precision and mean
            post_prec: NDArray = inv_cov0 + lr * add_cov
            c, lower = cho_factor(post_prec, check_finite=False)
            post_cov = cho_solve((c, lower), np.eye(self.dim_covariates), check_finite=False)
            rhs: NDArray = inv_cov0.dot(mean0) + lr * add_mean
            post_mean: NDArray = cho_solve((c, lower), rhs, check_finite=False)

            # sample new beta
            beta = self.rng.multivariate_normal(post_mean, post_cov)
            beta_samples.append(beta)

        beta_samples_array = np.vstack(beta_samples)

        # common: posterior mean and empirical covariance (current posterior)
        beta_bar = (
            beta_samples_array[burn_in:].mean(axis=0)
            if beta_samples_array.shape[0] > burn_in
            else beta_samples_array.mean(axis=0)
        )
        beta_post_samples = beta_samples_array[burn_in:]
        Sigma_curr = np.cov(beta_post_samples.T, ddof=1)

        # use previously created pairs/D
        # if pairs is empty (no events), return immediately
        if D.size == 0:
            return beta_samples_array
        # first compute beta_dagger
        score_pl, Q_pl = self._pl_score_hess(beta_bar, time, event)
        beta_dagger = beta_bar + np.linalg.solve(Q_pl, score_pl)

        # target covariance (PL)
        Q_pl_at = self._pl_score_hess(beta_dagger, time, event)[1]
        Sigma_target = np.linalg.inv(Q_pl_at)

        # current posterior covariance (computed only from post-burn samples)
        beta_post = beta_samples_array[burn_in:]
        Sigma_curr = np.cov(beta_post.T, ddof=1)

        # compute Omega as "target^{1/2} × curr^{-1/2}" (symmetric square root)
        Omega = self._sym_sqrt(Sigma_target, eps=1e-10) @ self._sym_invsqrt(Sigma_curr, eps=1e-10)

        # apply transformation only to post-burn samples (important: do not include burn-in)
        centered_post = beta_samples_array - beta_post.mean(axis=0)
        beta_pl_ofs_post = beta_dagger[None, :] + centered_post @ Omega.T

        if trim_burn_in:
            return beta_pl_ofs_post[burn_in:]  # trim burn-in

        return beta_pl_ofs_post  # return all samples


class CoxMHSampler(CoxSampler):
    """Class of Metropolis-Hastings sampler for Cox regression model in general Bayesian framework"""

    def __init__(self, covariates: NDArray, random_state: int = 2025) -> None:
        super().__init__(covariates, random_state)

    def log_pl_posterior(
        self, beta: NDArray, time: NDArray, event: NDArray, lr: float = 1.0, cov0: float = 10.0
    ) -> float:
        """Compute log posterior likelihood in generalized bayesian inference

        Args:
            beta (NDArray): parameters
            time (NDArray): time of occurring event
            event (NDArray): event flg (1: occurred)
            lr (float, optional): learning rate. Defaults to 1.0.
            cov0 (float, optional): scale parameter for prior distribution. Defaults to 10.0.

        Returns:
            float: log posterior likelihood
        """
        # log prior likelihood
        log_pl_prior: float = -0.5 * np.sum(beta**2) / (cov0**2)
        # log partial likelihood
        log_pl_post: float = self.log_partial_likelihood(beta, time, event)
        return log_pl_prior + lr * log_pl_post

    def _score_and_hessian(
        self,
        beta: NDArray,
        time: NDArray,
        event: NDArray,
        lr: float,
        cov0: float,
    ) -> Tuple[NDArray, NDArray]:
        """Compute score and Hessian matrix analytically

        Args:
            beta (NDArray): parameters
            time (NDArray): time of occurring event
            event (NDArray): event flg (1: occurred)
            lr (float, optional): learning rate for generalized Bayesian framework. Defaults to 1.0.
            cov0 (float): covariance of prior distribution.

        Returns:
            Tuple[NDArray, NDArray]:
                - score
                - Hessian matrix
        """
        linpred = self.covariates.dot(beta)
        exp_lp = np.exp(linpred)
        mask = event == 1
        # unique event times
        t_evt, inv = np.unique(time[mask], return_inverse=True)
        cnt_evt = np.bincount(inv)

        # sort and cumulative sum to get denominator and first and second moment
        order = np.argsort(time)
        X_sorted = self.covariates[order]
        exp_sorted = exp_lp[order]
        rev_cum_w = np.cumsum(exp_sorted[::-1])[::-1]
        rev_cum_Xw = np.cumsum((X_sorted * exp_sorted[:, None])[::-1], axis=0)[::-1]
        rev_cum_XXw = np.cumsum(
            ((X_sorted[:, :, None] * X_sorted[:, None, :]) * exp_sorted[:, None, None])[::-1], axis=0
        )[::-1]

        # get denominator and first and second moment
        first_pos = np.searchsorted(time[order], t_evt, side="left")
        S0 = rev_cum_w[first_pos]
        S1 = rev_cum_Xw[first_pos]
        S2 = rev_cum_XXw[first_pos]

        # score = \sum_i [ x_i - cnt_i * (S1_i / S0_i) ]
        X_evt = np.zeros((len(t_evt), self.dim_covariates))
        # sum of the mean x over the event sets at each unique time point
        for k, t_val in enumerate(t_evt):
            X_evt[k] = self.covariates[(time == t_val) & mask].sum(axis=0)
        score = X_evt.sum(axis=0) - (cnt_evt[:, None] * (S1 / S0[:, None])).sum(axis=0)
        # prior variance component
        score -= beta / (cov0**2)
        score *= lr

        # Hessian matrix = −lr * \sum_i cnt_i * [ (S2/S0) − (S1⊗S1)/S0^2 ]  − I/cov0^2
        H: NDArray = np.zeros((self.dim_covariates, self.dim_covariates))
        for k in range(len(t_evt)):
            E_xx = S2[k] / S0[k]
            E_x = S1[k] / S0[k]
            H += cnt_evt[k] * (E_xx - np.outer(E_x, E_x))
        H = -lr * H
        # prior variance Hessian
        H -= np.eye(self.dim_covariates) / (cov0**2)

        return score, H

    def sample(
        self,
        time: NDArray,
        event: NDArray,
        n_iter: int = 1000,
        lr: float = 1.0,
        beta_init: Optional[NDArray] = None,
        prior_cov_value: Optional[float] = None,
        scaling: float = 1.0,
        trim_burn_in: bool = False,
        burn_in: int = 500,
        verbose: bool = False,
    ) -> NDArray:
        """_summary_

        Args:
            time (NDArray): observed time
            event (NDArray): event indicator
            n_iter (int): iteration of sampling. Defaults to 1000.
            lr (float): learning rate in generalized Bayesian framework. Defaults to 1.0.
            beta_init (Optional[NDArray], optional): initial value of parameters. Defaults to None.
            prior_cov_value (Optional[NDArray], optional): initial covariance value of normal distribution.
                                                           Defaults to None.
            scaling (float, optional): controls the overall scale of proposal covariance matrix. Defaults to 1.0.
            trim_burn_in (bool): trim burn-in. Defaults to False.
            burn_in (int): burn-in period. Defaults to 500.
            verbose (bool, optional): print acceptance rate. Defaults to False.

        Returns:
            NDArray: sampling result
        """
        beta = np.zeros(self.dim_covariates) if beta_init is None else beta_init.copy()
        cov0 = prior_cov_value or 100.0
        lp_post = self.log_partial_likelihood(beta, time, event) * lr - 0.5 * np.sum(beta**2) / (cov0**2)
        accept = 0
        beta_samples = []

        for _ in range(n_iter):
            score, H = self._score_and_hessian(beta, time, event, lr, cov0)
            # proposal covariance
            try:
                cov_prop = scaling * np.linalg.inv(-H)
                # check for positive definite
                np.linalg.cholesky(cov_prop)
                # MH step
                beta_prop = beta + self.rng.multivariate_normal(np.zeros(self.dim_covariates), cov_prop)
            except np.linalg.LinAlgError:
                cov_prop = scaling * np.eye(self.dim_covariates) * 0.1
                # MH step
                beta_prop = beta + self.rng.multivariate_normal(np.zeros(self.dim_covariates), cov_prop)

            lp_prop = self.log_partial_likelihood(beta_prop, time, event) * lr - 0.5 * np.sum(beta_prop**2) / (cov0**2)
            if np.log(self.rng.random()) < (lp_prop - lp_post):
                beta, lp_post = beta_prop, lp_prop
                accept += 1

            beta_samples.append(beta)

        if verbose:
            print(f"acceptance rate: {accept/n_iter:.2f}")

        if trim_burn_in:
            return np.array(beta_samples)[burn_in:]

        return np.array(beta_samples)


class CoxPGSampler(CoxSampler):
    """Cox-P\'olya-Gamma algorithm
    Original citation:
        Benny Ren, Jeffrey S Morris, Ian Barnett, The Cox-P\'olya-Gamma algorithm for flexible Bayesian inference of multilevel survival models, Biometrics, Volume 81, Issue 3, September 2025, ujaf121, doi: 10.1093/biomtc/ujaf121  # noqa: E501
    """

    def __init__(self, covariates: NDArray, random_state: int = 2025):
        super().__init__(covariates, random_state)

        # Import only essential R packages
        utils = importr("utils")

        required_packages = ["BayesLogit", "relliptical", "condMVNorm"]
        for pkg in required_packages:
            try:
                importr(pkg)
            except Exception as e:
                print(f"Error importing R package: {pkg}: {e}")
                print(f"Installing R package: {pkg}")
                utils.install_packages(pkg)

        self.bayeslogit = importr("BayesLogit")
        self.relliptical = importr("relliptical")
        self.cond_mvnorm = importr("condMVNorm")

    def _create_monotonic_splines_delta(
        self, time: NDArray, event: NDArray, partitions: int = 5, weights: Optional[NDArray] = None
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Create monotonic splines delta

        Args:
            time (NDArray): observed time
            event (NDArray): event indicator
            partitions (int, optional): number of partitions. Defaults to 5.
            weights (Optional[NDArray], optional): weights. Defaults to None.

        Returns:
            Tuple[NDArray, NDArray, NDArray, NDArray]:
                - u_obs: observed u
                - Du_obs: observed Du
                - nj: number of events
                - time_seq: time sequence
        """

        if weights is None:
            weights = np.ones(self.data_num)

        max_time = np.max(time)
        scale = 2.0 * max_time
        t = time / scale
        e = event.astype(float)

        event_times = t[e == 1]
        J = min(len(np.unique(event_times)), partitions)
        time_seq = np.quantile(event_times, np.linspace(0, 1, J + 1))
        time_seq = np.sort(np.unique(time_seq))
        J = len(time_seq) - 1
        time_seq[-1] += 1e-7

        # same to R code: 2-step calculation

        # 1st step: calculate u_obs on the grid including boundaries (for interpolation)
        rmb_seq_initial = np.linspace(t.min(), t.max(), 1000)
        rmb_seq_with_boundaries = np.unique(np.sort(np.r_[rmb_seq_initial, time_seq]))

        u_all_for_interp = np.zeros((len(rmb_seq_with_boundaries), J))
        for j in range(J):
            lo, hi = time_seq[j], time_seq[j + 1]
            ind1 = (rmb_seq_with_boundaries >= lo) & (rmb_seq_with_boundaries < hi)
            ind2 = rmb_seq_with_boundaries >= hi
            u_all_for_interp[ind1, j] = rmb_seq_with_boundaries[ind1] - lo
            u_all_for_interp[ind2, j] = hi - lo

        # # for observed values (using linear interpolation)
        u_obs = np.zeros((self.data_num, J))
        for j in range(J):
            u_obs[:, j] = np.interp(t, rmb_seq_with_boundaries, u_all_for_interp[:, j])

        # 2nd step: calculate u_all on the grid including boundaries (for interpolation)
        rmb_seq = np.linspace(t.min(), t.max(), 1000)
        u_all = np.zeros((len(rmb_seq), J))

        for j in range(J):
            lo, hi = time_seq[j], time_seq[j + 1]
            ind1 = (rmb_seq >= lo) & (rmb_seq < hi)
            ind2 = rmb_seq >= hi
            u_all[ind1, j] = rmb_seq[ind1] - lo
            u_all[ind2, j] = hi - lo

        # # use the median of the grid (same to R code)
        med = np.median(u_all, axis=0)

        # # median centering
        u_obs -= med
        u_all -= med

        # # calculate Du_obs
        Du_obs = np.zeros((self.data_num, J))
        for j in range(J):
            lo, hi = time_seq[j], time_seq[j + 1]
            ind = (t >= lo) & (t < hi)
            Du_obs[:, j] = ind.astype(float)

        Du_obs *= e.reshape(-1, 1)
        nj = (Du_obs * weights.reshape(-1, 1)).sum(axis=0)

        return u_obs, Du_obs, nj, time_seq

    def _safe_log1pexp(self, x):
        # log(1+exp(x)) stably
        x = np.asarray(x)
        out = np.empty_like(x)
        pos = x > 0
        out[pos] = x[pos] + np.log1p(np.exp(-x[pos]))
        out[~pos] = np.log1p(np.exp(x[~pos]))
        return out

    def _initial_eta(self, y: NDArray, Du_obs: NDArray, M: NDArray, weights: NDArray) -> NDArray:
        """Compute initial eta

        Args:
            y (NDArray): event indicator
            Du_obs (NDArray): observed Du
            M (NDArray): design matrix
            weights (NDArray): weights

        Returns:
            NDArray: initial eta
        """

        J = Du_obs.shape[1]
        p = M.shape[1]
        eta = cp.Variable(p)

        # set lower bound for spline coefficients to prevent log(0)
        min_eta = 1e-6

        Du_eta = Du_obs @ eta[:J]
        z = M @ eta

        # log-sum-exp stabilization
        lw = np.log(np.maximum(weights, 1e-300))

        # log-sum-exp stabilization: log(Du_eta) instead of log(Du_eta + eps), handled by constraints
        obj = (
            -cp.sum(cp.multiply(y * weights, z))
            + cp.sum(cp.exp(z + lw))
            - cp.sum(cp.multiply(y * weights, cp.log(Du_eta)))
        )

        # set stricter lower bound for spline coefficients
        constraints = [eta[:J] >= min_eta]

        prob = cp.Problem(cp.Minimize(obj), constraints)

        # suppress warnings and execute
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            prob.solve(solver="SCS", max_iters=20000, verbose=False, eps=1e-4)

        val = eta.value
        if val is not None and prob.status in ["optimal", "optimal_inaccurate"]:
            val[:J] = np.maximum(val[:J], 1e-8)
            return val

        J = Du_obs.shape[1]
        p = M.shape[1]
        lw = np.log(np.maximum(weights, 1e-300))

        def objective(eta):
            z = M @ eta
            z = np.clip(z, -60.0, 60.0)
            Du_eta = Du_obs @ eta[:J]
            Du_eta = np.maximum(Du_eta, 1e-12)
            term1 = -np.sum(y * weights * z)
            term2 = np.exp(logsumexp(z + lw))
            term3 = -np.sum(y * weights * np.log(Du_eta))
            obj = term1 + term2 + term3
            if not np.isfinite(obj):
                return 1e100
            return obj

        def gradient(eta):
            z = M @ eta
            z = np.clip(z, -60.0, 60.0)
            Du_eta = Du_obs @ eta[:J]
            Du_eta = np.maximum(Du_eta, 1e-12)
            wz = np.exp(z + lw)
            g = -M.T @ (y * weights) + M.T @ wz
            g[:J] += -(Du_obs.T @ (y * weights / Du_eta))
            g = np.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6)
            return g

        # set initial value more conservatively
        eta0 = np.zeros(p)
        # set initial value of spline coefficients more strictly
        nj_sum = (Du_obs * weights.reshape(-1, 1)).sum(0)
        eta0[:J] = np.maximum(nj_sum / (np.sum(weights) + 1e-12), 0.1)

        rate = np.sum(y * weights) / (np.sum(weights) + 1e-12)
        eta0[J] = np.log(np.maximum(rate, 1e-6))

        # set lower bound more strictly
        lb = np.full(p, -np.inf)
        ub = np.full(p, np.inf)
        lb[:J] = 1e-6  # stricter lower bound
        bounds = Bounds(lb, ub)

        res = minimize(
            objective, eta0, jac=gradient, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=2000, ftol=1e-9)
        )
        best = res.x if res.success else eta0
        best[:J] = np.maximum(best[:J], 1e-6)
        return best

    def _conditional_mvn(
        self, mu: NDArray, sigma: NDArray, dependent_ind: NDArray, given_ind: NDArray, x_given: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """Compute conditional MVN using R's condMVNorm

        Args:
            mu (NDArray): mean
            sigma (NDArray): covariance
            dependent_ind (NDArray): dependent index
            given_ind (NDArray): given index
            x_given (NDArray): given x

        Returns:
            Tuple[NDArray, NDArray]:
                - conditional mean
                - conditional covariance
        """

        sigma = (sigma + sigma.T) / 2
        min_eig = np.min(np.real(np.linalg.eigvals(sigma)))
        if min_eig < 1e-8:
            sigma += np.eye(sigma.shape[0]) * (1e-6 - min_eig)
        sigma = (sigma + sigma.T) / 2

        r_mu = ro.FloatVector(mu)
        r_sigma = ro.r.matrix(sigma.flatten(), nrow=sigma.shape[0], ncol=sigma.shape[1])
        r_dependent = ro.IntVector(dependent_ind + 1)
        r_given = ro.IntVector(given_ind + 1)
        r_x_given = ro.FloatVector(x_given)

        try:
            result = self.cond_mvnorm.condMVN(
                mean=r_mu,
                sigma=r_sigma,
                dependent_ind=r_dependent,
                given_ind=r_given,
                X_given=r_x_given,
                check_sigma=False,
            )
        except Exception as e:
            print(f"Error in condMVNorm: {e}")
            sigma += np.eye(sigma.shape[0]) * 1e-4
            sigma = (sigma + sigma.T) / 2
            r_sigma = ro.r.matrix(sigma.flatten(), nrow=sigma.shape[0], ncol=sigma.shape[1])
            result = self.cond_mvnorm.condMVN(
                mean=r_mu,
                sigma=r_sigma,
                dependent_ind=r_dependent,
                given_ind=r_given,
                X_given=r_x_given,
                check_sigma=False,
            )

        cond_mean = np.array(result.rx2("condMean")).flatten()
        cond_var = np.array(result.rx2("condVar"))

        cond_var = (cond_var + cond_var.T) / 2
        min_eig = np.min(np.real(np.linalg.eigvals(cond_var)))
        if min_eig < 1e-10:
            cond_var += np.eye(cond_var.shape[0]) * (1e-8 - min_eig)

        return cond_mean, cond_var

    def _sample_truncated_mvn(
        self, mu: NDArray, sigma: NDArray, lower: NDArray, upper: NDArray, burn_in: int = 100
    ) -> NDArray:
        """Sample from truncated MVN using R's relliptical

        Args:
            mu (NDArray): mean
            sigma (NDArray): covariance
            lower (NDArray): lower bound
            upper (NDArray): upper bound
            burn_in (int, optional): burn-in. Defaults to 100.

        Returns:
            NDArray: sampling result
        """

        sigma = (sigma + sigma.T) / 2
        try:
            np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            min_eig = np.min(np.real(np.linalg.eigvals(sigma)))
            sigma += np.eye(sigma.shape[0]) * (1e-8 - min_eig)

        r_mu = ro.FloatVector(mu)
        r_sigma = ro.r.matrix(sigma.flatten(), nrow=sigma.shape[0], ncol=sigma.shape[1])
        r_lower = ro.FloatVector(lower)
        r_upper = ro.FloatVector(upper)

        result = self.relliptical.rtelliptical(
            n=1, mu=r_mu, Sigma=r_sigma, lower=r_lower, upper=r_upper, dist="Normal", burn_in=burn_in, thinning=1
        )

        return np.array(result).flatten()

    def _log_mh_prob(
        self, eta_new: NDArray, eta_old: NDArray, y: NDArray, M: NDArray, epsilon: float, weights: NDArray
    ) -> float:
        """Compute log MH acceptance probability

        Args:
            eta_new (NDArray): new eta
            eta_old (NDArray): old eta
            y (NDArray): event indicator
            M (NDArray): design matrix
            epsilon (float): epsilon
            weights (NDArray): weights

        Returns:
            float: log MH acceptance probability
        """

        eta_ep = -np.log(epsilon)

        def f_of_eta(eta):
            z = M @ eta
            z = np.clip(z, -60.0, 60.0)
            term1 = np.sum(weights * np.exp(z))
            term2 = np.sum(weights * (y + epsilon) * self._safe_log1pexp(z + eta_ep))
            val = term1 - term2
            if not np.isfinite(val):
                return np.inf
            return val

        f_old = f_of_eta(eta_old)
        f_new = f_of_eta(eta_new)
        diff = f_old - f_new
        if not np.isfinite(diff):
            return -np.inf
        return min(0.0, diff)

    def sample(
        self,
        time: NDArray,
        event: NDArray,
        n_iter: int = 1000,
        partitions: int = 5,
        epsilon: float = 1.0,
        calibration: bool = True,
        weights: Optional[NDArray] = None,
        prior_mean: Optional[NDArray] = None,
        prior_cov: Optional[NDArray] = None,
        beta_init: Optional[NDArray] = None,
        slice_burn_in: int = 100,
        Z_matrix_list: Optional[Dict[str, NDArray]] = None,
        tau_prior_a: float = 1e-3,
        tau_prior_b: float = 1e-3,
        trim_burn_in: bool = False,
        burn_in: int = 500,
        verbose: bool = False,
    ) -> NDArray:
        """Sample from posterior using Cox-PG algorithm

        Args:
            time (NDArray): observed time
            event (NDArray): event indicator
            n_iter (int, optional): iteration. Defaults to 1000.
            partitions (int, optional): partitions. Defaults to 5.
            epsilon (float, optional): epsilon. Defaults to 1.0.
            calibration (bool, optional): calibration. Defaults to True.
            weights (Optional[NDArray], optional): weights. Defaults to None.
            prior_mean (Optional[NDArray], optional): prior mean. Defaults to None.
            prior_cov (Optional[NDArray], optional): prior covariance. Defaults to None.
            beta_init (Optional[NDArray], optional): initial value of parameters. Defaults to None.
            slice_burn_in (int, optional): slice burn-in. Defaults to 100.
            Z_matrix_list (Optional[Dict[str, NDArray]], optional): Z matrix list. Defaults to None.
            tau_prior_a (float, optional): tau prior a. Defaults to 1e-3.
            tau_prior_b (float, optional): tau prior b. Defaults to 1e-3.
            trim_burn_in (bool, optional): trim burn-in. Defaults to False.
            burn_in (int, optional): burn-in. Defaults to 500.
            verbose (bool, optional): verbose. Defaults to False.

        Returns:
            NDArray: sampling result
        """

        if weights is None:
            weights = np.ones(self.data_num)

        mixed_model = Z_matrix_list is not None
        eta_ep = -np.log(epsilon)

        # Get monotonic splines
        u_obs, Du_obs, nj, partition_bounds = self._create_monotonic_splines_delta(time, event, partitions, weights)
        J = u_obs.shape[1]

        if verbose:
            print(f"Created {J} partitions")
            print(f"Events per partition: {nj}")

        # Build design matrix
        intercept = np.ones((self.data_num, 1))
        M = np.hstack([u_obs, intercept, self.covariates])
        P = M.shape[1]

        # Add random effects
        MM = 0
        MM_vec = []
        if mixed_model:
            assert Z_matrix_list is not None
            Z_list = []
            for _, Z_mat in Z_matrix_list.items():
                Z_list.append(Z_mat)
                MM_vec.append(Z_mat.shape[1])
            Z_combined = np.hstack(Z_list)
            M = np.hstack([M, Z_combined])
            MM = sum(MM_vec)

        # Setup priors - same to R code
        if prior_cov is None:
            A = np.diag(np.ones(P + MM) / 1e6)
        else:
            A = np.diag(np.ones(P + MM) / 1e6)
            # same to R code, set the whole matrix (including non-diagonal elements)
            prior_prec = np.linalg.inv(prior_cov)
            # prior_cov is usually P x P
            if prior_prec.shape[0] == P:
                A[:P, :P] = prior_prec
            else:
                # only covariate part
                A[J + 1 : J + 1 + prior_prec.shape[0], J + 1 : J + 1 + prior_prec.shape[0]] = prior_prec  # noqa: E203

        b_mu = np.zeros(P + MM)
        if prior_mean is not None:
            b_mu[J + 1 : J + 1 + len(prior_mean)] = prior_mean  # noqa: E203

        # Get initial eta using scipy optimization
        if verbose:
            print("Computing initial values using constrained optimization...")

        eta0 = self._initial_eta(event, Du_obs, M, weights)

        if verbose:
            print(f"Initial eta (first {min(10, len(eta0))}): {eta0[:min(10, len(eta0))]}")
            print(f"Initial spline coefficients: {eta0[:J]}")
            print(f"Initial intercept: {eta0[J]}")
            if P > J + 1:
                print(f"Initial beta: {eta0[J+1:P]}")

        if beta_init is not None:
            eta0[J + 1 : J + 1 + self.dim_covariates] = beta_init  # noqa: E203

        eta = eta0.copy()
        tau_vec = np.ones(len(MM_vec)) / 1e6 if mixed_model else None

        # Storage
        beta_samples = []
        accept = 0

        # MCMC loop
        for iter_idx in range(n_iter):

            if verbose and (iter_idx + 1) % 1000 == 0:
                acc_rate = accept / (iter_idx + 1)
                print(
                    f"Iter {iter_idx + 1}/{n_iter} | Acc: {acc_rate:.3f} | "
                    f"Beta: {eta[J+1:J+1+min(3, self.dim_covariates)]}"
                )

            # Step 1: Sample slice variables
            v_max = np.zeros(J)
            for j in range(J):
                v_tmp = beta_dist.rvs(nj[j], 1, random_state=self.rng)
                v_tmp = max(0.0, v_tmp)
                v_max[j] = v_tmp * eta[j]

            v_max = np.maximum(v_max, 0.0)  # type: ignore
            u_plus = np.full(J, 1e4)

            # Step 2: Sample PG auxiliaries
            psi = M @ eta + eta_ep
            psi = np.clip(psi, -100, 100)

            r_psi = ro.FloatVector(psi)
            r_weights_event = ro.FloatVector((event + epsilon) * weights)
            omega = np.array(self.bayeslogit.rpg(self.data_num, r_weights_event, r_psi))
            omega = np.maximum(omega, 1e-10)

            # Step 3: Compute Gaussian moments
            kappa = weights * (event - epsilon) / 2.0
            Q = M.T @ (omega.reshape(-1, 1) * M) + A

            cF = cho_factor(Q, lower=True, check_finite=False)
            mu = M.T @ (kappa - omega * eta_ep) + A @ b_mu
            mu_new = cho_solve(cF, mu, check_finite=False)

            I_Q = np.eye(Q.shape[0])
            sigma_new = cho_solve(cF, I_Q, check_finite=False)
            sigma_new = 0.5 * (sigma_new + sigma_new.T)

            # Step 4: Conditional sampling
            dependent_ind = np.arange(J)
            given_ind = np.arange(J, P + MM)

            cond_mean, cond_cov = self._conditional_mvn(mu_new, sigma_new, dependent_ind, given_ind, eta[given_ind])

            eta_spline = self._sample_truncated_mvn(cond_mean, cond_cov, v_max, u_plus, burn_in=slice_burn_in)

            cond_mean2, cond_cov2 = self._conditional_mvn(mu_new, sigma_new, given_ind, dependent_ind, eta_spline)

            try:
                L = np.linalg.cholesky(cond_cov2)
                eta_other = cond_mean2 + L @ self.rng.standard_normal(len(cond_mean2))
            except Exception as e:
                print(f"Error in cholesky: {e}")
                cond_cov2 += np.eye(cond_cov2.shape[0]) * 1e-8
                L = np.linalg.cholesky(cond_cov2)
                eta_other = cond_mean2 + L @ self.rng.standard_normal(len(cond_mean2))

            eta_new = np.concatenate([eta_spline, eta_other])

            # Step 5: MH calibration
            if calibration:
                log_acc_prob = self._log_mh_prob(eta_new, eta, event, M, epsilon, weights)
                if np.log(self.rng.random()) < log_acc_prob:
                    eta = eta_new
                    accept += 1
            else:
                eta = eta_new
                accept += 1

            # Step 6: Update precision - same to R code
            if mixed_model:
                assert tau_vec is not None
                # create new diagonal matrix
                A_new = np.diag(np.ones(P + MM) / 1e6)

                # keep fixed effect part of prior distribution matrix (same to R code)
                A_new[:P, :P] = A[:P, :P]

                # update precision of random effect part
                MM_counter = 0
                for i, M_dim in enumerate(MM_vec):
                    eta_mixed = eta[P + MM_counter : P + MM_counter + M_dim]  # noqa: E203
                    tau_vec[i] = gamma_dist.rvs(
                        a=tau_prior_a + M_dim / 2,
                        scale=1.0 / (tau_prior_b + np.sum(eta_mixed**2) / 2),
                        random_state=self.rng,
                    )
                    # update diagonal elements
                    for j in range(M_dim):
                        A_new[P + MM_counter + j, P + MM_counter + j] = tau_vec[i]
                    MM_counter += M_dim

                A = A_new

            # Store samples
            beta_samples.append(eta[J + 1 : J + 1 + self.dim_covariates].copy())  # noqa: E203

        if verbose:
            print(f"\nFinal acceptance rate: {accept / (n_iter):.3f}")
            print(f"Collected {len(beta_samples)} posterior samples")

        if trim_burn_in:
            return np.array(beta_samples)[burn_in:]

        return np.array(beta_samples)


class CoxHMCSampler(CoxSampler):
    """Hamiltonian Monte Carlo sampler for Cox regression model with generalized Bayesian framework"""

    def __init__(self, covariates: NDArray, random_state: int = 2025) -> None:
        super().__init__(covariates, random_state)

    def compute_gradient(self, beta: NDArray, time: NDArray, event: NDArray, lr: float, cov0: float) -> NDArray:
        """Compute gradient of log posterior

        Args:
            beta (NDArray): parameters
            time (NDArray): time of occurring event
            event (NDArray): event flg (1: occurred)
            lr (float): learning rate
            cov0 (float): covariance of prior distribution

        Returns:
            NDArray: gradient
        """
        linpred = self.covariates @ beta
        exp_lp = np.exp(linpred)
        mask = event == 1

        # Vectorized event grouping
        t_evt, inv = np.unique(time[mask], return_inverse=True)
        cnt_evt = np.bincount(inv)

        # Sort once
        order = np.argsort(time)
        X_sorted = self.covariates[order]
        exp_sorted = exp_lp[order]

        # Cumulative sums from right to left
        rev_cum_w = np.cumsum(exp_sorted[::-1])[::-1]
        rev_cum_Xw = np.cumsum((X_sorted * exp_sorted[:, None])[::-1], axis=0)[::-1]

        # Get risk set values at event times
        first_pos = np.searchsorted(time[order], t_evt, side="left")
        S0 = rev_cum_w[first_pos]
        S1 = rev_cum_Xw[first_pos]

        # Vectorized X_evt computation (OPTIMIZED)
        X_evt = np.zeros((len(t_evt), self.dim_covariates))
        for idx, _ in enumerate(t_evt):
            indices = inv == idx
            X_evt[idx] = self.covariates[mask][indices].sum(axis=0)

        # Gradient computation
        grad_like = X_evt.sum(axis=0) - (cnt_evt[:, None] * (S1 / S0[:, None])).sum(axis=0)
        grad_prior = -beta / (cov0**2)
        grad = lr * grad_like + grad_prior
        return grad

    def leapfrog(
        self,
        beta: NDArray,
        momentum: NDArray,
        time: NDArray,
        event: NDArray,
        lr: float,
        cov0: float,
        epsilon: float,
        L: int,
    ) -> Tuple[NDArray, NDArray]:
        """Leapfrog integrator for Hamiltonian dynamics

        Args:
            beta (NDArray): parameters
            momentum (NDArray): momentum
            time (NDArray): time of occurring event
            event (NDArray): event flg (1: occurred)
            lr (float): learning rate
            cov0 (float): covariance of prior distribution
            epsilon (float): epsilon
            L (int): number of leapfrog steps

        Returns:
            Tuple[NDArray, NDArray]:
                - beta_new
                - momentum_new
        """
        beta_new = beta.copy()
        momentum_new = momentum.copy()

        # Half step for momentum
        grad = self.compute_gradient(beta_new, time, event, lr, cov0)
        momentum_new += 0.5 * epsilon * grad

        # L full steps
        for i in range(L):
            beta_new += epsilon * momentum_new

            if i < L - 1:
                grad = self.compute_gradient(beta_new, time, event, lr, cov0)
                momentum_new += epsilon * grad

        # Final half step for momentum
        grad = self.compute_gradient(beta_new, time, event, lr, cov0)
        momentum_new += 0.5 * epsilon * grad

        return beta_new, momentum_new

    def sample(
        self,
        time: NDArray,
        event: NDArray,
        n_iter: int = 1000,
        lr: float = 1.0,
        beta_init: Optional[NDArray] = None,
        prior_cov_value: Optional[float] = None,
        epsilon: float = 0.01,
        L: int = 10,
        trim_burn_in: bool = False,
        burn_in: int = 500,
        verbose: bool = False,
    ) -> NDArray:
        """HMC sampling for Cox regression with generalized Bayesian framework

        Args:
            time (NDArray): time of occurring event
            event (NDArray): event flg (1: occurred)
            n_iter (int, optional): iteration. Defaults to 1000.
            lr (float, optional): learning rate. Defaults to 1.0.
            beta_init (Optional[NDArray], optional): initial value of parameters. Defaults to None.
            prior_cov_value (Optional[float], optional): initial covariance value of normal distribution. Defaults to None.  # noqa: E501
            epsilon (float, optional): epsilon. Defaults to 0.01.
            L (int, optional): number of leapfrog steps. Defaults to 10.
            trim_burn_in (bool, optional): trim burn-in. Defaults to False.
            burn_in (int, optional): burn-in. Defaults to 500.
            verbose (bool, optional): verbose. Defaults to False.

        Returns:
            NDArray: sampling result
        """
        beta = np.zeros(self.dim_covariates) if beta_init is None else beta_init.copy()
        cov0 = prior_cov_value or 100.0

        beta_samples = []
        accept = 0

        # Pre-compute constant
        inv_cov0_sq = 1.0 / (cov0**2)

        for _ in range(n_iter):
            momentum = self.rng.standard_normal(self.dim_covariates)

            # Current state
            current_log_prob = self.log_partial_likelihood(beta, time, event) * lr - 0.5 * np.sum(beta**2) * inv_cov0_sq
            current_kinetic = 0.5 * np.sum(momentum**2)

            # Leapfrog integration
            beta_prop, momentum_prop = self.leapfrog(beta, momentum, time, event, lr, cov0, epsilon, L)

            # Proposed state
            prop_log_prob = (
                self.log_partial_likelihood(beta_prop, time, event) * lr - 0.5 * np.sum(beta_prop**2) * inv_cov0_sq
            )
            prop_kinetic = 0.5 * np.sum(momentum_prop**2)

            # Metropolis acceptance
            log_accept = (prop_log_prob - prop_kinetic) - (current_log_prob - current_kinetic)

            if np.log(self.rng.random()) < log_accept:
                beta = beta_prop
                accept += 1

            beta_samples.append(beta.copy())

        if verbose:
            print(f"HMC acceptance rate: {accept/n_iter:.2f}")

        if trim_burn_in:
            return np.array(beta_samples)[burn_in:]

        return np.array(beta_samples)


class CoxNUTSampler(CoxSampler):
    """No-U-Turn Sampler for Cox regression with generalized Bayesian framework"""

    def __init__(self, covariates: NDArray, random_state: int = 2025):
        super().__init__(covariates, random_state)

    def compute_gradient(self, beta: NDArray, time: NDArray, event: NDArray, lr: float, cov0: float) -> NDArray:
        """Compute gradient of log posterior

        Args:
            beta (NDArray): parameters
            time (NDArray): time of occurring event
            event (NDArray): event flg (1: occurred)
            lr (float): learning rate
            cov0 (float): covariance of prior distribution

        Returns:
            NDArray: gradient
        """
        linpred = self.covariates @ beta
        exp_lp = np.exp(linpred)
        mask = event == 1

        t_evt, inv = np.unique(time[mask], return_inverse=True)
        cnt_evt = np.bincount(inv)

        order = np.argsort(time)
        X_sorted = self.covariates[order]
        exp_sorted = exp_lp[order]
        rev_cum_w = np.cumsum(exp_sorted[::-1])[::-1]
        rev_cum_Xw = np.cumsum((X_sorted * exp_sorted[:, None])[::-1], axis=0)[::-1]

        first_pos = np.searchsorted(time[order], t_evt, side="left")
        S0 = rev_cum_w[first_pos]
        S1 = rev_cum_Xw[first_pos]

        # Vectorized X_evt computation
        X_evt = np.zeros((len(t_evt), self.dim_covariates))
        for idx, _ in enumerate(t_evt):
            indices = inv == idx
            X_evt[idx] = self.covariates[mask][indices].sum(axis=0)

        grad_like = X_evt.sum(axis=0) - (cnt_evt[:, None] * (S1 / S0[:, None])).sum(axis=0)
        grad_prior = -beta / (cov0**2)
        grad = lr * grad_like + grad_prior
        return grad

    def leapfrog_step(
        self,
        beta: NDArray,
        momentum: NDArray,
        time: NDArray,
        event: NDArray,
        lr: float,
        cov0: float,
        epsilon: float,
    ) -> Tuple[NDArray, NDArray]:
        """Single leapfrog step

        Args:
            beta (NDArray): parameters
            momentum (NDArray): momentum
            time (NDArray): time of occurring event
            event (NDArray): event flg (1: occurred)
            lr (float): learning rate
            cov0 (float): covariance of prior distribution
            epsilon (float): epsilon

        Returns:
            Tuple[NDArray, NDArray]:
                - beta_new
                - momentum_new
        """
        grad = self.compute_gradient(beta, time, event, lr, cov0)
        momentum_half = momentum + 0.5 * epsilon * grad
        beta_new = beta + epsilon * momentum_half
        grad_new = self.compute_gradient(beta_new, time, event, lr, cov0)
        momentum_new = momentum_half + 0.5 * epsilon * grad_new
        return beta_new, momentum_new

    def build_tree(
        self,
        beta: NDArray,
        momentum: NDArray,
        log_u: float,
        v: int,
        j: int,
        time: NDArray,
        event: NDArray,
        lr: float,
        cov0: float,
        epsilon: float,
        log_joint0: float,
        delta_max: float = 1000.0,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, int, int, float, int]:
        """Build binary tree for NUTS

        Args:
            beta (NDArray): parameters
            momentum (NDArray): momentum
            log_u (float): log u
            v (int): v
            j (int): j
            time (NDArray): time of occurring event
            event (NDArray): event flg (1: occurred)
            lr (float): learning rate
            cov0 (float): covariance of prior distribution
            epsilon (float): epsilon
            log_joint0 (float): log joint
            delta_max (float, optional): delta max. Defaults to 1000.0.

        Returns:
            Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, int, int, float, int]:
                - beta_new
                - momentum_new
                - beta_m
                - momentum_m
                - beta_p
                - momentum_p
                - beta_prime
                - n_prime
                - s_prime
                - alpha_sum
                - n_alpha
        """
        if j == 0:
            # BASE CASE
            beta_new, momentum_new = self.leapfrog_step(beta, momentum, time, event, lr, cov0, v * epsilon)

            log_prob = self.log_partial_likelihood(beta_new, time, event) * lr - 0.5 * np.sum(beta_new**2) / (cov0**2)
            kinetic = 0.5 * np.sum(momentum_new**2)
            log_joint = log_prob - kinetic

            s_prime = int(log_joint > log_u - delta_max)
            n_prime = int(log_joint > log_u)
            alpha = min(1.0, np.exp(log_joint - log_joint0))

            return (beta_new, momentum_new, beta_new, momentum_new, beta_new, n_prime, s_prime, alpha, 1)
        else:
            # RECURSIVE CASE
            (beta_m, momentum_m, beta_p, momentum_p, beta_prime, n_prime, s_prime, alpha_sum, n_alpha) = (
                self.build_tree(beta, momentum, log_u, v, j - 1, time, event, lr, cov0, epsilon, log_joint0, delta_max)
            )

            if s_prime == 1:
                if v == -1:
                    (beta_m, momentum_m, _, _, beta_dprime, n_dprime, s_dprime, alpha_dprime, n_alpha_dprime) = (
                        self.build_tree(
                            beta_m, momentum_m, log_u, v, j - 1, time, event, lr, cov0, epsilon, log_joint0, delta_max
                        )
                    )
                else:
                    (_, _, beta_p, momentum_p, beta_dprime, n_dprime, s_dprime, alpha_dprime, n_alpha_dprime) = (
                        self.build_tree(
                            beta_p, momentum_p, log_u, v, j - 1, time, event, lr, cov0, epsilon, log_joint0, delta_max
                        )
                    )

                if n_dprime > 0:
                    accept_prob = n_dprime / max(1, n_prime + n_dprime)
                    if self.rng.random() < accept_prob:
                        beta_prime = beta_dprime

                delta = beta_p - beta_m
                no_u_turn = int(np.dot(delta, momentum_m) >= 0) * int(np.dot(delta, momentum_p) >= 0)
                s_prime = s_dprime * no_u_turn
                n_prime += n_dprime
                alpha_sum += alpha_dprime
                n_alpha += n_alpha_dprime

            return (beta_m, momentum_m, beta_p, momentum_p, beta_prime, n_prime, s_prime, alpha_sum, n_alpha)

    def sample(
        self,
        time: NDArray,
        event: NDArray,
        n_iter: int = 1000,
        lr: float = 1.0,
        beta_init: Optional[NDArray] = None,
        prior_cov_value: Optional[float] = None,
        epsilon: float = 0.01,
        max_depth: int = 10,
        adapt_epsilon: bool = False,
        target_accept: float = 0.65,
        trim_burn_in: bool = False,
        burn_in: int = 500,
        verbose: bool = False,
    ) -> NDArray:
        """NUTS sampling for Cox regression with generalized Bayesian framework

        Args:
            time (NDArray): time of occurring event
            event (NDArray): event flg (1: occurred)
            n_iter (int, optional): iteration. Defaults to 1000.
            lr (float, optional): learning rate. Defaults to 1.0.
            beta_init (Optional[NDArray], optional): initial value of parameters. Defaults to None.
            prior_cov_value (Optional[float], optional): initial covariance value of normal distribution. Defaults to None.  # noqa: E501
            epsilon (float, optional): epsilon. Defaults to 0.01.
            max_depth (int, optional): max depth. Defaults to 10.
            adapt_epsilon (bool, optional): adapt epsilon. Defaults to False.
            target_accept (float, optional): target accept. Defaults to 0.65.
            trim_burn_in (bool, optional): trim burn-in. Defaults to False.
            burn_in (int, optional): burn-in. Defaults to 500.
            verbose (bool, optional): verbose. Defaults to False.

        Returns:
            NDArray: sampling result
        """
        beta = np.zeros(self.dim_covariates) if beta_init is None else beta_init.copy()
        cov0 = prior_cov_value or 100.0
        accept = 0

        beta_samples = []
        epsilon_samples = []
        tree_depth_samples = []

        if adapt_epsilon:
            mu = np.log(10 * epsilon)
            epsilon_bar = 1.0
            H_bar = 0.0
            gamma = 0.05
            t0 = 10.0
            kappa = 0.75

        for iter_num in range(n_iter):
            momentum = self.rng.standard_normal(self.dim_covariates)

            log_prob = self.log_partial_likelihood(beta, time, event) * lr - 0.5 * np.sum(beta**2) / (cov0**2)
            kinetic = 0.5 * np.sum(momentum**2)
            log_joint = log_prob - kinetic
            log_joint0 = log_joint

            log_u = log_joint - self.rng.exponential(1.0)

            beta_m = beta.copy()
            beta_p = beta.copy()
            momentum_m = momentum.copy()
            momentum_p = momentum.copy()

            j = 0
            beta_new = beta.copy()
            n = 1
            s = 1
            beta_old = beta.copy()

            while s == 1 and j < max_depth:
                v = 2 * self.rng.integers(2) - 1

                if v == -1:
                    (beta_m, momentum_m, _, _, beta_prime, n_prime, s_prime, alpha_sum, n_alpha) = self.build_tree(
                        beta_m, momentum_m, log_u, v, j, time, event, lr, cov0, epsilon, log_joint0
                    )
                else:
                    (_, _, beta_p, momentum_p, beta_prime, n_prime, s_prime, alpha_sum, n_alpha) = self.build_tree(
                        beta_p, momentum_p, log_u, v, j, time, event, lr, cov0, epsilon, log_joint0
                    )

                if s_prime == 1:
                    accept_prob = n_prime / (n + n_prime)
                    if self.rng.random() < accept_prob:
                        beta_new = beta_prime

                n += n_prime

                delta_minus = beta_m - beta
                delta_plus = beta_p - beta
                s = s_prime * int(np.dot(delta_minus, momentum_m) >= 0) * int(np.dot(delta_plus, momentum_p) >= 0)

                j += 1

            if not np.array_equal(beta_old, beta_new):
                accept += 1

            beta = beta_new

            if adapt_epsilon and iter_num < burn_in:
                alpha_avg = alpha_sum / max(n_alpha, 1)
                H_bar = (1.0 - 1.0 / (iter_num + 1 + t0)) * H_bar + (target_accept - alpha_avg) / (iter_num + 1 + t0)
                log_epsilon = mu - np.sqrt(iter_num + 1) / gamma * H_bar
                epsilon = np.exp(log_epsilon)
                eta = (iter_num + 1) ** (-kappa)
                epsilon_bar = np.exp((1 - eta) * np.log(epsilon_bar) + eta * log_epsilon)
            elif adapt_epsilon and iter_num == burn_in:
                epsilon = epsilon_bar
                if verbose:
                    print(f"Adapted epsilon: {epsilon:.6f}")

            beta_samples.append(beta.copy())
            epsilon_samples.append(epsilon)
            tree_depth_samples.append(j)

            if (iter_num + 1) % 100 == 0:
                recent_depth = np.mean(tree_depth_samples[-100:])
                if verbose:
                    print(
                        f"NUTS iteration {iter_num + 1}/{n_iter}, "
                        f"avg tree depth: {recent_depth:.1f}, "
                        f"epsilon: {epsilon:.6f}"
                    )
            accept += 1

        if verbose:
            print(f"NUTS acceptance rate: {accept/n_iter:.2f}")

        if trim_burn_in:
            return np.array(beta_samples)[burn_in:]

        return np.array(beta_samples)


class CoxMALASampler(CoxSampler):
    """Metropolis-Adjusted Langevin Algorithm for Cox regression with generalized Bayesian framework"""

    def __init__(self, covariates: NDArray, random_state: int = 2025):
        super().__init__(covariates, random_state)

    def compute_gradient(self, beta: NDArray, time: NDArray, event: NDArray, lr: float, cov0: float) -> NDArray:
        """Compute gradient of log posterior

        Args:
            beta (NDArray): parameters
            time (NDArray): time of occurring event
            event (NDArray): event flg (1: occurred)
            lr (float): learning rate
            cov0 (float): covariance of prior distribution

        Returns:
            NDArray: gradient
        """
        linpred = self.covariates @ beta
        exp_lp = np.exp(linpred)
        mask = event == 1

        t_evt, inv = np.unique(time[mask], return_inverse=True)
        cnt_evt = np.bincount(inv)

        order = np.argsort(time)
        X_sorted = self.covariates[order]
        exp_sorted = exp_lp[order]
        rev_cum_w = np.cumsum(exp_sorted[::-1])[::-1]
        rev_cum_Xw = np.cumsum((X_sorted * exp_sorted[:, None])[::-1], axis=0)[::-1]

        first_pos = np.searchsorted(time[order], t_evt, side="left")
        S0 = rev_cum_w[first_pos]
        S1 = rev_cum_Xw[first_pos]

        # Vectorized X_evt computation
        X_evt = np.zeros((len(t_evt), self.dim_covariates))
        for idx, _ in enumerate(t_evt):
            indices = inv == idx
            X_evt[idx] = self.covariates[mask][indices].sum(axis=0)

        grad_like = X_evt.sum(axis=0) - (cnt_evt[:, None] * (S1 / S0[:, None])).sum(axis=0)
        grad_prior = -beta / (cov0**2)
        grad = lr * grad_like + grad_prior
        return grad

    def sample(
        self,
        time: NDArray,
        event: NDArray,
        n_iter: int = 1000,
        lr: float = 1.0,
        beta_init: Optional[NDArray] = None,
        prior_cov_value: Optional[float] = None,
        step_size: float = 0.01,
        trim_burn_in: bool = False,
        burn_in: int = 500,
        verbose: bool = False,
    ) -> NDArray:
        """MALA sampling for Cox regression with generalized Bayesian framework

        Args:
            time (NDArray): time of occurring event
            event (NDArray): event flg (1: occurred)
            n_iter (int, optional): iteration. Defaults to 1000.
            lr (float, optional): learning rate. Defaults to 1.0.
            beta_init (Optional[NDArray], optional): initial value of parameters. Defaults to None.
            prior_cov_value (Optional[float], optional): initial covariance value of normal distribution. Defaults to None.  # noqa: E501
            step_size (float, optional): step size. Defaults to 0.01.
            trim_burn_in (bool, optional): trim burn-in. Defaults to False.
            burn_in (int, optional): burn-in. Defaults to 500.
            verbose (bool, optional): verbose. Defaults to False.

        Returns:
            NDArray: sampling result
        """
        beta = np.zeros(self.dim_covariates) if beta_init is None else beta_init.copy()
        cov0 = prior_cov_value or 100.0

        beta_samples = []
        accept = 0

        # Pre-compute constants
        sqrt_step = np.sqrt(step_size)
        half_step = 0.5 * step_size
        inv_step = 1.0 / step_size
        inv_cov0_sq = 1.0 / (cov0**2)

        for _ in range(n_iter):
            grad_current = self.compute_gradient(beta, time, event, lr, cov0)

            # Propose new state
            beta_prop = beta + half_step * grad_current + sqrt_step * self.rng.standard_normal(self.dim_covariates)

            grad_prop = self.compute_gradient(beta_prop, time, event, lr, cov0)

            # Log posterior
            log_prob_current = self.log_partial_likelihood(beta, time, event) * lr - 0.5 * np.sum(beta**2) * inv_cov0_sq
            log_prob_prop = (
                self.log_partial_likelihood(beta_prop, time, event) * lr - 0.5 * np.sum(beta_prop**2) * inv_cov0_sq
            )

            # Proposal densities
            diff_forward = beta_prop - (beta + half_step * grad_current)
            diff_backward = beta - (beta_prop + half_step * grad_prop)

            log_q_forward = -0.5 * np.sum(diff_forward**2) * inv_step
            log_q_backward = -0.5 * np.sum(diff_backward**2) * inv_step

            # Metropolis-Hastings acceptance
            log_accept = (log_prob_prop + log_q_backward) - (log_prob_current + log_q_forward)

            if np.log(self.rng.random()) < log_accept:
                beta = beta_prop
                accept += 1

            beta_samples.append(beta.copy())

        if verbose:
            print(f"MALA acceptance rate: {accept/n_iter:.2f}")

        if trim_burn_in:
            return np.array(beta_samples)[burn_in:]

        return np.array(beta_samples)
