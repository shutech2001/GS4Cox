from collections import deque
from typing import Optional, Tuple, Dict, Deque

import numpy as np
from polyagamma import random_polyagamma  # type: ignore


class CoxSampler:
    def __init__(self, covariates: np.ndarray):
        """
        Args:
            covariates (np.ndarray)
        """
        self.covariates: np.ndarray = covariates
        self.data_num: int
        self.dim_covariates: int
        self.data_num, self.dim_covariates = covariates.shape

    def build_risk_sets(
        self, time: np.ndarray, event: np.ndarray
    ) -> Tuple[Dict[float, np.ndarray], np.ndarray, Deque[int]]:
        """"Construct the atrisk set at each event occurrence time.

        Args:
            time (np.ndarray): observed time
            event (np.ndarray): event indicator

        Returns:
            Tuple[Dict[float, np.ndarray], np.ndarray, Deque[int]]:
                - dictionary of event time and index
                    e.g., {1.0: np.array([2, 4, 5, 0, 1, 3])}
                - time of occurring event
                - the number of events at that time
        """
        event_times: np.ndarray = np.sort(np.unique(time[event == 1]))
        risk_sets: dict[float, np.ndarray] = {}
        event_nums: deque[int] = deque()
        for t in event_times:
            atrisk_idxs: np.ndarray = np.where(time >= t)[0]
            event_idxs: np.ndarray = np.where((time == t) & (event == 1))[0]
            event_nums.append(event_idxs.size)
            if event_idxs.size > 0:
                no_event_idxs: np.ndarray = np.setdiff1d(atrisk_idxs, event_idxs, assume_unique=True)
                sorted_risk_sets: np.ndarray = np.concatenate([event_idxs, np.sort(no_event_idxs)])
            else:
                sorted_risk_sets = np.sort(atrisk_idxs)
            risk_sets[t] = sorted_risk_sets
        return risk_sets, event_times, event_nums


class CoxPGSampler(CoxSampler):
    """Cox-P\'olya-Gamma Gibbs sampler
    """
    def __init__(self, covariates: np.ndarray) -> None:
        super().__init__(covariates)

    def _compute_event_stats(
        self,
        event_idx: int,
        at_risk_idxs: np.ndarray,
        beta0: np.ndarray,
        beta: np.ndarray,
    ) -> Tuple[float, np.ndarray, float]:
        """Compute the values of eta, tilde x, and offset for an event in the at risk

        Args:
            event_idx (int): index of a person who caused the event
            at_risk_idxs (np.ndarray): indexes of atrisk set
            beta0 (np.ndarray): β before the update
            beta (np.ndarray): β after the update

        Returns:
            Tuple[float, np.ndarray, float]:
                - eta_i
                - tilde x_i
                - offset_i
        """
        event_cov: np.ndarray = self.covariates[event_idx, :]
        event_linpred: float = event_cov.dot(beta)

        # atrisk excluding oneself
        other_idxs: np.ndarray = at_risk_idxs[at_risk_idxs != event_idx]
        other_cov: np.ndarray = self.covariates[other_idxs, :]
        # sum of exponential of linear predictors
        sum_exp_other_linpred: float = np.exp(other_cov.dot(beta)).sum()
        eta: float = event_linpred - np.log(sum_exp_other_linpred)

        # local linearization with \beta_0
        exp_other_linpred_center: np.ndarray = np.exp(other_cov.dot(beta0))
        sum_exp_other_linpred_center: float = exp_other_linpred_center.sum()
        # calculate \eta \approx \tilde_x - offset
        weighted_avg_other_cov: np.ndarray = (
            exp_other_linpred_center[:, np.newaxis] * other_cov
        ).sum(axis=0) / sum_exp_other_linpred_center
        tilde_x: np.ndarray = event_cov - weighted_avg_other_cov
        offset: float = np.log(sum_exp_other_linpred_center) - weighted_avg_other_cov.dot(beta0)
        return eta, tilde_x, offset

    def compute_event_contribution(
        self,
        beta0: np.ndarray,
        beta: np.ndarray,
        risk_sets: dict[float, np.ndarray],
        event_times: np.ndarray,
        event_nums: Deque[int]
    ) -> Tuple[Deque[float], Deque[np.ndarray], Deque[float]]:
        """Compute elements for post distribution

        Args:
            beta0 (np.ndarray): beta of previous iteration
            beta (np.ndarray): beta of current iteration
            risk_sets (dict[float, np.ndarray]): dictionary of event time and index
            event_times (np.ndarray): time of occurring event
            event_nums (deque[int]): the number of events at that time

        Returns:
            tuple[deque[float], deque[np.ndarray], deque[float]]:
                - list of eta
                - list of tilde x
                - list of offset
        """
        eta_list: deque[float] = deque()
        tilde_x_list: deque[np.ndarray] = deque()
        offset_list: deque[float] = deque()
        for event_num, event_time in zip(event_nums, event_times):
            at_risk_idxs: np.ndarray = risk_sets[event_time]
            if at_risk_idxs.size < 2:
                # if only one person at risk, no contribution to the likelihood
                continue
            # calculate \sum_{j\i\in d(t)} \exp(x_i^\top \beta)
            event_idxs: np.ndarray = at_risk_idxs[:event_num]
            for event_idx in event_idxs:
                eta, tilde_x, offset = self._compute_event_stats(event_idx, at_risk_idxs, beta0, beta)
                eta_list.append(eta)
                tilde_x_list.append(tilde_x)
                offset_list.append(offset)

        return eta_list, tilde_x_list, offset_list

    def cox_pg_sample(
        self,
        time: np.ndarray,
        event: np.ndarray,
        n_iter: int = 1000,
        burn_in: int = 500,
        lr: float = 1.0,
        beta_init: Optional[np.ndarray] = None,
        prior_mean: Optional[np.ndarray] = None,
        prior_cov: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sampling by Cox P'olya Gamma Gibbs sampler

        Args:
            time (np.ndarray): time of occurring event
            event (np.ndarray): event flg (1: occurred)
            n_iter (int, optional): iteration of sampling. Defaults to 1000.
            burn_in (int, optional): burn in of sampling. Defaults to 500.
            lr (float, optional): learning rate for generalized Bayesian framework. Defaults to 1.0.
            beta_init (Optional[np.ndarray], optional): initial beta. Defaults to None.
            prior_mean (Optional[np.ndarray], optional): mean of prior distribution. Defaults to None.
            prior_cov (Optional[np.ndarray], optional): covariance of prior distribution. Defaults to None.

        Returns:
            np.ndarray:
                - β samples excluding the initial burn-in iterations
        """
        mean0: np.ndarray = prior_mean if prior_mean is not None else np.zeros(self.dim_covariates)
        cov0: np.ndarray = prior_cov if prior_cov is not None else np.eye(self.dim_covariates) * 100
        beta0: np.ndarray = beta_init.copy() if beta_init is not None else np.zeros(self.dim_covariates)
        beta: np.ndarray = np.random.multivariate_normal(mean0, cov0)

        beta_samples: deque[np.ndarray] = deque()
        risk_sets, event_times, event_nums = self.build_risk_sets(time, event)

        for _ in range(n_iter):
            eta_list, tilde_x_list, offset_list = self.compute_event_contribution(
                beta0, beta, risk_sets, event_times, event_nums
            )
            # sampling from P\'olya gamma distribution
            omega_array: np.ndarray = np.array([random_polyagamma(1, eta) for eta in eta_list])
            # correct kappa associated with the local linearization
            kappa_array: np.ndarray = 0.5 + omega_array * np.array(offset_list)

            # calculate additional term of posterior distribution
            add_cov: np.ndarray = np.sum(
                [omega * np.outer(x, x) for x, omega in zip(tilde_x_list, omega_array)], axis=0
            )
            add_mean: np.ndarray = np.sum(
                [tilde_x * (offset * omega + kappa) for tilde_x, offset, omega, kappa in zip(
                    tilde_x_list, offset_list, omega_array, kappa_array
                )],
                axis=0
            )
            # calculate parameters of post distribution
            post_cov: np.ndarray = np.linalg.inv(np.linalg.inv(cov0) + lr * add_cov)
            post_mean: np.ndarray = post_cov.dot(np.linalg.inv(cov0).dot(mean0) + lr * add_mean)

            # store previous beta for next iteration's local linearization
            _beta0 = beta.copy()
            try:
                beta = np.random.multivariate_normal(post_mean, post_cov)
                beta_samples.append(beta)
            except np.linalg.LinAlgError as e:
                print(str(e))
                beta_samples.append(_beta0)
            finally:
                beta0 = _beta0

        return np.array(beta_samples)[burn_in:]


class CoxMHSampler(CoxSampler):
    def __init__(self, covariates: np.ndarray) -> None:
        super().__init__(covariates)

    def log_partial_likelihood(
        self,
        beta: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
    ) -> float:
        """Compute log partial likelihood

        Args:
            beta (np.ndarray): parameters
            time (np.ndarray): time of occurring event
            event (np.ndarray): event flg (1: occurred)

        Returns:
            float: value of log partial likelihood
        """
        log_pl: float = 0.0
        # unique value of event times
        event_times: np.ndarray = np.unique(time[event == 1])
        for t in event_times:
            # who happen event
            event_idxs: np.ndarray = np.where((time == t) & (event == 1))[0]
            sum_event_linpred: float = np.sum(self.covariates[event_idxs, :].dot(beta))
            # set of at risk
            at_risk_idx: np.ndarray = np.where(time >= t)[0]
            sum_exp_at_risk_linpred: float = np.sum(np.exp(self.covariates[at_risk_idx, :].dot(beta)))

            log_pl += sum_event_linpred - len(event_idxs) * np.log(sum_exp_at_risk_linpred)
        return log_pl

    def log_pl_posterior(
        self,
        beta: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        lr: float = 1.0,
        cov0: float = 10.0
    ) -> float:
        """Compute log posterior likelihood in generalized bayesian inference

        Args:
            beta (np.ndarray): parameters
            time (np.ndarray): time of occurring event
            event (np.ndarray): event flg (1: occurred)
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

    def cox_mh_sample(
        self,
        time: np.ndarray,
        event: np.ndarray,
        n_iter: int = 1000,
        burn_in: int = 500,
        lr: float = 1.0,
        beta_init: Optional[np.ndarray] = None,
        prior_cov_value: Optional[float] = None,
        proposal_scale: float = 10,
    ) -> Tuple[np.ndarray, float]:
        """Sampling by Metropolis-Hastings algorithm

        Args:
            time (np.ndarray): time of occurring event
            event (np.ndarray): event flg (1: occurred)
            n_iter (int, optional): iteration of sampling. Defaults to 1000.
            burn_in (int, optional): burn in of sampling. Defaults to 500.
            lr (float, optional): learning rate for generalized Bayesian framework. Defaults to 1.0.
            beta_init (Optional[np.ndarray], optional): initial beta. Defaults to None.
            prior_cov_value (Optional[float], optional): covariance of prior distribution. Defaults to None.
            proposal_scale (Optional[float], optional): covariance of proposal distribution. Defaults to 10.

        Returns:
            Tuple[np.ndarray, float]:
                - β samples excluding the initial burn-in iterations
                - acceptance rate
        """
        beta_samples: Deque[np.ndarray] = deque()
        cov0: float = prior_cov_value if prior_cov_value is not None else 100
        beta: np.ndarray = beta_init.copy() if beta_init is not None else np.zeros(self.dim_covariates)

        log_pl_post: float = self.log_pl_posterior(beta, time, event, lr, cov0)
        accept_count: int = 0

        for _ in range(n_iter):
            # proposal distribution: multi variable normal distribution
            beta_proposal: np.ndarray = beta + np.random.normal(0, proposal_scale, size=self.dim_covariates)
            log_pl_post_proposal: float = self.log_pl_posterior(beta_proposal, time, event, lr, cov0)
            # accept probability for log
            log_alpha: float = log_pl_post_proposal - log_pl_post
            # compare with log(Uniform(0,1))
            if np.log(np.random.uniform(0, 1)) < log_alpha:
                beta = beta_proposal
                log_pl_post = log_pl_post_proposal
                accept_count += 1
            beta_samples.append(beta)

        accept_rate: float = accept_count / n_iter
        return np.array(beta_samples)[burn_in:], accept_rate
