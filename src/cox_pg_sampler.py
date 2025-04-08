from collections import deque
from tqdm import tqdm
from typing import Optional

import numpy as np
from polyagamma import random_polyagamma  # type: ignore


class CoxPGSampler:
    def __init__(self, covariates: np.ndarray):
        """
        Args:
            covariates (np.ndarray)
        """
        self.covariates: np.ndarray = covariates
        self.data_num: int = covariates.shape[0]
        self.dim_covariates: int = covariates.shape[1]

    def build_risk_sets(
        self, time: np.ndarray, event: np.ndarray
    ) -> tuple[dict[float, np.ndarray], np.ndarray, deque[int]]:
        """_summary_

        Args:
            time (np.ndarray): _description_
            event (np.ndarray): _description_

        Returns:
            tuple[dict[float, np.ndarray], np.ndarray, deque[int]]:
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
            event_nums.append(len(event_idxs))
            if len(event_idxs) > 0:
                no_event_idxs: np.ndarray = np.setdiff1d(atrisk_idxs, event_idxs, assume_unique=True)
                new_order: np.ndarray = np.concatenate([event_idxs, np.sort(no_event_idxs)])
            else:
                new_order = np.sort(atrisk_idxs)
            risk_sets[t] = new_order
        return risk_sets, event_times, event_nums

    def compute_event_contribution(
        self,
        beta0: np.ndarray,
        beta: np.ndarray,
        risk_sets: dict[float, np.ndarray],
        event_times: np.ndarray,
        event_nums: deque[int]
    ) -> tuple[deque[float], deque[np.ndarray], deque[float]]:
        """compute elements for post distribution

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
            indices: np.ndarray = risk_sets[event_time]
            if len(indices) < 2:
                # if only one person at risk, no contribution to the likelihood
                continue

            # calculate \sum_{j\i\in d(t)} \exp(x_i^\top \beta)
            event_idxs: np.ndarray = indices[:event_num]

            for event_accumulated_num, event_idx in enumerate(event_idxs):
                one_event_cov: np.ndarray = self.covariates[event_idx, :]
                one_event_linpred: float = one_event_cov.dot(beta)
                at_risks_exclude_one_event_idx: np.ndarray = indices[indices != event_idx]
                at_risks_exclude_one_event_cov = self.covariates[at_risks_exclude_one_event_idx, :]
                at_risks_exclude_one_event_linpred: np.ndarray = at_risks_exclude_one_event_cov.dot(beta)
                exp_at_risks_exclude_one_event_linpred: np.ndarray = np.exp(at_risks_exclude_one_event_linpred)
                sum_exp_at_risks_exclude_one_event_linpred: float = sum(exp_at_risks_exclude_one_event_linpred)

                at_risks_exclude_one_event_linpred_center: np.ndarray = at_risks_exclude_one_event_cov.dot(beta0)
                exp_at_risks_exclude_one_event_linpred_center: np.ndarray = np.exp(at_risks_exclude_one_event_linpred_center)  # noqa: E501
                sum_exp_at_risks_exclude_one_event_linpred_center: float = sum(exp_at_risks_exclude_one_event_linpred_center)  # noqa: E501

                eta_list.append(one_event_linpred - np.log(sum_exp_at_risks_exclude_one_event_linpred))

                numerator: np.ndarray = sum(
                    e * at_risks_exclude_one_event_cov[i, :] for i, e in enumerate(exp_at_risks_exclude_one_event_linpred_center)  # noqa: E501
                )
                tilde_x_list.append(
                    self.covariates[event_idx, :] - (numerator/sum_exp_at_risks_exclude_one_event_linpred_center)  # noqa: E501
                )

                offset_list.append(
                    np.log(sum_exp_at_risks_exclude_one_event_linpred_center) - (numerator/sum_exp_at_risks_exclude_one_event_linpred_center).dot(beta0)  # noqa: E501
                )

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
        """sampling by Cox Polya Gamma Gibbs sampler

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
            np.ndarray: _description_
        """
        if prior_mean is None:
            prior_mean = np.zeros(self.dim_covariates)
        if prior_cov is None:
            prior_cov = np.eye(self.dim_covariates) * 100
        if beta_init is None:
            beta0: np.ndarray = np.zeros(self.dim_covariates)
            beta: np.ndarray = np.random.multivariate_normal(prior_mean, prior_cov)
        else:
            beta0 = beta_init.copy()
            beta = np.random.multivariate_normal(prior_mean, prior_cov)

        beta_samples: deque[np.ndarray] = deque()
        risk_sets, event_times, event_nums = self.build_risk_sets(time, event)

        for it in tqdm(range(n_iter)):
            eta_list, tilde_x_list, offset_list = self.compute_event_contribution(
                beta0, beta, risk_sets, event_times, event_nums
            )
            # sampling from Poly\'a gamma distribution
            omega_array: np.ndarray = np.array([random_polyagamma(1, eta) for eta in eta_list])
            # correct \kappa associated with the local linearization
            kappa_array: np.ndarray = 0.5 + omega_array*np.array(offset_list)

            # calculate additional term of post distribution
            add_cov: np.ndarray = sum(omega * np.outer(x, x) for x, omega in zip(tilde_x_list, omega_array))
            add_mean: np.ndarray = sum(
                tilde_x * (offset * omega + kappa)
                for tilde_x, offset, omega, kappa in zip(tilde_x_list, offset_list, omega_array, kappa_array)
            )
            # calculate parameters of post distribution
            post_cov: np.ndarray = np.linalg.inv(np.linalg.inv(prior_cov) + lr*add_cov)
            post_mean: np.ndarray = post_cov.dot(np.linalg.inv(prior_cov).dot(prior_mean) + lr*add_mean)

            # store previous beta for next iteration's local linearization
            beta0 = beta.copy()
            beta = np.random.multivariate_normal(post_mean, post_cov)
            beta_samples.append(beta)

        _beta_samples: np.ndarray = np.array(beta_samples)
        return _beta_samples[burn_in:]
