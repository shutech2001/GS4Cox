from __future__ import annotations

from collections import deque
from typing import Optional, Tuple, Dict, Deque

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve  # type: ignore
from polyagamma import random_polyagamma  # type: ignore


class CoxSampler:
    """Parent Class of Cox MH/Gibbs Sampler
    """
    def __init__(self, covariates: NDArray):
        """
        Args:
            covariates (NDArray)
        """
        self.covariates: NDArray = covariates
        self.data_num: int
        self.dim_covariates: int
        self.data_num, self.dim_covariates = covariates.shape

    def build_risk_sets(
        self, time: NDArray, event: NDArray
    ) -> Tuple[Dict[float, NDArray], NDArray, Deque[int]]:
        """"Construct the risk set at each event occurrence time.

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


class GBPairwiseCoxPGSampler(CoxSampler):
    """Cox-P\'olya-Gamma Gibbs sampler in general Bayesian framework with Pairwise likelihood
    """
    def __init__(self, covariates: NDArray) -> None:
        super().__init__(covariates)

    def _build_pairs(
        self,
        risk_sets: Dict[float, NDArray],
        event_times: NDArray,
        event_nums: Deque[int],
    ) -> Tuple[NDArray, NDArray]:
        """
        Vectorized construction of all (i,j) pairs for pairwise likelihood.
        Returns arrays of event indices and risk-set indices.

        Args:
            Returns of `build_risk_sets`
                - risk_sets (Dict[float, NDArray])
                - event_times (NDArray)
                - event_nums (Deque[int])

        Returns:
            Tuple[NDArray, NDArray]:
                - event indices
                - risk-set indices
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
            pairs_i = np.array([])
            pairs_j = np.array([])
        return pairs_i, pairs_j

    def gb_pairwise_cox_pg_sample(
        self,
        time: NDArray,
        event: NDArray,
        n_iter: int = 1000,
        lr: float = 1.0,
        beta_init: Optional[NDArray] = None,
        prior_mean: Optional[NDArray] = None,
        prior_cov: Optional[NDArray] = None,
    ) -> NDArray:
        """Sampling via PG-augmented pairwise Cox Partial likelihood with vectorized updates.

        Args:
            time (NDArray): observed time
            event (NDArray): event indicator
            n_iter (int, optional): iteration of sampling. Defaults to 1000.
            lr (float, optional): learning rate in generalized Bayesian framework. Defaults to 1.0.
            beta_init (Optional[NDArray], optional): initial value of parameters. Defaults to None.
            prior_mean (Optional[NDArray], optional): mean of prior normal distribution. Defaults to None.
            prior_cov (Optional[NDArray], optional): covariance matrix of prior normal distribution. Defaults to None.

        Returns:
            NDArray: sampling result
        """
        # set priors
        mean0: NDArray = prior_mean if prior_mean is not None else np.zeros(self.dim_covariates)
        cov0: NDArray = prior_cov if prior_cov is not None else np.eye(self.dim_covariates) * 100
        inv_cov0: NDArray = np.linalg.inv(cov0)
        beta: NDArray = beta_init.copy() if beta_init is not None else np.zeros(self.dim_covariates)
        beta_samples: Deque[NDArray] = deque()

        # risk set and pair precomputation (only once)
        risk_sets, event_times, event_nums = self.build_risk_sets(time, event)
        pairs_i, pairs_j = self._build_pairs(risk_sets, event_times, event_nums)
        # delta matrix: shape (P, d)
        D: NDArray = self.covariates[pairs_i] - self.covariates[pairs_j]
        kappa: float = 0.5

        for _ in range(n_iter):
            # vectorized dot products
            psi: NDArray = D.dot(beta)
            # batch PG sampling
            omega: NDArray = random_polyagamma(1, psi)
            # compute weighted covariance and mean
            W: NDArray = omega[:, None]
            add_cov: NDArray = D.T.dot(D * W)
            add_mean: NDArray = kappa * D.sum(axis=0)

            # posterior precision
            post_prec: NDArray = inv_cov0 + lr * add_cov
            c, lower = cho_factor(post_prec, check_finite=False)
            post_cov = cho_solve((c, lower), np.eye(self.dim_covariates), check_finite=False)
            rhs: NDArray = inv_cov0.dot(mean0) + lr * add_mean
            post_mean: NDArray = cho_solve((c, lower), rhs, check_finite=False)

            # sample new beta
            beta = np.random.multivariate_normal(post_mean, post_cov)
            beta_samples.append(beta)

        return np.vstack(beta_samples)


class CoxMHSampler(CoxSampler):
    """Metropolis sampler for Cox regression model in general Bayesian framework
    """
    def __init__(self, covariates: np.ndarray) -> None:
        super().__init__(covariates)

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
        # 1. Xβ を一度だけ計算
        linpred = self.covariates.dot(beta)  # shape (n,)
        # 2. イベント時刻でグループ化
        mask = event == 1
        t_evt, idx_evt = np.unique(time[mask], return_inverse=True)
        # sum of linpred for each unique event time
        sum_evt = np.bincount(idx_evt, weights=linpred[mask])
        # カウント
        cnt_evt = np.bincount(idx_evt)
        # 3. リスク集合の和をソート + 累積和
        #    ソートは「昇順」にしておき，後ろから累積和を取る
        order = np.argsort(time)
        sorted_time = time[order]
        sorted_lp = linpred[order]
        cum_exp = np.cumsum(np.exp(sorted_lp[::-1]))[::-1]
        # 各 unique イベント時刻 t に対し、最初に出現する位置を探す
        first_pos = np.searchsorted(sorted_time, t_evt, side='left')
        risk_sums = cum_exp[first_pos]
        # 最終的な対数部分尤度
        return np.sum(sum_evt - cnt_evt * np.log(risk_sums))

    def log_pl_posterior(
        self,
        beta: NDArray,
        time: NDArray,
        event: NDArray,
        lr: float = 1.0,
        cov0: float = 10.0
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

    def cox_mh_sample(
        self,
        time: NDArray,
        event: NDArray,
        n_iter: int = 1000,
        lr: float = 1.0,
        beta_init: Optional[NDArray] = None,
        prior_cov_value: Optional[float] = None,
        proposal_scale: float = 10,
    ) -> Tuple[NDArray, float]:
        """Sampling by Metropolis-Hastings algorithm

        Args:
            time (NDArray): time of occurring event
            event (NDArray): event flg (1: occurred)
            n_iter (int, optional): iteration of sampling. Defaults to 1000.
            lr (float, optional): learning rate for generalized Bayesian framework. Defaults to 1.0.
            beta_init (Optional[NDArray], optional): initial beta. Defaults to None.
            prior_cov_value (Optional[float], optional): covariance of prior distribution. Defaults to None.
            proposal_scale (Optional[float], optional): covariance of proposal distribution. Defaults to 10.

        Returns:
            Tuple[NDArray, float]:
                - β samples excluding the initial burn-in iterations
                - acceptance rate
        """
        beta_samples: Deque[NDArray] = deque()
        cov0: float = prior_cov_value if prior_cov_value is not None else 100
        beta: NDArray = beta_init.copy() if beta_init is not None else np.zeros(self.dim_covariates)
        prop_cov: NDArray = np.eye(self.dim_covariates) * proposal_scale

        log_pl_post: float = self.log_pl_posterior(beta, time, event, lr, cov0)
        accept_count: int = 0

        for _ in range(n_iter):
            # proposal distribution: multi variable normal distribution
            beta_proposal: NDArray = np.random.multivariate_normal(beta, prop_cov)
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
        return np.array(beta_samples), accept_rate

    def _score_and_hessian(
        self,
        beta: NDArray,
        time: NDArray,
        event: NDArray,
        lr: float,
        cov0: float,
    ) -> Tuple[NDArray, NDArray]:
        """解析的にスコア関数（勾配）とヘッセ行列を同時計算"""
        linpred = self.covariates.dot(beta)      # (n,)
        exp_lp = np.exp(linpred)                # (n,)
        mask = event == 1
        # unique event times
        t_evt, inv = np.unique(time[mask], return_inverse=True)
        cnt_evt = np.bincount(inv)              # (#events,)

        # ソート＋累積和でリスク集合の分母と一次モーメント、二次モーメントを取得
        order = np.argsort(time)
        X_sorted = self.covariates[order]        # (n, p)
        exp_sorted = exp_lp[order]               # (n,)
        rev_cum_w = np.cumsum(exp_sorted[::-1])[::-1]            # (n,)
        rev_cum_Xw = np.cumsum((X_sorted * exp_sorted[:, None])[::-1], axis=0)[::-1]  # (n, p)
        rev_cum_XXw = np.cumsum(
            ((X_sorted[:, :, None] * X_sorted[:, None, :]) * exp_sorted[:, None, None])[::-1], axis=0
        )[::-1]  # (n,p,p)

        # 各 unique 時刻 t の最初の位置
        first_pos = np.searchsorted(time[order], t_evt, side='left')
        # 分母
        S0 = rev_cum_w[first_pos]               # (#events,)
        # 一次モーメント
        S1 = rev_cum_Xw[first_pos]              # (#events, p)
        # 二次モーメント
        S2 = rev_cum_XXw[first_pos]             # (#events, p, p)

        # スコア = Σ_i [ x_i − cnt_i * (S1_i / S0_i) ]
        X_evt = np.zeros((len(t_evt), self.dim_covariates))
        # 各 unique 時刻のイベント集合ごとの平均 x の和
        for k, tval in enumerate(t_evt):
            X_evt[k] = self.covariates[(time == tval) & mask].sum(axis=0)
        score = X_evt.sum(axis=0) - (cnt_evt[:, None] * (S1 / S0[:, None])).sum(axis=0)
        # 事前分散成分
        score -= beta / (cov0**2)
        score *= lr

        # ヘッセ行列 = −lr * Σ_i cnt_i * [ (S2/S0) − (S1⊗S1)/S0^2 ]  − I/cov0^2
        H: NDArray = np.zeros((self.dim_covariates, self.dim_covariates))
        for k in range(len(t_evt)):
            E_xx = S2[k] / S0[k]
            E_x = S1[k] / S0[k]
            H += cnt_evt[k] * (E_xx - np.outer(E_x, E_x))
        H = - lr * H
        # 事前分散ヘッセ
        H -= np.eye(self.dim_covariates) / (cov0**2)

        return score, H

    def cox_mh_with_hessian_sample(
        self,
        time: NDArray,
        event: NDArray,
        n_iter: int = 1000,
        lr: float = 1.0,
        beta_init: Optional[NDArray] = None,
        prior_cov_value: Optional[float] = None,
        scaling: float = 1.0,
    ) -> NDArray:
        # 省略：beta_samples, 初期値設定は同様
        beta = np.zeros(self.dim_covariates) if beta_init is None else beta_init.copy()
        cov0 = prior_cov_value or 100.0
        lp_post = self.log_partial_likelihood(beta, time, event) * lr - 0.5 * np.sum(beta**2)/(cov0**2)
        accept = 0
        beta_samples = []

        for _ in range(n_iter):
            # 解析的スコア＋ヘッセを取得
            score, H = self._score_and_hessian(beta, time, event, lr, cov0)
            # 提案共分散
            try:
                cov_prop = scaling * np.linalg.inv(-H)
                # 正定値確認
                np.linalg.cholesky(cov_prop)
            except np.linalg.LinAlgError:
                cov_prop = scaling * np.eye(self.dim_covariates) * 0.1

            # MHステップ
            beta_prop = beta + np.random.multivariate_normal(np.zeros(self.dim_covariates), cov_prop)
            lp_prop = self.log_partial_likelihood(beta_prop, time, event) * lr - 0.5 * np.sum(beta_prop**2)/(cov0**2)
            if np.log(np.random.rand()) < (lp_prop - lp_post):
                beta, lp_post = beta_prop, lp_prop
                accept += 1

            beta_samples.append(beta)
        return np.array(beta_samples)
