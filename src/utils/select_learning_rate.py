from __future__ import annotations

from tqdm import tqdm  # type: ignore
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Type, Tuple
import numpy as np
from numpy.typing import NDArray

from cox_sampler import CoxSampler


class SelectLearningRate:
    """Select learning rate for Cox regression model in general Bayesian framework Class
    """
    def __init__(
        self,
        Sampler: Type[CoxSampler],
        sampling_method_name: str,
        covariates: NDArray,
        time: NDArray,
        event: NDArray,
        n_jobs: int = 1,
    ) -> None:
        """
        Args:
            Sampler (Type[CoxSampler]): Sampler Class for Cox regression model defined in cox_sampler.py
            sampling_method_name (str): Method name of sampling
            covariates (NDArray): covariates numpy array
            time (NDArray): observed time
            event (NDArray): event indicator
            n_jobs (int): number of parallel processing
        """
        self.Sampler: Type[CoxSampler] = Sampler
        self.sampling_method_name: str = sampling_method_name
        self.covariates: NDArray = covariates
        self.time: NDArray = time
        self.event: NDArray = event
        self.data_num: int
        self.dim_covariates: int
        self.data_num, self.dim_covariates = covariates.shape
        self.n_jobs: int = n_jobs

    def compute_region(
        self,
        args: Tuple[float, float, int, int],
    ) -> Tuple[NDArray, NDArray]:
        """Compute alpha/2 - 1-alpha/2 quantiles

        Args:
            args:
                eta (float): previous learning rate
                alpha (float): (1-alpha) credible interval
                n_iter (int): iteration of sampling
                burn_in (int): burn-in from sampling

        Returns:
            Tuple[NDArray, NDArray]:
                - lower bounds of (1-alpha) credible interval
                - upper bounds of (1-alpha) credible interval
        """
        eta, alpha, n_iter, burn_in = args
        # draw bootstrap indices
        idx = np.random.choice(self.data_num, size=self.data_num, replace=True)
        time_b, event_b, covariates_b = self.time[idx], self.event[idx], self.covariates[idx]

        # re‑instantiate the sampler (or just re‑set lr)
        sampler_b: CoxSampler = self.Sampler(covariates_b)
        sampling_method = getattr(sampler_b, self.sampling_method_name)
        # run whichever sampling you use:
        chain_b: NDArray = sampling_method(time_b, event_b, n_iter=n_iter, burn_in=burn_in, lr=eta)
        chain_b = chain_b[burn_in:]

        # compute the α/2 and 1-α/2 quantiles
        lower: NDArray = np.quantile(chain_b, alpha/2, axis=0)
        upper: NDArray = np.quantile(chain_b, 1 - alpha/2, axis=0)
        return lower, upper

    def select_eta_gpc(
        self,
        point_estimate: NDArray,
        alpha: float = 0.05,
        max_iter: int = 1000,
        tol: float = 1e-3,
        eta_init: float = 1.0,
        bootstrap: int = 1000,
        n_iter: int = 1000,
        burn_in: int = 500,
    ) -> float:
        """Select learning rate by Generalized Posterior Calibration (proposed by Syring and Martin, 2019)

        Args:
            point_estimate (NDArray): point estimates, such as MLE, MPLE
            alpha (float, optional): (1-alpha) credible interval. Defaults to 0.05.
            max_iter (int, optional): iteration of optimizing learning rate. Defaults to 1000.
            tol (float, optional): sequential renewal criteria (stop iteration when met). Defaults to 1e-3.
            eta_init (float, optional): initial value of learning rate. Defaults to 1.0.
            bootstrap (int, optional): the number of bootstrap. Defaults to 1000.
            n_iter (int, optional): iteration of sampling in bootstrap. Defaults to 1000.
            burn_in (int, optional): iteration from sampling in bootstrap. Defaults to 500.

        Returns:
            float: learning rate
        """
        eta: float = eta_init
        for t in range(1, max_iter + 1):
            cover_true: float = 0
            # prepare arguments list for parallel processing
            args_list = [(eta, alpha, n_iter, burn_in)] * bootstrap

            # execute parallel processing
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(self.compute_region, args) for args in args_list]
                # for checking progress bar
                for future in tqdm(as_completed(futures), total=bootstrap, desc=f"Boot[{t}]"):
                    lower, upper = future.result()
                    mask: NDArray = np.logical_and(lower <= point_estimate, point_estimate <= upper)
                    cover_true += float(mask.all())

            # compute coverage probability
            cover: float = cover_true/bootstrap
            # update learning rate
            eta += (1/t) * (cover - (1-alpha))
            eta = max(eta, 1e-4)
            if abs(cover - (1-alpha)) < tol:
                break
        return float(eta)
