from __future__ import annotations

import numpy as np
from typing import Tuple

from numpy.typing import NDArray


class SyntheticDataGenerator4CoxReg:
    def __init__(self, n: int, beta_true: NDArray, seed: int = 42) -> None:
        """
        Args:
            n (int): size of data
            beta_true (NDArray): true value of parameters
            seed (int, optional): seed value. Defaults to 42.
        """
        self.data_num: int = n
        self.beta_true: NDArray = beta_true
        self.seed: int = seed

    def simulate_cox_data(self) -> Tuple[NDArray, NDArray, NDArray]:
        """Generate synthetic data for Cox regression

        Returns:
            Tuple[NDArray, NDArray, NDArray]:
                covariates: covariates data
                time: observed time
                event: identifier of event (1: event occurred, 0: not occurred)
        """
        np.random.seed(self.seed)
        num_param: int = len(self.beta_true)
        covariates: NDArray = np.random.randn(self.data_num, num_param)
        linpred: NDArray = covariates.dot(self.beta_true)
        # generate event time and censoring time from exponential distribution
        event_time: NDArray = np.random.exponential(scale=1/np.exp(linpred))
        censor_time: NDArray = np.random.exponential(scale=1.0, size=self.data_num)
        # checking for observed time and occur event
        time: NDArray = np.minimum(event_time, censor_time)
        event: NDArray = (event_time <= censor_time).astype(int)
        return covariates, time, event

    def simulate_cox_data_ties(
        self, rounding: float = 0.001
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Generate synthetic data for Cox regression including tie data

        Args:
            rounding (float, optional): Units of rounding for observed time. Defaults to 1.0.

        Returns:
            tuple[NDArray, NDArray, NDArray]:
                covariates: covariates data
                time: observed time
                event: identifier of event (1: event occurred, 0: not occurred)
        """
        np.random.seed(self.seed)
        num_param: int = len(self.beta_true)
        covariates: NDArray = np.random.randn(self.data_num, num_param)
        linpred: NDArray = covariates.dot(self.beta_true)
        # generate event time and censoring time from exponential distribution
        event_time: NDArray = np.random.exponential(scale=1/np.exp(linpred))
        censor_time: NDArray = np.random.exponential(scale=1.0, size=self.data_num)
        # checking for observed time and occur event
        _time: NDArray = np.minimum(event_time, censor_time)
        # rounding time for occurring multi events in the same time
        time: NDArray = np.round(_time / rounding) * rounding
        event: NDArray = (event_time <= censor_time).astype(int)
        return covariates, time, event
