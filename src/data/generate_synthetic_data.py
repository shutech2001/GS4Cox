import numpy as np
from typing import Tuple


class SyntheticDataGenerater4CoxReg:
    def __init__(self, n: int, beta_true: np.ndarray, seed: int = 42) -> None:
        """
        Args:
            n (int): size of data
            beta_true (np.ndarray): true value of parameters
            seed (int, optional): seed value. Defaults to 42.
        """
        self.n: int = n
        self.beta_true: np.ndarray = beta_true
        self.seed: int = seed

    def simulate_cox_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic data for Cox regression

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                covariates: covariates data
                time: observed time
                event: identifier of event (1: event occurred, 0: not occurred)
        """
        np.random.seed(self.seed)
        num_param: int = len(self.beta_true)
        covariates: np.ndarray = np.random.randn(self.n, num_param)
        linpred: np.ndarray = covariates.dot(self.beta_true)
        # generate event time and censoring time from exponential distribution
        event_time: np.ndarray = np.random.exponential(scale=1/np.exp(linpred))
        censor_time: np.ndarray = np.random.exponential(scale=1.0, size=self.n)
        # checking for observed time and occur event
        time: np.ndarray = np.minimum(event_time, censor_time)
        event: np.ndarray = (event_time <= censor_time).astype(int)
        return covariates, time, event

    def simulate_cox_data_ties(
        self, rounding: float = 0.001
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic data for Cox regression including tie data

        Args:
            rounding (float, optional): Units of rounding for observed time. Defaults to 1.0.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                covariates: covariates data
                time: observed time
                event: identifier of event (1: event occurred, 0: not occurred)
        """
        np.random.seed(self.seed)
        num_param: int = len(self.beta_true)
        covariates: np.ndarray = np.random.randn(self.n, num_param)
        linpred: np.ndarray = covariates.dot(self.beta_true)
        # generate event time and censoring time from exponential distribution
        event_time: np.ndarray = np.random.exponential(scale=1/np.exp(linpred))
        censor_time: np.ndarray = np.random.exponential(scale=1.0, size=self.n)
        # checking for observed time and occur event
        _time: np.ndarray = np.minimum(event_time, censor_time)
        # rounding time for occurring multi events in the same time
        time: np.ndarray = np.round(_time / rounding) * rounding
        event: np.ndarray = (event_time <= censor_time).astype(int)
        return covariates, time, event

    def simulate_cox_data_with_random_effects(
        self, random_effect_var: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic data for Cox regression with random effects

        Args:
            random_effect_var (float, optional): variance of random effect. Defaults to 1.0.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                covariates: covariates data
                time: observed time
                event: identifier of event (1: event occurred, 0: not occurred)
        """
        np.random.seed(self.seed)
        num_param: int = len(self.beta_true)
        covariates: np.ndarray = np.random.randn(self.n, num_param)
        random_effects: np.ndarray = np.random.normal(loc=0, scale=random_effect_var, size=self.n)
        linpred: np.ndarray = covariates.dot(self.beta_true) + random_effects
        # generate event time and censoring time from exponential distribution
        event_time: np.ndarray = np.random.exponential(scale=1/np.exp(linpred))
        censor_time: np.ndarray = np.random.exponential(scale=1.0, size=self.n)
        # checking for observed time and occur event
        time: np.ndarray = np.minimum(event_time, censor_time)
        event: np.ndarray = (event_time <= censor_time).astype(int)
        return covariates, time, event
