import numpy as np
import pandas as pd  # type: ignore

from lifelines import CoxPHFitter  # type: ignore

from cox_pg_sampler import CoxPGSampler
from generate_synthetic_data import SyntheticDataGenerater4CoxReg

if __name__ == '__main__':
    # Generate Data
    beta_true: np.ndarray = np.array([5.0, 3.5])
    n: int = 100
    data_generater = SyntheticDataGenerater4CoxReg(n=n, beta_true=beta_true)
    covariates, time, event = data_generater.simulate_cox_data()

    # Cox-PG-Sampler
    cpg = CoxPGSampler(covariates=covariates)
    cpg_samples: np.ndarray = cpg.cox_pg_sample(
        time, event, n_iter=500, burn_in=400,
    )
    print(f'Estimated coefficients by Cox-PG: {cpg_samples.mean(axis=0)}')

    # simple Cox Regression
    df = pd.DataFrame(covariates, columns=[f'X{i+1}' for i in range(len(beta_true))])
    df['time'] = time
    df['event'] = event
    cph = CoxPHFitter()
    cph.fit(df, duration_col='time', event_col='event')
    print("Estimated coefficients by normal cox:", cph.params_)
