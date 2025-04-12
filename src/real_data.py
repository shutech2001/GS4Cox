import argparse
from typing import Tuple
import time as t
import numpy as np
import pandas as pd  # type: ignore

from lifelines import CoxPHFitter  # type: ignore

from cox_sampler import GBCoxPGSampler, CoxMHSampler
from data import import_r_data
from eval import compute_ess, compute_esr

# global setting for output
np.set_printoptions(precision=2, suppress=True)


def preprocess4cox(
    df: pd.DataFrame,
    id_col_name: str,
    time_col_name: str,
    event_col_name: str,
    event_ind: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariate_columns = [
        col for col in df.columns if col not in [f'{id_col_name}', f'{time_col_name}', f'{event_col_name}']
    ]
    covariates: np.ndarray = df[covariate_columns].to_numpy()
    time: np.ndarray = df[f'{time_col_name}'].to_numpy()
    event: np.ndarray = (df[f'{event_col_name}'] == event_ind).astype(int).to_numpy()
    return covariates, time, event


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cox Regression Simulation with Cox-PG Sampler and naive Cox Regression'
    )
    parser.add_argument(
        '--iteration', '--I',
        type=int,
        default=1000,
        help='the number of total iterations (default: 1000).'
    )
    parser.add_argument(
        '--burn-in', '--B',
        type=int,
        default=500,
        help='the number of burn-in (default: 500).'
    )
    parser.add_argument(
        '--proposal-scale', '--P',
        type=float,
        default=10,
        help='covariance scale for proposal distribution (default: 10).'
    )
    args = parser.parse_args()

    _df: pd.DataFrame = import_r_data(dataset_name='lung', package_name='survival')
    df = _df.dropna()
    print('loading completed!')

    covariates, time, event = preprocess4cox(df, 'inst', 'time', 'status', 2)
    # Estimate by Cox-PG Gibbs sampler
    cpg = GBCoxPGSampler(covariates=covariates)
    start = t.time()
    cpg_samples: np.ndarray = cpg.gb_cox_pg_sample(time, event, n_iter=args.iteration)
    end = t.time()
    print(f'{end - start}')
    cpg_burn_in: np.ndarray = cpg_samples[args.burn_in:]
    print(f'\nEstimated coefficients by Cox-PG: {cpg_burn_in.mean(axis=0)}')
    print(f'compute ess: {compute_ess(cpg_samples).mean(axis=0):.2f}')
    print(f'compute esr: {compute_esr(cpg_samples, runtime=end-start).mean(axis=0):.2f}')
    # print(f'compute dist: {compute_dist(cpg_samples)}')

    # Estimate by Cox MH sampler
    cmh = CoxMHSampler(covariates=covariates)
    start = t.time()
    cmh_samples, acceptance_rate = cmh.cox_mh_sample(
        time, event, n_iter=args.iteration, proposal_scale=args.proposal_scale
    )
    end = t.time()
    print(f'{end - start}')
    cmh_burn_in: np.ndarray = cmh_samples[args.burn_in:]
    print(f'\nEstimated coefficients by Cox-MH: {cmh_burn_in.mean(axis=0)}')
    print(f'acceptance rate: {acceptance_rate:.2f}')
    print(f'compute ess: {compute_ess(cmh_burn_in).mean(axis=0):.2f}')
    print(f'compute esr: {compute_esr(cmh_burn_in, runtime=end-start).mean(axis=0):.2f}')
    # print(f'compute dist: {compute_dist(cmh_samples)}')

    start = t.time()
    cmh_h_samples, acceptance_rate = cmh.cox_mh_with_hessian_sample(time, event, n_iter=args.iteration)
    end = t.time()
    print(f'{end - start}')
    cmh_h_burn_in: np.ndarray = cmh_h_samples[args.burn_in:]
    print(f'\nEstimated coefficients by Cox-MH: {cmh_h_burn_in.mean(axis=0)}')
    print(f'acceptance rate: {acceptance_rate:.2f}')
    print(f'compute ess: {compute_ess(cmh_h_burn_in).mean(axis=0):.2f}')
    print(f'compute esr: {compute_esr(cmh_h_burn_in, runtime=end-start).mean(axis=0):.2f}')
    # print(f'compute dist: {compute_dist(cmh_samples)}')

    # Estimate by naive Cox Regression
    df = pd.DataFrame(covariates, columns=[f'X{i+1}' for i in range(covariates.shape[1])])
    df['time'] = time
    df['event'] = event
    cph = CoxPHFitter()
    cph.fit(df, duration_col='time', event_col='event')
    print("\nEstimated coefficients by normal cox:", np.array(cph.params_))
