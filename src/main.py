import argparse
import time as t
import numpy as np
import pandas as pd  # type: ignore

from lifelines import CoxPHFitter  # type: ignore

from cox_sampler import GBCoxPGSampler, CoxMHSampler
from data import SyntheticDataGenerater4CoxReg
from eval import compute_esr, compute_ess

# global setting for output
np.set_printoptions(precision=2, suppress=True)


def run_simulation(
    n: int,
    beta_true: np.ndarray,
    n_iter: int,
    burn_in: int,
    proposal_scale: float,
    use_ties: bool,
    rounding: float,
) -> None:
    data_generater = SyntheticDataGenerater4CoxReg(n=n, beta_true=beta_true)

    if use_ties:
        covariates, time, event = data_generater.simulate_cox_data_ties(rounding=rounding)
    else:
        covariates, time, event = data_generater.simulate_cox_data()

    # Estimate by Cox-PG Gibbs sampler
    cpg = GBCoxPGSampler(covariates=covariates)
    start = t.time()
    cpg_samples: np.ndarray = cpg.gb_cox_pg_sample(time, event, n_iter=n_iter)
    end = t.time()
    print(f'{end - start}')
    cpg_burn_in: np.ndarray = cpg_samples[burn_in:]
    print(f'Estimated coefficients by Cox-PG: {cpg_burn_in.mean(axis=0)}')
    print(f'compute ess: {compute_ess(cpg_samples).mean(axis=0):.2f}')
    print(f'compute esr: {compute_esr(cpg_samples, runtime=end-start).mean(axis=0):.2f}')
    # print(f'compute dist: {compute_dist(cpg_samples)}')

    # Estimate by Cox MH sampler
    cmh = CoxMHSampler(covariates=covariates)
    start = t.time()
    cmh_samples, acceptance_rate = cmh.cox_mh_sample(time, event, n_iter=n_iter, proposal_scale=proposal_scale)
    end = t.time()
    print(f'{end - start}')
    cmh_burn_in: np.ndarray = cmh_samples[burn_in:]
    print(f'Estimated coefficients by Cox-MH: {cmh_burn_in.mean(axis=0)}')
    print(f'acceptance rate: {acceptance_rate:.2f}')
    print(f'compute ess: {compute_ess(cmh_burn_in).mean(axis=0):.2f}')
    print(f'compute esr: {compute_esr(cmh_burn_in, runtime=end-start).mean(axis=0):.2f}')
    # print(f'compute dist: {compute_dist(cmh_samples)}')

    start = t.time()
    cmh_h_samples, acceptance_rate = cmh.cox_mh_with_hessian_sample(time, event, n_iter=n_iter)
    end = t.time()
    print(f'{end - start}')
    cmh_h_burn_in: np.ndarray = cmh_h_samples[burn_in:]
    print(f'Estimated coefficients by Cox-MH optimal: {cmh_h_burn_in.mean(axis=0)}')
    print(f'acceptance rate: {acceptance_rate:.2f}')
    print(f'compute ess: {compute_ess(cmh_h_burn_in).mean(axis=0):.2f}')
    print(f'compute esr: {compute_esr(cmh_h_burn_in, runtime=end-start).mean(axis=0):.2f}')
    # print(f'compute dist: {compute_dist(cmh_samples)}')

    # Estimate by naive Cox Regression
    df = pd.DataFrame(covariates, columns=[f'X{i+1}' for i in range(len(beta_true))])
    df['time'] = time
    df['event'] = event
    cph = CoxPHFitter()
    cph.fit(df, duration_col='time', event_col='event')
    print("Estimated coefficients by normal cox:", np.array(cph.params_))


def parse_beta(beta_str: str) -> np.ndarray:
    """Generate an np.ndarray from a comma-separated string

    Args:
        beta_str (str): comma-separated parameter (e.g., 5.0,3.5)

    Returns:
        np.ndarray: true coefficient vector
    """
    try:
        beta_list: list[float] = [float(b.strip()) for b in beta_str.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError('beta_true must be a comma-separated list of numbers. (e.g., 5.0,3.5)') from e
    return np.array(beta_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cox Regression Simulation with Cox-PG Sampler and naive Cox Regression'
    )
    parser.add_argument(
        '--data-size', '--N',
        type=int,
        default=300,
        help='sample size (default: 300).'
    )
    parser.add_argument(
        '--beta-true', '--T',
        type=parse_beta,
        default='3.0,1.5',
        help="true value of coefficents (default: '3.0,1.5')."
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
        '--use-ties', '--U',
        type=bool,
        default=False,
        help='set to `True` when performing simulation based on the same event occurrence data (default: False).'
    )
    parser.add_argument(
        '--rounding', '--R',
        type=float,
        default=0.001,
        help='rounding unit for generating tie data (default: 0.001).'
    )
    parser.add_argument(
        '--proposal-scale', '--P',
        type=float,
        default=10,
        help='covariance scale for proposal distribution (default: 10).'
    )
    args = parser.parse_args()

    run_simulation(
        n=args.data_size,
        beta_true=args.beta_true,
        n_iter=args.iteration,
        burn_in=args.burn_in,
        proposal_scale=args.proposal_scale,
        use_ties=args.use_ties,
        rounding=args.rounding,
    )
