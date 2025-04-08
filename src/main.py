import argparse
import numpy as np
import pandas as pd  # type: ignore

from lifelines import CoxPHFitter  # type: ignore

from cox_pg_sampler import CoxPGSampler
from generate_synthetic_data import SyntheticDataGenerater4CoxReg


def run_simulation(
    n: int,
    beta_true: np.ndarray,
    n_iter: int,
    burn_in: int,
    use_ties: bool,
    rounding: float,
) -> None:
    data_generater = SyntheticDataGenerater4CoxReg(n=n, beta_true=beta_true)

    if use_ties:
        covariates, time, event = data_generater.simulate_cox_data_ties(rounding=rounding)
    else:
        covariates, time, event = data_generater.simulate_cox_data()

    # Estimate by Cox-PG Gibbs sampler
    cpg = CoxPGSampler(covariates=covariates)
    cpg_samples: np.ndarray = cpg.cox_pg_sample(time, event, n_iter=n_iter, burn_in=burn_in)
    print(f'Estimated coefficients by Cox-PG: {cpg_samples.mean(axis=0)}')

    # Estimate by naive Cox Regression
    df = pd.DataFrame(covariates, columns=[f'X{i+1}' for i in range(len(beta_true))])
    df['time'] = time
    df['event'] = event
    cph = CoxPHFitter()
    cph.fit(df, duration_col='time', event_col='event')
    print("Estimated coefficients by normal cox:", cph.params_)


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
        default=100,
        help='sample size (default: 100).'
    )
    parser.add_argument(
        '--beta-true', '--T',
        type=parse_beta,
        default='1.0,0.5',
        help="true value of coefficents (default: '1.0,0.5')."
    )
    parser.add_argument(
        '--iteration', '--I',
        type=int,
        default=500,
        help='the number of total iterations (default: 500).'
    )
    parser.add_argument(
        '--burn-in', '--B',
        type=int,
        default=400,
        help='the number of burn-in (default: 400).'
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
    args = parser.parse_args()

    run_simulation(
        n=args.data_size,
        beta_true=args.beta_true,
        n_iter=args.iteration,
        burn_in=args.burn_in,
        use_ties=args.use_ties,
        rounding=args.rounding,
    )
