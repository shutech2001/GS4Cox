from __future__ import annotations

import argparse
from pathlib import Path
import time as t
import warnings

from lifelines import CoxPHFitter  # type: ignore
import numpy as np
from numpy.typing import NDArray
import pandas as pd  # type: ignore

from cox_sampler import GS4Cox, CoxMHSampler
from data import SyntheticDataGenerator4CoxReg
from utils.evaluation_metrics import compute_esr, compute_ess
from utils.pl_score_hessian import cox_score_and_hess
from utils.plot_figure import PlotSyntheticResult

warnings.filterwarnings('ignore')
# global setting for output
np.set_printoptions(precision=2, suppress=True)


def run_simulation(
    n: int,
    beta_true: NDArray,
    learning_rate: float,
    n_iter: int,
    burn_in: int,
    use_ties: bool,
    rounding: float,
    ablation_correction: bool = False,
) -> None:
    data_generator = SyntheticDataGenerator4CoxReg(n=n, beta_true=beta_true)

    if use_ties:
        covariates, time, event = data_generator.simulate_cox_data_ties(rounding=rounding)
    else:
        covariates, time, event = data_generator.simulate_cox_data()

    # Estimate by GS4Cox
    if ablation_correction:
        gs4c = GS4Cox(covariates=covariates)
        start = t.time()
        gs4c_samples: NDArray = gs4c.gs4cox_without_finite_correction(time, event, n_iter=n_iter, lr=learning_rate)
        end = t.time()
        gs4c_burn_in: NDArray = gs4c_samples[burn_in:]
        print(f'Estimated coefficients by GS4Cox (without finite correction): {gs4c_burn_in.mean(axis=0)}')
        print(f'executing time: {end - start:.2f}')
        print(f'Compute ess: {compute_ess(gs4c_samples).mean(axis=0):.2f}')
        print(f'Compute esr: {compute_esr(gs4c_samples, runtime=end-start).mean(axis=0):.2f}')

        # Estimate by GS4Cox without correction
        score, hess = cox_score_and_hess(gs4c_burn_in.mean(axis=0), covariates, time, event)
        gs4c_samples_corrected = gs4c_samples + np.linalg.solve(hess, score)
        gs4c_burn_in_corrected: NDArray = gs4c_samples_corrected[burn_in:]
        print(f'Estimated coefficients by GS4Cox (with finite correction): {gs4c_burn_in_corrected.mean(axis=0)}')

    else:
        gs4c = GS4Cox(covariates=covariates)
        start = t.time()
        gs4c_samples = gs4c.gs4cox_with_finite_correction(time, event, n_iter=n_iter, burn_in=burn_in, lr=learning_rate)
        end = t.time()
        gs4c_burn_in = gs4c_samples[burn_in:]
        print(f'Estimated coefficients by GS4Cox: {gs4c_burn_in.mean(axis=0)}')
        print(f'executing time: {end - start:.2f}')
        print(f'Compute ess: {compute_ess(gs4c_samples).mean(axis=0):.2f}')
        print(f'Compute esr: {compute_esr(gs4c_samples, runtime=end-start).mean(axis=0):.2f}')

    # Estimate by Cox MH sampler
    cmh = CoxMHSampler(covariates=covariates)
    start = t.time()
    cmh_h_samples = cmh.cox_mh_with_hessian_sample(time, event, n_iter=n_iter, lr=learning_rate)
    end = t.time()
    cmh_h_burn_in: NDArray = cmh_h_samples[burn_in:]
    print(f'Estimated coefficients by MH optimal: {cmh_h_burn_in.mean(axis=0)}')
    print(f'executing time: {end - start:.2f}')
    print(f'Computed ess: {compute_ess(cmh_h_burn_in).mean(axis=0):.2f}')
    print(f'Computed esr: {compute_esr(cmh_h_burn_in, runtime=end-start).mean(axis=0):.2f}')

    # Estimate by naive Cox Regression
    df = pd.DataFrame(covariates, columns=[f'X{i+1}' for i in range(len(beta_true))])
    df['time'] = time
    df['event'] = event
    cph = CoxPHFitter()
    cph.fit(df, duration_col='time', event_col='event')
    cph_mple: NDArray = np.array(cph.params_)
    print("Maximum partial likelihood estimates:", cph_mple)

    # plot some figures
    if ablation_correction:
        psr = PlotSyntheticResult(
            n_iter=n_iter,
            mh_samples=cmh_h_samples,
            gs4c_samples=gs4c_samples,
            gs4c_samples_corrected=gs4c_samples_corrected,
            cph_mpl_estimates=cph_mple,
            savefig_root=Path('../fig'),
        )
        psr.pict_trace_plot(beta_true=beta_true)
        psr.pict_post_dist(beta_true=beta_true, burn_in=burn_in)
        psr.pict_correlogram(beta_true=beta_true)


def parse_beta(beta_str: str) -> NDArray:
    """Generate an NDArray from a comma-separated string

    Args:
        beta_str (str): comma-separated parameter (e.g., 5.0,3.5)

    Returns:
        NDArray: true coefficient vector
    """
    try:
        beta_list: list[float] = [float(b.strip()) for b in beta_str.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError('beta_true must be a comma-separated list of numbers. (e.g., 5.0,3.5)') from e
    return np.array(beta_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cox Regression Simulation with GS4Cox, optimal MH and Standard Cox Regression'
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
        default='1.0,0.5,-1.5,3.0',
        help="true value of coefficients (default: 1.0,0.5,-1.5,3.0)."
    )
    parser.add_argument(
        '--learning-rate', '--L',
        type=float,
        default='1.0',
        help="learning rate for general Bayesian framework (default: 1.0)."
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
        '--ablation-correction', '--A',
        type=bool,
        default=False,
        help='set to `True` when comparing results with and without finite-sample corrections (default: False).'
    )
    args = parser.parse_args()

    run_simulation(
        n=args.data_size,
        beta_true=args.beta_true,
        learning_rate=args.learning_rate,
        n_iter=args.iteration,
        burn_in=args.burn_in,
        use_ties=args.use_ties,
        rounding=args.rounding,
        ablation_correction=args.ablation_correction
    )
