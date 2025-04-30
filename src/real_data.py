import argparse
from typing import Tuple, List
import time as t
import numpy as np
from numpy.typing import NDArray
import pandas as pd  # type: ignore

from lifelines import CoxPHFitter  # type: ignore

from cox_sampler import GS4Cox, CoxMHSampler
from data import import_r_data
from evaluation_metrics import compute_ess, compute_esr
from select_learning_rate import SelectLearningRate

# global setting for output
np.set_printoptions(precision=2, suppress=True)


def preprocess4cox(
    df: pd.DataFrame,
    time_col_name: str,
    event_col_name: str,
    event_ind: int,
    covariates_col_names: List[str],
) -> Tuple[NDArray, NDArray, NDArray]:
    """Preprocessing for MCMC

    Args:
        df (pd.DataFrame): dataframe
        time_col_name (str): column name of representing observed time
        event_col_name (str): column name of representing event indicator
        event_ind (int): indicator of event
        covariatese_col_name (list)

    Returns:
        Tuple[NDArray, NDArray, NDArray]:
            covariates: covariates data
            time: observed time
            event: identifier of event (1: event occurred, 0: not occurred)
    """
    covariate_columns = [
        col for col in df.columns if col in covariates_col_names
    ]
    covariates: NDArray = df[covariate_columns].to_numpy()
    time: NDArray = df[f'{time_col_name}'].to_numpy()
    event: NDArray = (df[f'{event_col_name}'] == event_ind).astype(int).to_numpy()
    return covariates, time, event


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cox Regression Simulation with Cox-PG Sampler and naive Cox Regression'
    )
    parser.add_argument(
        '--package', '--P',
        type=str,
        default='survival',
        help='name of the R package that contains the data to be imported (default: "survival").'
    )
    parser.add_argument(
        '--dataset', '--D',
        type=str,
        default='lung',
        help='name of the R dataset (default: "lung").'
    )
    parser.add_argument(
        '--time-col-name', '--TC',
        type=str,
        default='time',
        help='column name of representing `time` (default: "time").'
    )
    parser.add_argument(
        '--event-col-name', '--EC',
        type=str,
        default='status',
        help='column name of representing `event status` (default: "status").'
    )
    parser.add_argument(
        '--event-indicator', '--EI',
        type=int,
        default=2,
        help='indicator representing the event in the `--event-col-name` (default: 2).'
    )
    parser.add_argument(
        '--covariates-column-names', '--CN',
        nargs='+',
        default=['age', 'sex', 'ph.ecog', 'ph.karno', 'pat.karno', 'meal.cal', 'wt.loss'],
        help='column names of regression covariates (default: "age sex ph.ecog ph.karno pat.karno meal.cal wt.loss")'
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
        '--learning-rate', '--L',
        type=float,
        default=1.0,
        help="learning rate for general Bayesian framework (default: 1.0). If set to 0, execute learning rate selection by GPC."  # noqa: E501
    )
    args = parser.parse_args()

    # import data
    _df: pd.DataFrame = import_r_data(dataset_name=args.dataset, package_name=args.package)
    df = _df.dropna()
    print(f'[COMPLETED] loading {args.dataset} from {args.package}')

    # maximum partial likelihood estimates
    covariates, time, event = preprocess4cox(
        df,
        args.time_col_name,
        args.event_col_name,
        args.event_indicator,
        args.covariates_column_names,
    )
    df4cox = pd.DataFrame(covariates, columns=[f'X{i+1}' for i in range(covariates.shape[1])])
    df4cox['time'] = time
    df4cox['event'] = event
    cph = CoxPHFitter()
    cph.fit(df4cox, duration_col='time', event_col='event')
    print("\nEstimated coefficients by normal cox:", np.array(cph.params_))

    lr: float = args.learning_rate
    if lr == 0:
        # select learning rate by GPC
        slr_GS = SelectLearningRate(GS4Cox, 'composite_gb_cox_pg_sample', covariates, time, event)
        lr_GS = slr_GS.select_eta_gpc(point_estimate=np.array(cph.params_))
        print(f'gb pairwise cox learning rate: {lr_GS}')
    else:
        lr_GS = lr
    # Estimate by GS4Cox
    gs4c = GS4Cox(covariates=covariates)
    start = t.time()
    gs4c_samples: np.ndarray = gs4c.composite_gb_cox_pg_sample(time, event, n_iter=args.iteration, lr=lr_GS)
    end = t.time()
    cpg_burn_in: np.ndarray = gs4c_samples[args.burn_in:]
    print(f'\nEstimated coefficients by Cox-PG: {cpg_burn_in.mean(axis=0)}')
    print(f'executing time: {end - start:.2f}')
    print(f'compute ess: {compute_ess(gs4c_samples).mean(axis=0):.2f}')
    print(f'compute esr: {compute_esr(gs4c_samples, runtime=end-start).mean(axis=0):.2f}')

    if lr == 0:
        # select learning rate by GPC
        slr_MH = SelectLearningRate(CoxMHSampler, 'cox_mh_with_hessian_sample', covariates, time, event)
        lr_MH = slr_MH.select_eta_gpc(point_estimate=np.array(cph.params_))
        print(f'cox mh with hessian learning rate: {lr_MH}')
    else:
        lr_MH = lr
    # Estimate by MH-Hessian
    cmh = CoxMHSampler(covariates=covariates)
    start = t.time()
    cmh_h_samples = cmh.cox_mh_with_hessian_sample(time, event, n_iter=args.iteration, lr=lr_MH)
    end = t.time()
    cmh_h_burn_in: np.ndarray = cmh_h_samples[args.burn_in:]
    print(f'\nEstimated coefficients by Cox-MH: {cmh_h_burn_in.mean(axis=0)}')
    print(f'executing time: {end - start:.2f}')
    print(f'compute ess: {compute_ess(cmh_h_burn_in).mean(axis=0):.2f}')
    print(f'compute esr: {compute_esr(cmh_h_burn_in, runtime=end-start).mean(axis=0):.2f}')
