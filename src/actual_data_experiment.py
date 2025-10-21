from __future__ import annotations

import argparse
import time as t
from pathlib import Path
from typing import Tuple
import warnings

import numpy as np
from numpy.typing import NDArray
import pandas as pd  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from lifelines import CoxPHFitter  # type: ignore

from cox_sampler import CoxSampler, GS4Cox, CoxMHSampler, CoxHMCSampler, CoxNUTSampler, CoxMALASampler, CoxPGSampler
from data import import_r_data
from utils.evaluation_metrics import compute_ess, compute_esr, compute_mcse
from utils.plot import plot_trace, plot_correlogram, plot_forestplot

warnings.filterwarnings("ignore")
# global setting for output
np.set_printoptions(precision=2, suppress=True)


def preprocess4cox(
    df: pd.DataFrame,
    time_col_name: str,
    event_col_name: str,
    event_ind: int,
    covariates_col_names: list[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocessing for MCMC

    Args:
        df (pd.DataFrame): dataframe
        time_col_name (str): column name of representing observed time
        event_col_name (str): column name of representing event indicator
        event_ind (int): indicator of event

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            covariates: covariates data
            time: observed time
            event: identifier of event (1: event occurred, 0: not occurred)
    """
    covariate_columns = [col for col in df.columns if col in covariates_col_names]
    covariates: np.ndarray = df[covariate_columns].to_numpy()
    time: np.ndarray = df[f"{time_col_name}"].to_numpy()
    event: np.ndarray = (df[f"{event_col_name}"] == event_ind).astype(int).to_numpy()
    return covariates, time, event


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cox Regression Simulation with Cox-PG Sampler and naive Cox Regression"
    )
    parser.add_argument(
        "--package",
        "--P",
        type=str,
        default="survival",
        help='name of the R package that contains the data to be imported (default: "survival").',
    )
    parser.add_argument("--dataset", "--D", type=str, default="lung", help='name of the R dataset (default: "lung").')
    parser.add_argument(
        "--time-col-name",
        "--TC",
        type=str,
        default="time",
        help='column name of representing `time` (default: "time").',
    )
    parser.add_argument(
        "--event-col-name",
        "--EC",
        type=str,
        default="status",
        help='column name of representing `event status` (default: "status").',
    )
    parser.add_argument(
        "--event-indicator",
        "--EI",
        type=int,
        default=2,
        help="indicator representing the event in the `--event-col-name` (default: 2).",
    )
    parser.add_argument(
        "--covariates-column-names",
        "--CN",
        nargs="+",
        default=["age", "sex", "ph.ecog", "ph.karno", "pat.karno", "meal.cal", "wt.loss"],
        help='column names of regression covariates (default: "age sex ph.ecog ph.karno pat.karno meal.cal wt.loss")',
    )
    parser.add_argument(
        "--iteration", "--I", type=int, default=1000, help="the number of total iterations (default: 1000)."
    )
    parser.add_argument("--burn-in", "--B", type=int, default=500, help="the number of burn-in (default: 500).")
    parser.add_argument(
        "--learning-rate",
        "--L",
        type=float,
        default=1.0,
        help="learning rate for general Bayesian framework (default: 1.0). If set to 0, execute learning rate selection by GPC.",  # noqa: E501
    )
    parser.add_argument(
        "--calc-intervals",
        "--CI",
        type=bool,
        default=True,
        help="set to `True` when calculating 95% confidence/credible intervals of estimated coefficients (default: True).",  # noqa: E501
    )
    parser.add_argument(
        "--plot-results",
        "--P",
        type=bool,
        default=False,
        help="set to `True` when plotting results (default: False).",
    )
    parser.add_argument(
        "--savefig-root",
        "--S",
        type=str,
        default="figures",
        help="root directory for saving figures (default: 'figures').",
    )
    args = parser.parse_args()

    # calculate lower and upper bounds of confidence interval
    alpha: float = 0.05
    lower_bound: float = alpha / 2
    upper_bound: float = 1 - alpha / 2

    # settings
    n_iter: int = args.iteration
    burn_in: int = args.burn_in
    learning_rate: float = args.learning_rate
    calc_intervals: bool = args.calc_intervals
    plot_results: bool = args.plot_results
    savefig_root: Path = Path(args.savefig_root)
    # import data
    _df: pd.DataFrame = import_r_data(dataset_name=args.dataset, package_name=args.package)
    df = _df.dropna().copy()
    if args.dataset == "lung":
        df["sex"] = df["sex"] - 1  # convert to 0-1 coding
    print(f"[COMPLETED] loading {args.dataset} from {args.package}")

    # standardize covariates
    scaler = StandardScaler()
    df[args.covariates_column_names] = scaler.fit_transform(df[args.covariates_column_names])

    # maximum partial likelihood estimates
    covariates, time, event = preprocess4cox(
        df,
        args.time_col_name,
        args.event_col_name,
        args.event_indicator,
        args.covariates_column_names,
    )
    df4cox = pd.DataFrame(covariates, columns=[f"X{i+1}" for i in range(covariates.shape[1])])
    df4cox["time"] = time
    df4cox["event"] = event
    cph = CoxPHFitter()
    cph.fit(df4cox, duration_col="time", event_col="event")
    cph_mple: NDArray = np.array(cph.params_)
    print("Maximum partial likelihood estimates:", cph_mple)
    if args.calc_intervals:
        cph_mple_lower: NDArray = np.array(cph.confidence_intervals_["95% lower-bound"])
        cph_mple_upper: NDArray = np.array(cph.confidence_intervals_["95% upper-bound"])
        for i, (low, up) in enumerate(zip(cph_mple_lower, cph_mple_upper)):
            print(f"95% confidence interval of estimated coefficient {i}: {low:.2f} - {up:.2f}")

    # set global seeds
    CoxSampler.set_global_seeds()
    # GS4Cox
    gs4 = GS4Cox(covariates=covariates)
    start = t.time()
    gs4_samples: NDArray = gs4.sample(time, event, n_iter=n_iter, lr=learning_rate)
    end = t.time()
    gs4_burn_in: NDArray = gs4_samples[burn_in:]
    print(f"Estimated coefficients by GS4Cox: {gs4_burn_in.mean(axis=0)}")
    print(f"{end - start}")
    print(f"compute ess : {compute_ess(gs4_burn_in).mean(axis=0):.2f}")
    print(f"compute esr : {compute_esr(gs4_burn_in, runtime=end-start).mean(axis=0):.2f}")
    print(f"compute mcse: {compute_mcse(gs4_burn_in).mean(axis=0):.4f}")
    if calc_intervals:
        gs4_lower: NDArray = np.quantile(gs4_burn_in, lower_bound, axis=0)
        gs4_upper: NDArray = np.quantile(gs4_burn_in, upper_bound, axis=0)
        for i, (low, up) in enumerate(zip(gs4_lower, gs4_upper)):
            print(f"95% credible interval of estimated coefficient {i}: {low:.2f} - {up:.2f}")

    # Metropolis-Hastings algorithm
    mh = CoxMHSampler(covariates=covariates)
    start = t.time()
    mh_samples = mh.sample(time, event, n_iter=n_iter, lr=learning_rate)
    end = t.time()
    mh_burn_in: NDArray = mh_samples[burn_in:]
    print(f"\nEstimated coefficients by Cox-MH: {mh_burn_in.mean(axis=0)}")
    print(f"{end - start}")
    print(f"compute ess : {compute_ess(mh_burn_in).mean(axis=0):.2f}")
    print(f"compute esr : {compute_esr(mh_burn_in, runtime=end-start).mean(axis=0):.2f}")
    print(f"compute mcse: {compute_mcse(mh_burn_in).mean(axis=0):.4f}")
    if calc_intervals:
        mh_lower: NDArray = np.quantile(mh_burn_in, lower_bound, axis=0)
        mh_upper: NDArray = np.quantile(mh_burn_in, upper_bound, axis=0)
        for i, (low, up) in enumerate(zip(mh_lower, mh_upper)):
            print(f"95% credible interval of estimated coefficient {i}: {low:.2f} - {up:.2f}")

    # Hamiltonian Monte Carlo
    hmc = CoxHMCSampler(covariates=covariates)
    start = t.time()
    hmc_samples = hmc.sample(time, event, n_iter=n_iter, lr=learning_rate)
    end = t.time()
    hmc_burn_in: NDArray = hmc_samples[burn_in:]
    print(f"\nEstimated coefficients by Cox-HMC: {hmc_burn_in.mean(axis=0)}")
    print(f"{end - start}")
    print(f"compute ess : {compute_ess(hmc_burn_in).mean(axis=0):.2f}")
    print(f"compute esr : {compute_esr(hmc_burn_in, runtime=end-start).mean(axis=0):.2f}")
    print(f"compute mcse: {compute_mcse(hmc_burn_in).mean(axis=0):.4f}")
    if calc_intervals:
        hmc_lower: NDArray = np.quantile(hmc_burn_in, lower_bound, axis=0)
        hmc_upper: NDArray = np.quantile(hmc_burn_in, upper_bound, axis=0)
        for i, (low, up) in enumerate(zip(hmc_lower, hmc_upper)):
            print(f"95% credible interval of estimated coefficient {i}: {low:.2f} - {up:.2f}")

    # No-U-Turn Sampler
    nuts = CoxNUTSampler(covariates=covariates)
    start = t.time()
    nuts_samples = nuts.sample(time, event, n_iter=n_iter, lr=learning_rate)
    end = t.time()
    print(f"{end - start}")
    nuts_burn_in: NDArray = nuts_samples[burn_in:]
    print(f"\nEstimated coefficients by Cox-NUTS: {nuts_burn_in.mean(axis=0)}")
    print(f"{end - start}")
    print(f"compute ess : {compute_ess(nuts_burn_in).mean(axis=0):.2f}")
    print(f"compute esr : {compute_esr(nuts_burn_in, runtime=end-start).mean(axis=0):.2f}")
    print(f"compute mcse: {compute_mcse(nuts_burn_in).mean(axis=0):.4f}")
    if calc_intervals:
        nuts_lower: NDArray = np.quantile(nuts_burn_in, lower_bound, axis=0)
        nuts_upper: NDArray = np.quantile(nuts_burn_in, upper_bound, axis=0)
        for i, (low, up) in enumerate(zip(nuts_lower, nuts_upper)):
            print(f"95% credible interval of estimated coefficient {i}: {low:.2f} - {up:.2f}")

    # Metropolis-Adjusted Langevin Algorithm
    mala = CoxMALASampler(covariates=covariates)
    start = t.time()
    mala_samples = mala.sample(time, event, n_iter=n_iter, lr=learning_rate)
    end = t.time()
    mala_burn_in: NDArray = mala_samples[burn_in:]
    print(f"\nEstimated coefficients by Cox-MALA: {mala_burn_in.mean(axis=0)}")
    print(f"{end - start}")
    print(f"compute ess : {compute_ess(mala_burn_in).mean(axis=0):.2f}")
    print(f"compute esr : {compute_esr(mala_burn_in, runtime=end-start).mean(axis=0):.2f}")
    print(f"compute mcse: {compute_mcse(mala_burn_in).mean(axis=0):.4f}")
    if calc_intervals:
        mala_lower: NDArray = np.quantile(mala_burn_in, lower_bound, axis=0)
        mala_upper: NDArray = np.quantile(mala_burn_in, upper_bound, axis=0)
        for i, (low, up) in enumerate(zip(mala_lower, mala_upper)):
            print(f"95% credible interval of estimated coefficient {i}: {low:.2f} - {up:.2f}")

    # Cox-P\'olya-Gamma algorithm proposed by Ren et al. (2025)
    cpg = CoxPGSampler(covariates=covariates)
    start = t.time()
    cpg_samples = cpg.sample(time=time, event=event, n_iter=n_iter, calibration=True)
    end = t.time()
    cpg_burn_in = cpg_samples[burn_in:]
    print(f"\nEstimated coefficients by Cox-PG: {cpg_burn_in.mean(axis=0)}")
    print(f"{end - start}")
    print(f"compute ess : {compute_ess(cpg_burn_in).mean(axis=0):.2f}")
    print(f"compute esr : {compute_esr(cpg_burn_in, runtime=end-start).mean(axis=0):.2f}")
    print(f"compute mcse: {compute_mcse(cpg_burn_in).mean(axis=0):.4f}")
    if calc_intervals:
        cpg_lower: NDArray = np.quantile(cpg_burn_in, lower_bound, axis=0)
        cpg_upper: NDArray = np.quantile(cpg_burn_in, upper_bound, axis=0)
        for i, (low, up) in enumerate(zip(cpg_lower, cpg_upper)):
            print(f"95% credible interval of estimated coefficient {i}: {low:.2f} - {up:.2f}")

    if plot_results:
        plot_trace(
            param_idx=0,
            n_iter=n_iter,
            methods=[
                ("GS4Cox", gs4_samples),
                ("MH", mh_samples),
                ("HMC", hmc_samples),
                ("NUTS", nuts_samples),
                ("MALA", mala_samples),
                ("Cox-PG", cpg_samples),
            ],
            mple_estimates=cph_mple,
            savefig_root=savefig_root,
            file_name=Path("num_trace_plot_beta1.png"),
        )
        plot_correlogram(
            param_idx=0,
            n_iter=n_iter,
            methods=[
                ("GS4Cox", gs4_samples),
                ("MH", mh_samples),
                ("HMC", hmc_samples),
                ("NUTS", nuts_samples),
                ("MALA", mala_samples),
                ("Cox-PG", cpg_samples),
            ],
            savefig_root=savefig_root,
            file_name=Path("num_correlogram_beta1.png"),
        )
        if calc_intervals:
            plot_forestplot(
                samples=[gs4_samples, mh_samples, hmc_samples, nuts_samples, mala_samples, cpg_samples],
                burn_in=burn_in,
                mple_estimates=cph_mple,
                mple_lower=cph_mple_lower,
                mple_upper=cph_mple_upper,
                savefig_root=savefig_root,
                file_name=Path("num_forestplot_beta1.png"),
            )
