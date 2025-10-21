from __future__ import annotations

import argparse
import time as t
import warnings

from lifelines import CoxPHFitter  # type: ignore
import numpy as np
from numpy.typing import NDArray
import pandas as pd  # type: ignore

from cox_sampler import CoxSampler, GS4Cox, CoxMHSampler, CoxHMCSampler, CoxNUTSSampler, CoxMALASampler, CoxPGSampler
from data import SyntheticDataGenerator4CoxReg
from utils.evaluation_metrics import compute_esr, compute_ess, compute_mcse

warnings.filterwarnings("ignore")
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
    calc_intervals: bool,
) -> None:
    data_generator = SyntheticDataGenerator4CoxReg(n=n, beta_true=beta_true)

    if use_ties:
        covariates, time, event = data_generator.simulate_cox_data_ties(rounding=rounding)
    else:
        covariates, time, event = data_generator.simulate_cox_data()

    # calculate lower and upper bounds of confidence interval
    alpha: float = 0.05
    lower_bound: float = alpha / 2
    upper_bound: float = 1 - alpha / 2

    # Maximum partial likelihood estimates
    df = pd.DataFrame(covariates, columns=[f"X{i+1}" for i in range(len(beta_true))])
    df["time"] = time
    df["event"] = event
    cph = CoxPHFitter()
    cph.fit(df, duration_col="time", event_col="event")
    cph_mple: NDArray = np.array(cph.params_)
    print("Maximum partial likelihood estimates:", cph_mple)
    if calc_intervals:
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
    nuts = CoxNUTSSampler(covariates=covariates)
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


def parse_beta(beta_str: str) -> NDArray:
    """Generate an NDArray from a comma-separated string

    Args:
        beta_str (str): comma-separated parameter (e.g., 5.0,3.5)

    Returns:
        NDArray: true coefficient vector
    """
    try:
        beta_list: list[float] = [float(b.strip()) for b in beta_str.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError("beta_true must be a comma-separated list of numbers. (e.g., 5.0,3.5)") from e
    return np.array(beta_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cox Regression Simulation with GS4Cox, optimal MH and Standard Cox Regression"
    )
    parser.add_argument("--data-size", "--N", type=int, default=300, help="sample size (default: 300).")
    parser.add_argument(
        "--beta-true",
        "--T",
        type=parse_beta,
        default="1.0,-1.0,0.5,-0.5,0.3,-0.3,0.1,-0.1",
        help="true value of coefficients (default: 1.0,-1.0,0.5,-0.5,0.3,-0.3,0.1,-0.1).",
    )
    parser.add_argument(
        "--learning-rate",
        "--L",
        type=float,
        default="1.0",
        help="learning rate for general Bayesian framework (default: 1.0).",
    )
    parser.add_argument(
        "--iteration", "--I", type=int, default=1000, help="the number of total iterations (default: 1000)."
    )
    parser.add_argument("--burn-in", "--B", type=int, default=500, help="the number of burn-in (default: 500).")
    parser.add_argument(
        "--use-ties",
        "--U",
        type=bool,
        default=False,
        help="set to `True` when performing simulation based on the same event occurrence data (default: False).",
    )
    parser.add_argument(
        "--rounding", "--R", type=float, default=0.001, help="rounding unit for generating tie data (default: 0.001)."
    )
    parser.add_argument(
        "--calc-intervals",
        "--CI",
        type=bool,
        default=True,
        help="set to `True` when calculating 95% confidence/credible intervals of estimated coefficients (default: True).",  # noqa: E501
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
        calc_intervals=args.calc_intervals,
    )
