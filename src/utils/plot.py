from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from statsmodels.tsa import stattools  # type: ignore

plt.style.use("seaborn-v0_8-bright")


# trace plot
def plot_trace(
    param_idx: int,
    n_iter: int,
    methods: List,
    true_values: Optional[np.ndarray] = None,
    mple_estimates: Optional[np.ndarray] = None,
    savefig_root: Path = Path("figures"),
    file_name: Path = Path("trace_plot.png"),
) -> None:
    """Plot trace plot

    Args:
        param_idx (int): index of parameter
        n_iter (int): number of iterations
        methods (List): list of methods. like below:
            methods = [
                ("GS4Cox", gs4_samples),
                ("MH", mh_samples),
                ("HMC", hmc_samples),
                ("NUTS", nuts_samples),
                ("MALA", mala_samples),
                ("Cox-PG", cpg_samples),
            ]
        true_values (Optional[np.ndarray], optional): true values of parameters. Defaults to None.
        mple_estimates (Optional[np.ndarray], optional): maximum partial likelihood estimates. Defaults to None.
        file_name (str, optional): file name of trace plot. Defaults to "trace_plot.png".
    """
    y_all = np.concatenate([m[1][:, param_idx] for m in methods])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    ref_vals = [
        float(true_values[param_idx]) if true_values is not None else None,
        float(mple_estimates[param_idx]) if mple_estimates is not None else None,
    ]
    ref_vals = [v for v in ref_vals if v is not None]
    if len(ref_vals) > 0:
        y_min = min(y_min, *[v for v in ref_vals if v is not None])
        y_max = max(y_max, *[v for v in ref_vals if v is not None])
    pad = 0.05 * (y_max - y_min + 1e-12)
    ylim = (y_min - pad, y_max + pad)
    if param_idx == 0:
        # for paper figure
        ylim = (0 - 1e-5, y_max + pad)

    fig, ax_array = plt.subplots(2, 3, figsize=(48, 27), tight_layout=True)
    axes = ax_array.flatten()
    iters = np.arange(n_iter)
    for ax, (name, samples) in zip(axes, methods):
        ax.plot(iters, samples[:, param_idx], linewidth=5, alpha=0.8)

        if true_values is not None:
            ax.axhline(
                y=float(true_values[param_idx]), linestyle="--", color="black", linewidth=7.5, label="True parameter"
            )
        if mple_estimates is not None:
            ax.axhline(
                y=float(mple_estimates[param_idx]),
                linestyle=":",
                color="red",
                linewidth=7.5,
                label="Maximum partial likelihood estimate",
            )
        ax.set_title(name, fontsize=50)
        ax.set_xlim(0, n_iter - 1)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=30)

    fig.supxlabel("Iteration", fontsize=50, y=0.0)
    fig.supylabel(rf"Trace of $\beta_{{{param_idx+1}}}$", x=0, fontsize=50)

    handles = [
        Line2D([0], [0], color="black", linestyle="dashed", linewidth=7.5, label="True parameter"),
        Line2D([0], [0], color="red", linestyle="dotted", linewidth=7.5, label="Maximum partial likelihood estimate"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07), fontsize=50)

    plt.tight_layout()
    os.makedirs(savefig_root, exist_ok=True)
    plt.savefig(savefig_root / file_name)


def plot_correlogram(
    param_idx: int,
    n_iter: int,
    methods: List,
    savefig_root: Path = Path("figures"),
    file_name: Path = Path("correlogram.png"),
) -> None:
    nlags = min(100, n_iter // 5)
    ci_line = 1.96 / np.sqrt(n_iter)

    fig, ax_array = plt.subplots(2, 3, figsize=(48, 27), tight_layout=True)
    axes = ax_array.flatten()

    for ax, (name, samples) in zip(axes, methods):
        x = np.asarray(samples[:, param_idx])

        acf_vals = stattools.acf(x, fft=True, nlags=nlags)  # [0..nlags]
        lags = np.arange(len(acf_vals))

        markerline, stemlines, baseline = ax.stem(lags, acf_vals, basefmt=" ")
        markerline.set_markerfacecolor("k")
        markerline.set_markersize(8)
        plt.setp(stemlines, linewidth=4)

        ax.axhline(0.0, linestyle="--", color="black", linewidth=5.0, label="0")
        ax.axhline(+ci_line, linestyle=":", color="gray", linewidth=6.0)
        ax.axhline(-ci_line, linestyle=":", color="gray", linewidth=6.0)

        ax.set_title(name, fontsize=50)
        ax.set_xlim(0, nlags)
        ax.set_ylim(-1.0, 1.0)
        ax.tick_params(labelsize=30)

    fig.supxlabel("Lag", fontsize=50, y=0.0)
    fig.supylabel(r"Autocorrelation of $\beta_{1}$ samples", fontsize=50, x=0.0)

    plt.tight_layout()
    os.makedirs(savefig_root, exist_ok=True)
    plt.savefig(savefig_root / file_name)


def plot_forestplot(
    samples: List[np.ndarray],
    burn_in: int,
    mple_estimates: Optional[np.ndarray] = None,
    mple_lower: Optional[np.ndarray] = None,
    mple_upper: Optional[np.ndarray] = None,
    savefig_root: Path = Path("figures"),
    file_name: Path = Path("forestplot.png"),
):
    """Plot forest plot

    Args:
        samples (List[np.ndarray]): samples from each method. like below:
            samples = [gs4_samples, mh_samples, hmc_samples, nuts_samples, mala_samples, cpg_samples]
        burn_in (int): burn-in period
        mple_estimates (Optional[np.ndarray], optional): maximum partial likelihood estimates. Defaults to None.
        mple_lower (Optional[np.ndarray], optional): lower bound of 95% confidence interval. Defaults to None.
        mple_upper (Optional[np.ndarray], optional): upper bound of 95% confidence interval. Defaults to None.
        savefig_root (Path, optional): root directory for saving figures. Defaults to Path("figures").
        file_name (Path, optional): file name of forest plot. Defaults to Path("forestplot.png").
    """
    alpha = 0.05
    lower_bound = alpha / 2
    upper_bound = 1 - alpha / 2

    lower_values: List[np.ndarray] = []
    upper_values: List[np.ndarray] = []
    mean_values: List[np.ndarray] = []

    if mple_lower is not None:
        lower_values.append(mple_lower)
    if mple_upper is not None:
        upper_values.append(mple_upper)
    if mple_estimates is not None:
        mean_values.append(mple_estimates)

    for sample in samples:
        sample_bi = sample[burn_in:]
        lower_values.append(np.quantile(sample_bi, lower_bound, axis=0))
        upper_values.append(np.quantile(sample_bi, upper_bound, axis=0))
        mean_values.append(sample_bi.mean(axis=0))

    lv = np.asarray(lower_values)
    uv = np.asarray(upper_values)
    mv = np.asarray(mean_values)

    n_methods, n_params = mv.shape

    idx_rev = np.arange(n_params - 1, -1, -1)
    colnames = [rf"$\beta_{i+1}$" for i in range(n_params)]
    colnames_rev = [colnames[i] for i in idx_rev]
    mv_rev = mv[:, idx_rev]
    lv_rev = lv[:, idx_rev]
    uv_rev = uv[:, idx_rev]

    y_base = np.arange(n_params)
    width = 0.08
    offsets = np.linspace(-((n_methods - 1) / 2) * width, ((n_methods - 1) / 2) * width, n_methods)

    methods = [
        "MPLE + CI",
        "GS4Cox",
        "MH",
        "HMC",
        "NUTS",
        "MALA",
        "Cox-PG",
    ]

    palette = {
        "MPLE + CI": "gray",
        "GS4Cox": "blue",
        "MH": "red",
        "HMC": "green",
        "NUTS": "yellow",
        "MALA": "purple",
        "Cox-PG": "orange",
    }

    markers = {
        "MPLE + CI": "o",
        "GS4Cox": "s",
        "MH": "X",
        "HMC": "v",
        "NUTS": ">",
        "MALA": "<",
        "Cox-PG": "h",
    }

    fig, ax = plt.subplots(figsize=(32, 40), tight_layout=True)

    for i, (method, dx) in enumerate(zip(methods, offsets)):
        y = y_base + dx
        x = mv_rev[i]
        err_low = x - lv_rev[i]
        err_high = uv_rev[i] - x
        xerr = np.vstack([err_low, err_high])

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt=markers[method],
            linestyle="none",
            markersize=20,
            capsize=12,
            linewidth=6,
            color=palette[method],
            ecolor=palette[method],
            label=method,
        )

    ax.set_yticks(y_base)
    ax.set_yticklabels(colnames_rev, fontsize=70)
    ax.tick_params(labelsize=40)
    ax.set_xlabel("Estimates with 95% CI/CrI (log-hazard scale)", fontsize=50)
    ax.grid(axis="x", linestyle=":", linewidth=1.5, alpha=0.7)

    ax.legend(fontsize=50, bbox_to_anchor=(0.5, 1.00), loc="lower center", ncol=4, frameon=False)

    os.makedirs(savefig_root, exist_ok=True)
    plt.savefig(savefig_root / file_name)
