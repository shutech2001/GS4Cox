from __future__ import annotations

import os
import string
from typing import List
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from statsmodels.tsa import stattools  # type: ignore

plt.style.use('seaborn-v0_8-bright')


class PlotResult:
    """Parent Class of Plot Result
    """
    def __init__(
        self,
        n_iter: int,
        mh_samples: NDArray,
        gs4c_samples: NDArray,
        gs4c_samples_corrected: NDArray,
        cph_mpl_estimates: NDArray,
        savefig_root: Path,
    ) -> None:
        """
        Args:
            n_iter (int): iteration of sampling.
            mh_samples (NDArray): samples from MH-sampler
            gs4c_samples (NDArray): samples from GS4Cox (without finite-sample corrections)
            gs4c_samples_corrected (NDArray): samples from GS4Cox (with finite-sample corrections)
            cph_mpl_estimates (NDArray): maximum partial likelihood estimates
            savefig_root (Path): root directory for saving figures
        """
        self.n_iter: int = n_iter
        self.mh_samples: NDArray = mh_samples
        self.gs4c_samples: NDArray = gs4c_samples
        self.gs4c_samples_corrected: NDArray = gs4c_samples_corrected
        self.cph_mpl_estimates: NDArray = cph_mpl_estimates
        self.savefig_root: Path = savefig_root
        os.makedirs(savefig_root, exist_ok=True)


class PlotSyntheticResult(PlotResult):
    def __init__(
        self,
        n_iter: int,
        mh_samples: NDArray,
        gs4c_samples: NDArray,
        gs4c_samples_corrected: NDArray,
        cph_mpl_estimates: NDArray,
        savefig_root: Path,
    ) -> None:
        super().__init__(
            n_iter,
            mh_samples,
            gs4c_samples,
            gs4c_samples_corrected,
            cph_mpl_estimates,
            savefig_root,
        )

    def pict_trace_plot(self, beta_true: NDArray, file_name: str = "synthetic_trace_plot.png") -> None:
        """pict trace plot for each sampler

        Args:
            beta_true (NDArray): true values of parameters
            file_name (str, optional): file name of trace plot. Defaults to "synthetic_trace_plot.png".
        """
        fig, ax_array = plt.subplots(
            int(len(beta_true) // 2 + (1 if len(beta_true) % 2 == 1 else 0)),
            2,
            figsize=(25, 16),
            tight_layout=True
        )
        axes = ax_array.flatten()

        for i, ax in enumerate(axes):
            # Trace plots of each method
            ax.plot(
                [i for i in range(self.n_iter)],
                self.mh_samples[:, i],
                linewidth=3,
                alpha=1.0,
                label='MH-Hessian',
                color='limegreen',
            )
            ax.plot(
                [i for i in range(self.n_iter)],
                self.gs4c_samples[:, i],
                linewidth=3,
                alpha=1.0,
                label='GS4Cox (pre-correction)',
                color='violet',
            )
            ax.plot(
                [i for i in range(self.n_iter)],
                self.gs4c_samples_corrected[:, i],
                linewidth=3,
                alpha=1.0,
                label='GS4Cox',
                color='red',
            )

            # plot horizontal line
            ax.axhline(
                y=beta_true[i],
                linestyle='dashed',
                color='black',
                linewidth=4.5,
                label='True Value',
            )
            ax.axhline(
                y=self.cph_mpl_estimates[i],
                linestyle='dotted',
                color='blue',
                linewidth=4.5,
                label="Estimated Value in Cox's regression models",
            )

            ax.set_title(fr'$\beta_{i+1}={beta_true[i]:.1f}$', fontsize=40)
            ax.set_xlim(0.5, self.n_iter)
            ax.tick_params(labelsize=30)

        fig.supxlabel('Iteration', fontsize=40)
        fig.tight_layout()
        fig.legend(
            labels=['MH-Hessian', 'GS4Cox (pre-correction)', 'GS4Cox'],
            fontsize=45,
            bbox_to_anchor=(0.5, 1.05),
            ncol=4,
            loc='center'
        )
        plt.savefig(self.savefig_root / file_name)

    def pict_post_dist(self, beta_true: NDArray, burn_in: int, file_name: str = "synthetic_post_dist.png") -> None:
        """pict posterior distribution

        Args:
            beta_true (NDArray): true values of parameters
            burn_in (int): burn-in period.
            file_name (str, optional): file name of histogram for posterior distribution.
                                       Defaults to "synthetic_post_dist.png".
        """
        fig, ax_array = plt.subplots(
            int(len(beta_true) // 2 + (1 if len(beta_true) % 2 == 1 else 0)),
            2,
            figsize=(25, 16),
            tight_layout=True
        )
        axes = ax_array.flatten()

        for i, ax in enumerate(axes):
            # Posterior distribution of each method
            ax.hist(self.mh_samples[burn_in:][:, i], label='MH-Hessian', color='limegreen', alpha=0.4)
            ax.hist(self.gs4c_samples[burn_in:][:, i], label='GS4Cox (pre-correction)', color='violet', alpha=0.4)
            ax.hist(self.gs4c_samples_corrected[burn_in:][:, i], label='GS4Cox', color='red', alpha=0.4)

            # plot vertical line
            ax.axvline(
                x=beta_true[i],
                linestyle='dashdot',
                color='black',
                linewidth=4.0,
                label='True Value',
            )
            ax.axvline(
                x=self.cph_mpl_estimates[i],
                linestyle='dotted',
                color='blue',
                linewidth=4.0,
                label="Estimated Value in Cox's regression models",
            )

            ax.set_title(fr'$\beta_{i+1}={beta_true[i]:.1f}$', fontsize=40)
            ax.set_yscale('log')
            ax.tick_params(labelsize=30)
        fig.legend(
            labels=['MH-Hessian', 'GS4Cox (pre-correction)', 'GS4Cox'],
            fontsize=45,
            bbox_to_anchor=(0.5, 1.05),
            ncol=4,
            loc='center',
        )
        plt.savefig(self.savefig_root / file_name)

    def pict_correlogram(self, beta_true: NDArray, file_name: str = "synthetic_correlogram.png") -> None:
        """pict correlogram

        Args:
            beta_true (NDArray): true values of parameters
            file_name (str, optional): file name of correlogram. Defaults to "synthetic_correlogram.png".
        """
        methods = {
            'MH-Hessian': self.mh_samples,
            'GS4Cox (Ours)': self.gs4c_samples_corrected,
        }

        param_names = [r'$\beta_{}$'.format(i+1) for i in range(len(beta_true))]
        labels = list(string.ascii_lowercase)
        fig, axes = plt.subplots(2, int(len(beta_true)), figsize=(35, 12))
        axes = axes.flatten()

        for i, (method_name, samples) in enumerate(methods.items()):
            for j in range(int(len(beta_true))):
                idx = i * int(len(beta_true)) + j
                ax = axes[idx]

                acf_vals = stattools.acf(samples[:, j], fft=True, nlags=40)
                lags = np.arange(len(acf_vals))

                markerline, _, _ = ax.stem(lags, acf_vals, basefmt=" ")
                plt.setp(markerline, 'markerfacecolor', 'b')
                ax.set_title(f'({labels[idx]}) {method_name} - {param_names[j]}', fontsize=35)
                ax.tick_params(labelsize=20)
        fig.supxlabel('Lag', fontsize=45)
        fig.supylabel('Autocorrelation', fontsize=45, x=-0.0001)
        plt.tight_layout()
        plt.savefig(self.savefig_root / file_name)


class PlotActualResult(PlotResult):
    def __init__(
        self,
        n_iter: int,
        mh_samples: NDArray,
        gs4c_samples: NDArray,
        gs4c_samples_corrected: NDArray,
        cph_mpl_estimates: NDArray,
        savefig_root: Path,
        cph_mpl_lower: NDArray,
        cph_mpl_upper: NDArray,
    ) -> None:
        """
        Args:
            ...
            cph_mpl_lower (NDArray): lower bound of maximum partial likelihood estimates
                - e.g., if using CoxPHFitter as cph, then `cph.confidence_intervals_['95% lower-bound']`
            cph_mpl_upper (NDArray): upper bound of maximum partial likelihood estimates
                - e.g., if using CoxPHFitter as cph, then `cph.confidence_intervals_['95% upper-bound']`
        """
        super().__init__(
            n_iter,
            mh_samples,
            gs4c_samples,
            gs4c_samples_corrected,
            cph_mpl_estimates,
            savefig_root,
        )
        self.cph_mpl_lower: NDArray = cph_mpl_lower
        self.cph_mpl_upper: NDArray = cph_mpl_upper

    def pict_trace_plot(self, colnames: List, file_name: str = "actual_trace_plot.png") -> None:
        """pict trace plot for each sampler

        Args:
            colnames (List): names of columns
            file_name (str, optional): file name of trace plot. Defaults to "actual_trace_plot.png".
        """
        fig, ax_array = plt.subplots(
            int(len(colnames) // 2 + (1 if len(colnames) % 2 == 1 else 0)),
            2,
            figsize=(35, 25),
            tight_layout=True,
        )
        axes = ax_array.flatten()

        for i, ax in enumerate(axes):
            if i + 1 > len(colnames):
                ax.axis('off')
                continue
            # Trace plots of each method
            ax.plot(
                [i for i in range(self.n_iter)],
                self.mh_samples[:, i],
                linewidth=3,
                alpha=1.0,
                label='MH-Hessian',
                color='limegreen',
            )
            ax.plot(
                [i for i in range(self.n_iter)],
                self.gs4c_samples[:, i],
                linewidth=3,
                alpha=1.0,
                label='GS4Cox (pre-correction)',
                color='violet',
            )
            ax.plot(
                [i for i in range(self.n_iter)],
                self.gs4c_samples_corrected[:, i],
                linewidth=3,
                alpha=1.0,
                label='GS4Cox',
                color='red',
            )

            # plot horizontal line
            ax.axhline(
                y=self.cph_mpl_estimates[i],
                linestyle='dotted',
                color='blue',
                linewidth=4.5,
                label="Estimated Value in Cox's regression models",
            )

            ax.set_title(f'coefficient for {colnames[i]}', fontsize=45)
            ax.set_xlim(0.5, self.n_iter)
            ax.tick_params(labelsize=30)

        fig.supxlabel('Iteration', fontsize=45, y=-0.001)
        fig.tight_layout()
        fig.legend(
            labels=['MH-Hessian', 'GS4Cox (pre-correction)', 'GS4Cox'],
            fontsize=50,
            bbox_to_anchor=(0.5, 1.04),
            ncol=3,
            loc='center'
        )
        plt.savefig(self.savefig_root / file_name)

    def pict_post_dist(self, colnames: List, burn_in: int, file_name: str = "actual_post_dist.png") -> None:
        """pict posterior distribution

        Args:
            colnames (List): names of columns
            burn_in (int): burn-in period
            file_name (str, optional): names of histogram for posterior distribution.
                                       Defaults to "actual_post_dist.png".
        """
        fig, ax_array = plt.subplots(
            int(len(colnames) // 2 + (1 if len(colnames) % 2 == 1 else 0)),
            2,
            figsize=(35, 30),
            tight_layout=True,
        )
        axes: NDArray = ax_array.flatten()

        for i, ax in enumerate(axes):
            if i + 1 > len(colnames):
                ax.axis('off')
                continue
            # Posterior distribution of each method
            ax.hist(self.mh_samples[burn_in:][:, i], label='MH-Hessian', color='limegreen', alpha=0.4)
            ax.hist(self.gs4c_samples[burn_in:][:, i], label='GS4Cox (pre-correction)', color='violet', alpha=0.4)
            ax.hist(self.gs4c_samples_corrected[burn_in:][:, i], label='GS4Cox', color='red', alpha=0.4)

            # plot vertical line
            ax.axvline(
                x=self.cph_mpl_estimates[i],
                linestyle='dotted',
                color='blue',
                linewidth=10.0,
                label="Estimated Value in Cox's regression models",
            )

            ax.set_title(f'coefficient for {colnames[i]}', fontsize=45)
            ax.set_yscale('log')
            ax.tick_params(labelsize=30)
        fig.legend(
            labels=['MH-Hessian', 'GS4Cox (pre-correction)', 'GS4Cox'],
            fontsize=55,
            bbox_to_anchor=(0.5, 1.03),
            ncol=3,
            loc='center',
        )
        plt.savefig(self.savefig_root / file_name)

    def pict_correlogram(
        self,
        colnames: List,
        file_name_without_method: str = "actual_correlogram.png",
        empty_ax: bool = False
    ) -> None:
        """pict correlogram for each method

        Args:
            colnames (List): names of columns
            file_name_without_method (str, optional): file name of correlogram without method name.
                                                      Defaults to "actual_correlogram.png".
                - output file name is 'actual_correlogram_mh.png' for MH sampler
                - output file name is 'actual_correlogram_gs.png' for Gibbs sampler
            empty_ax (bool, optional): whether to leave a blank when there is an empty subfigure.
                                       Defaults to False.
                - (Note) This is intended only for application to the R lung dataset, so 'True' is not recommended.
        """
        name, ext = file_name_without_method.rsplit('.', 1)
        mh_file_name: str = name + "_mh" + '.' + ext
        gs_file_name: str = name + "_gs" + '.' + ext

        param_names = colnames

        # correlogram for MH-Hessian
        fig, axes = plt.subplots(
            2,
            int(len(colnames) // 2 + (1 if len(colnames) % 2 == 1 else 0)),
            figsize=(30, 12),
        )
        axes = axes.flatten()
        for j in range(len(colnames)):
            idx = j
            ax = axes[idx]
            if (idx == 7) and empty_ax:
                ax.axis('off')
                continue
            acf_vals = stattools.acf(self.mh_samples[:, j], fft=True, nlags=40)
            lags = np.arange(len(acf_vals))

            markerline, _, _ = ax.stem(lags, acf_vals, basefmt=" ")
            plt.setp(markerline, 'markerfacecolor', 'b')
            ax.set_title(f'{param_names[j]}', fontsize=30)
            ax.tick_params(labelsize=20)
        fig.supxlabel('Lag', fontsize=35)
        fig.supylabel('Autocorrelation', fontsize=35, x=-0.0001)
        plt.tight_layout()
        plt.savefig(self.savefig_root / mh_file_name)

        # correlogram for GS4Cox
        fig, axes = plt.subplots(
            2,
            int(len(colnames) // 2 + (1 if len(colnames) % 2 == 1 else 0)),
            figsize=(30, 12),
        )
        axes = axes.flatten()
        for j in range(len(colnames)):
            idx = j
            ax = axes[idx]
            if (idx == 7) and empty_ax:
                ax.axis('off')
                continue
            acf_vals = stattools.acf(self.gs4c_samples_corrected[:, j], fft=True, nlags=40)
            lags = np.arange(len(acf_vals))

            markerline, _, _ = ax.stem(lags, acf_vals, basefmt=" ")
            plt.setp(markerline, 'markerfacecolor', 'b')
            ax.set_title(f'{param_names[j]}', fontsize=30)
            ax.tick_params(labelsize=20)
        fig.supxlabel('Lag', fontsize=35)
        fig.supylabel('Autocorrelation', fontsize=35, x=-0.0001)
        plt.tight_layout()
        plt.savefig(self.savefig_root / gs_file_name)

    def pict_forestplot(
        self,
        colnames: List,
        burn_in: int,
        alpha: float = .05,
        methods: List = ['Std. Cox Reg.', 'MH-Hessian', 'GS4Cox (pre-correction)', 'GS4Cox'],
        file_name: str = "actual_forestplot.png"
    ) -> None:
        """pict forest plot

        Args:
            colnames (List): names of columns
            burn_in (int): burn-in periord.
            alpha (float, optional): significance level. Defaults to .05.
            methods (List, optional): names of methods.
                                      Defaults to ['Std. Cox Reg.', 'MH-Hessian', 'GS4Cox (pre-correction)', 'GS4Cox'].
            file_name (str, optional): file name of forest plot. Defaults to "actual_forestplot.png".
        """
        lowers = [
            self.cph_mpl_lower,
            np.quantile(self.mh_samples[burn_in:], alpha/2, axis=0),
            np.quantile(self.gs4c_samples[burn_in:], alpha/2, axis=0),
            np.quantile(self.gs4c_samples_corrected[burn_in:], alpha/2, axis=0),
        ]
        uppers = [
            self.cph_mpl_upper,
            np.quantile(self.mh_samples[burn_in:], 1-alpha/2, axis=0),
            np.quantile(self.gs4c_samples[burn_in:], 1-alpha/2, axis=0),
            np.quantile(self.gs4c_samples_corrected[burn_in:], 1-alpha/2, axis=0),
        ]
        mean_values = [
            self.cph_mpl_estimates,
            self.mh_samples[burn_in:].mean(axis=0),
            self.gs4c_samples[burn_in:].mean(axis=0),
            self.gs4c_samples_corrected[burn_in:].mean(axis=0),
        ]

        n_methods = len(methods)
        n_params = len(colnames)

        y_base = np.arange(n_params)
        width = 0.15
        offsets = np.linspace(-1.5*width, 1.5*width, n_methods)
        colors = ['blue', 'limegreen', 'violet', 'red']
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, (method, dx, color) in enumerate(zip(methods, offsets, colors)):
            y = y_base + dx
            x = mean_values[i]
            err_low = x - lowers[i]
            err_high = uppers[i] - x
            err = np.vstack((err_low, err_high))

            ax.errorbar(
                x, y,
                xerr=err,
                fmt='o',
                capsize=4,
                markersize=6,
                label=method,
                color=color
            )

        ax.set_yticks(y_base)
        ax.set_yticklabels(colnames, fontsize=15)
        ax.tick_params(labelsize=15)
        ax.set_xlabel('Estimate with 95% CI/CrI', fontsize=20)
        ax.grid(axis='x', linestyle=':', linewidth=0.5)

        fig.legend(
            methods, 
            fontsize=20,
            bbox_to_anchor=(0.5, 1.1),
            ncol=2,
            loc='center',
        )

        plt.tight_layout()
        plt.savefig(self.savefig_root / file_name)
