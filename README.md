# GS4Cox

Materials for "[**Efficient Gibbs Sampling in Cox Regression Models Using Composite Partial Likelihood and P´olya-Gamma Augmentation**](https://arxiv.org/abs/2506.04675)".

## What is this repo?

This repository includes an implementation of GS4Cox, a Gibbs sampler for Cox regression, as described in our paper.
It also contains the numerical experiments and actual data experiments presented in the paper

### Requirements and Setup
```
# clone the repository
git clone git@github.com:shutech2001/GS4Cox.git

# build the environment with poetry
poetry install

# activate virtual environment
eval $(poetry env activate)

# [Option] to activate the interpreter, select the following output as the interpreter.
poetry env info --path
```

### executing simulation
- `python src/synthetic_data_experiment.py`
  - `--data-size` | `--N`
    - sample size (default: `300`).
  - `--beta-true` | `--T`
    - true value of coefficents (default: `'1.0,-1.0,0.5,-0.5,0.3,-0.3,0.1,-0.1'`).
  - `--learning-rate` | `--L`
    - learning rate for general Bayesian framework (default: `1.0`).
  - `--iteration` | `--I`
    - the number of total iterations (default: `1000`).
  - `--burn-in` | `--B`
    - the number of burn-in (default: `500`).
  - `--use-ties` | `--U`
    - set to `True` when performing simulation based on the same event occurrence data (default: `False`).
  - `--rounding` | `--R`
    - rounding unit for generating tie data (default: `0.001`).
  - `--calc-intervals` | `--CI`
    - set to `True` when calculating 95% confidence/credible intervals of estimated coefficients (default: `True`).
  - `--plot-results` | `--P`
    - set to `True` when plotting trace plots, correlograms and forest plots (default: `False`).
  - `--savefig-root` | `--S`
    - root directory for saving figures (default: `figures`).

- `python src/actual_data_experiment.py`
  - `--package` | `--P`
    - name of the R package that contains the data to be imported (default: `survival`).
  - `--dataset` | `--D`
    - name of the R dataset (default: `lung`).
  - `--time-col-name` | `--TC`
    - column name of representing 'time' (default: `time`).
  - `--event-col-name` | `--EC`
    - column name of representing 'event status' (default: `status`).
  - `--event-indicator` | `--EI`
    - indicator representing the event in the 'event-col-name' (default: `2`).
  - `--covariates-column-names` | `--CN`
    - column names of regression covariates (default: `age sex ph.ecog ph.karno pat.karno meal.cal wt.loss`)
  - `--iteration` | `--I`
    - the number of total iterations (default: `1000`).
  - `--burn-in` | `--B`
    - the number of burn-in (default: `500`).
  - `--learning-rate` | `--L`
    - learning rate for general Bayesian framework (default: `1.0`).
    - If set to `0`, execute learning rate selection by GPC (proposed by Syring and Martin, 2019).
  - `--calc-intervals` | `--CI`
    - set to `True` when calculating 95% confidence/credible intervals of estimated coefficients (default: `True`).
  - `--plot-results` | `--P`
    - set to `True` when plotting trace plots, correlograms and forest plots (default: `False`).
  - `--savefig-root` | `--S`
    - root directory for saving figures (default: `figures`).

### File Description
- src/cox_sampler.py
  - `CoxSampler`
    - Parent class of MCMC sampler for Cox regression models
  - `GS4Cox`
    - __This class constitutes our main contribution.__
    - Gibbs Sampler for the Cox regression based on four key components class
      - general Bayesian framework
      - composite partial likelihood
      - P'olya-Gamma augmentation scheme
      - open-faced sandwich
  - `CoxMHSampler`
    - Metropolis sampler for Cox regression in general Bayesian framework class
  - `CoxPGSampler`
    - Cox-P'olya-Gamma algorithm class
    - Original citation: Benny Ren, Jeffrey S Morris, Ian Barnett, The Cox-P\'olya-Gamma algorithm for flexible Bayesian inference of multilevel survival models, Biometrics, Volume 81, Issue 3, September 2025, ujaf121, doi: 10.1093/biomtc/ujaf121
  - `CoxHMCSampler`
    - Hamiltonian Monte Carlo sampler for Cox regression in general Bayesian framework class
  - `CoxNUTSsampler`
    - No-U-Turn sampler for Cox regression in general Bayesian framework class
  - `CoxMALASampler`
    - Metropolis-Adjusted Langevin algorithm sampler for Cox regression in general Bayesian framework class

- src/utils/evaluation_metrics.py
  - functions for evaluating MCMC performance
- src/utils/plot.py
  - functions for plotting figure

- src/data/generate_synthetic_data.py
  - class for generating synthetic data for Cox regression Class
- src/data/import_r_data.py
  - functions for importing data from R packages

## Citation
```
@article{tamano2025efficient,
    author={Tamano, Shu and Tomo, Yui},
    journal={arXiv preprint arXiv:2506.04675},
    title={Efficient {Gibbs} Sampling in {Cox} Regression Models Using Composite Partial Likelihood and {P\'olya-Gamma} Augmentation},
    year={2025},
}
```

## Contact

If you have any question, please feel free to contact: stamano@niid.go.jp