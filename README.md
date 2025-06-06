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
    - sample size (default: `100`).
  - `--beta-true` | `--T`
    - true value of coefficents (default: `'1.0,0.5'`).
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
  - `--ablation-correction` | `--A`
    - set to `True` when comparing results with and without finite-sample corrections (default: `False`).

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
  - `--ablation-correction` | `--A`
    - set to `True` when comparing results with and without finite-sample corrections (default: `False`).

### File Description
- src/cox_sampler.py
  - `GS4Cox`
    - __This class constitutes our main contribution.__
    - Gibbs Sampler for the Cox regression based on four key components Class
      - general Bayesian framework
      - composite partial likelihood
      - P'olya-Gamma augmentation scheme
      - finite correction
  - `CoxMHSampler`
    - Metropolis sampler for Cox regression in general Bayesian framework Class

- src/utils/evaluation_metrics.py
  - functions for evaluating MCMC performance
- src/utils/plot_figure.py
  - class for plotting figure
- src/utils/pl_score_hessian.py
  - functions for calculating score and Hessian for ablation study
- src/utils/select_learning_rate.py
  - class for selecting learning rate of general Bayesian inference

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