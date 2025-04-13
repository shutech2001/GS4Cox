# P'olya-Gamma Gibbs Sampler for Cox Regression
Polya-Gamma Gibbs Sampler for Cox Regression Models in General Bayesian Framework

## Abstract

## How to use
### making environment
- when create project
1. `brew install poetry`
2. move to working directory
3. `poetry init`
4. `poetry env use python3.13`
- when use existing project
1. `poetry install`
2. `eval $(poetry env activate)`
3. interpreter as an output of `poetry env info --path`

### executing simulation
- `python src/main.py`
  - `--data-size` | `--N`
    - sample size (default: `100`).
  - `--beta-true` | `--T`
    - true value of coefficents (default: `'1.0,0.5'`).
  - `--learning-rate` | `--L`
    - learning rate for general Bayesian framework (default: `1.0`).
  - `--iteration` | `--I`
    - the number of total iterations (default: `500`).
  - `--burn-in` | `--B`
    - the number of burn-in (default: `400`).
  - `--use-ties` | `--U`
    - set to `True` when performing simulation based on the same event occurrence data (default: `False`).
  - `--rounding` | `--R`
    - rounding unit for generating tie data (default: `0.001`).
  - `--proposal-scale` | `--P`
    - covariance scale of proposal distribution for Metropolis-Hastings (default: `10`).

- `python src/real_data.py`
  - `--package` | `--P`
    - name of the R package that contains the data to be imported (default: `survival`).
  - `--dataset` | `--D`
    - name of the R dataset (default: `lung`).
  - `--id-col-name` | `--IC`
    - column name of representing 'ID' (default: `inst`).
  - `--time-col-name` | `--TC`
    - column name of representing 'time' (default: `time`).
  - `--event-col-name` | `--EC`
    - column name of representing 'event status' (default: `status`).
  - `--event-indicator` | `--EI`
    - indicator representing the event in the 'event-col-name' (default: `2`).
  - `--iteration` | `--I`
    - the number of total iterations (default: `500`).
  - `--burn-in` | `--B`
    - the number of burn-in (default: `400`).
  - `--proposal-scale` | `--P`
    - covariance scale of proposal distribution for Metropolis-Hastings (default: `10`).

## File Description
- src/cox_sampler.py
  - `GBCoxPGSampler`
    - P\'olya Gamma Gibbs sampler for Cox regression in general Bayesian framework Class 
  - `CoxMHSampler`
    - Metropolis sampler for Cox regression in general Bayesian framework Class
- src/evaluation_metrics.py
  - functions for evaluating MCMC performance

- src/data/generate_synthetic_data.py
  - `SyntheticDataGenerater4CoxReg`
    - generate synthetic data for Cox regression Class
- src/data/import_r_data.py
  - functions for importing data from R packages

## Citation
```
```