# P\'olya-Gamma Gibbs Sampler for Cox Regression
Polya-Gamma Gibbs Sampler for Cox Regression Models in General Bayes Framework

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
  - `--iteration` | `--I`
    - the number of total iterations (default: `500`).
  - `--burn-in` | `--B`
    - the number of burn-in (default: `400`).
  - `--use-ties` | `--U`
    - set to `True` when performing simulation based on the same event occurrence data (default: `False`).
  - `--rounding` | `--R`
    - rounding unit for generating tie data (default: `0.001`).

## File Description
- src/cox_pg_sampler.py
  - `CoxPGSampler`
    - P\'olya Gamma Gibbs sampler for Cox Regression Class 
- src/generate_synthetic_data.py
  - `SyntheticDataGenerater4CoxReg`
    - generate synthetic data Class

## Citation
```
```