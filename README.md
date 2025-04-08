# $P\'olya$-$Gamma$ Gibbs Sampler for Cox Regression
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

## File Description
- src/cox_pg_sampler.py
  - `CoxPGSampler`
    - P\'olya Gamma Gibbs sampler for Cox Regression Class 
- src/generate_synthetic_data.py
  - `SyntheticDataGenerater4CoxReg`
    - generate synthetic data Class

# TODO
- simulation
  - naive cox
  - naive MH with generalized bayesian
  - previous study with PG
  - ours
- real data (R open data)
  compare with execution time

## Citation
```
```