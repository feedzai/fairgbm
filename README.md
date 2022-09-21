# FairGBM

FairGBM is an easy-to-use and lightweight fairness-aware ML algorithm with state-of-the-art performance on tabular datasets.

FairGBM builds upon the popular [LightGBM](https://github.com/microsoft/LightGBM) algorithm and adds customizable 
constraints for group-wise fairness (_e.g._, equal opportunity, predictive equality) and other global goals (_e.g._, 
specific Recall or FPR prediction targets).

Please consult [the paper](https://arxiv.org/pdf/2209.07850.pdf) for further details.

- [Install](#install)
- [Getting started](#getting-started)
  - [Parameter list](#parameter-list)
  - [_fit(X, Y, constraint_group=S)_](#fitx-y-constraint_groups)
- [Features](#features)
  - [Fairness constraints](#fairness-constraints)
  - [Global constraints](#global-constraints)
- [Technical details](#technical-details)
- [Citing FairGBM](#citing-fairgbm)


## Install

<!--
FairGBM can be installed from [PyPI](https://pypi.org/project/fairgbm/)

```pip install fairgbm```

or from GitHub

```
git clone --recurse-submodules https://github.com/feedzai/fairgbm.git
pip install fairgbm/python-package/
```
-->

Installation instructions:
```
git clone --recurse-submodules https://github.com/feedzai/fairgbm.git
pip install fairgbm/python-package/
```

> **Note**
> Install requires [CMake](https://cmake.org) and an up-to-date C++ compiler (gcc, clang, or mingw).
> You may need to install wheel via `pip install wheel` first.
> For Linux users, glibc >= 2.14 is required.
> For more details see LightGBM's [installation guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html),
> or follow [this link](https://github.com/microsoft/LightGBM/tree/master/python-package) for the Python package
> installation instructions.


## Getting started

You can get FairGBM up and running in just a few lines of Python code:

```python
from fairgbm import FairGBMClassifier

# Instantiate
fairgbm_clf = FairGBMClassifier(
    constraint_type="FNR",    # constraint on equal group-wise TPR (equal opportunity)
    n_estimators=200,         # core parameters from vanilla LightGBM
    random_state=42,          # ...
)

# Train using features (X), labels (Y), and sensitive attributes (S)
fairgbm_clf.fit(X, Y, constraint_group=S)
# NOTE: labels (Y) and sensitive attributes (S) must be in numeric format

# Predict
Y_test_pred = fairgbm_clf.predict_proba(X_test)[:, -1]  # Compute continuous class probabilities (recommended)
# Y_test_pred = fairgbm_clf.predict(X_test)             # Or compute discrete class predictions
```

A more in-depth explanation and other usage examples can be found in the [**_examples folder_**](/examples).

**For Python examples see the [_notebooks folder_](/examples/FairGBM-python-notebooks).**


### Parameter list

The following parameters can be used as key-word arguments for the `FairGBMClassifier` Python class.

| _Name_ | _Description_ | _Default_ |
|:------:|---------------|:---------:|
| `constraint_type` | The type of fairness (group-wise equality) constraint to use (if any). | `FPR,FNR` |
| `global_constraint_type` | The type of global equality constraint to use (if any). | _None_ |
| `multiplier_learning_rate` | The learning rate for the gradient ascent step (w.r.t. Lagrange multipliers). | `0.1` |
| `constraint_fpr_tolerance` | The slack when fulfilling _group-wise_ FPR constraints. | `0.01` |
| `constraint_fnr_tolerance` | The slack when fulfilling _group-wise_ FNR constraints. | `0.01` |
| `global_target_fpr` | Target rate for the _global_ FPR (inequality) constraint. | _None_ |
| `global_target_fnr` | Target rate for the _global_ FNR (inequality) constraint. | _None_ |
| `constraint_stepwise_proxy` | Differentiable proxy for the step-wise function in _group-wise_ constraints. | `cross_entropy` |
| `objective_stepwise_proxy` | Differentiable proxy for the step-wise function in _global_ constraints. | `cross_entropy` |
| `stepwise_proxy_margin` | Intercept value for the proxy function: value at `f(logodds=0.0)` | `1.0` |
| `score_threshold` | Score threshold used when assessing _group-wise_ FPR or FNR in training. | `0.5` |
| `global_score_threshold` | Score threshold used when assessing _global_ FPR or FNR in training. | `0.5` |
| `init_multipliers` | The initial value of the Lagrange multipliers. | `0` for each constraint |
| ... | _Any core [`LGBMClassifier` parameter](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm-lgbmclassifier) can be used with FairGBM as well._ |  |

Please consult [this list](https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters) for a detailed
view of all vanilla LightGBM parameters (_e.g._, `n_estimators`, `n_jobs`, ...).

> **Note** 
> The `objective` is the only core LightGBM parameter that cannot be changed when using FairGBM, as you must use
> the constrained loss function `objective="constrained_cross_entropy"`.
> Using a standard non-constrained objective will fallback to using standard LightGBM.


### _fit(X, Y, constraint_group=S)_

In addition to the usual `fit` arguments, features `X` and labels `Y`, FairGBM takes in the sensitive attributes `S`
column for training.

**Regarding the sensitive attributes column `S`:**
- It should be in numeric format, and have each different protected group take a different integer value, starting at `0`.
- It is not restricted to binary sensitive attributes: you can use _two or more_ different groups encoded in the same column;
- It is only required for training and **not** for computing predictions;

Here is an example pre-processing for the sensitive attributes on the UCI Adult dataset:
```python
# Given X, Y, S
X, Y, S = load_dataset()

# The sensitive attributes S must be in numeric format
S = [1. if val == "Female" else 0. for val in S]

# The labels Y must be binary and in numeric format: {0, 1}
Y = [1. if val == ">50K" else 0. for val in Y]

# And the features X may be numeric or categorical, but make sure categorical columns are in the correct format
X: Union[pd.DataFrame, np.ndarray]      # any array-like can be used

# Train FairGBM
fairgbm_clf.fit(X, Y, constraint_group=S)
```


## Features

FairGBM enables you to train a GBM model to **minimize a loss function** (_e.g._, cross-entropy) **subject to fairness
constraints** (_e.g._, equal opportunity).

Namely, you can target equality of performance metrics (FPR, FNR, or both) across instances from _two or more_ different
protected groups (see [fairness constraints](#fairness-constraints) section).
Simultaneously (and optionally), you can add global constraints on specific metrics (see [global constraints](#global-constraints) section).

### Fairness constraints

You can use FairGBM to equalize the following metrics across _two or more_ protected groups:
- Equalize FNR (equivalent to equalizing TPR or Recall)
    - also known as _equal opportunity_ [(Hardt _et al._, 2016)](https://arxiv.org/abs/1610.02413)
- Equalize FPR (equivalent to equalizing TNR or Specificity)
    - also known as _predictive equality_ [(Corbett-Davies _et al._, 2017)](https://arxiv.org/abs/1701.08230)
- Equalize both FNR and FPR simultaneously
    - also known as _equal odds_ [(Hardt _et al._, 2016)](https://arxiv.org/abs/1610.02413)

> **Example for _equality of opportunity_** in college admissions:
> your likelihood of getting admitted to a certain college (predicted positive) given that you're a qualified candidate
> (label positive) should be the same regardless of your race (sensitive attribute).

<!--
Take the following hypothetical example:

If you're training an algorithm to predict mortgage defaults, a valuable fairness criterion may be equalizing FPR 
among people from different ethnicities.
This ensures that for two people that will successfully repay their loans, their likelihood of being wrongly denied
access to credit is the same regardless of ethnicity.
This is known as a _punitive_ setting, as a positive prediction (predicted to default) leads to a negative outcome
(loan application denied).

Conversely, if you're training an ML model in an _assistive_ setting (_i.e._, a positive prediction leads to a 
positive outcome for the person), you may want to target equalizing FNR.
-->

### Global constraints

You can also target specific FNR or FPR goals.
For example, in cases where high accuracy is trivially achieved (_e.g._, problems with high class imbalance),
you may want to maximize TPR with a constraint on FPR (_e.g._, "maximize TPR with at most 5% FPR").
You can set a constraint on global FPR ≤ 0.05 by using `global_target_fpr=0.05` and 
`global_constraint_type="FPR"`.

You can simultaneously set constraints on group-wise metrics (fairness constraints) and constraints on global metrics.
<!-- TODO! [This notebook](/examples/FairGBM-python-notebooks) shows an example on a highly class imbalanced dataset that makes use of both group-level and global constraints. -->


## Technical Details

FairGBM is a framework that enables _constrained optimization_ of Gradient Boosting Machines (GBMs).
This way, we can train a GBM model to minimize some loss function (usually the _binary cross-entropy_) subject to a set
of constraints that should be met in the training dataset (_e.g._, equality of opportunity).

FairGBM applies the [method of Lagrange multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier), and uses 
iterative and interleaved steps of gradient descent (on the function space, by adding new trees to the GBM model) and 
gradient ascent (on the space of Lagrange multipliers, **Λ**).

The main obstacle with enforcing fairness constraints in training is that these constraints are often 
_non-differentiable_. To side-step this issue, we use a differentiable proxy of the step-wise function.
The following plot shows an example of _hinge-based_ and _cross-entropy-based_ proxies for the _false positive_ value
of a _label negative_ instance.

<p align="center">
    <img src="https://user-images.githubusercontent.com/13498941/189664020-70ebbae4-6b93-4f38-af7d-f870381a8a22.png" width="40%" alt="example of proxy FPR function" />
</p>

For a more in-depth explanation of FairGBM please consult [the paper](https://arxiv.org/pdf/2209.07850.pdf).

[comment]: <> (### Important C++ source files **TODO**)


[comment]: <> (## Results)
[comment]: <> (%% TODO: results and run-time comparisons against fairlearn, TFCO, and others)


## Contact

For commercial uses of FairGBM please contact <oss-licenses@feedzai.com>.

## Citing FairGBM

The paper is publicly available at this [arXiv link](https://arxiv.org/abs/2209.07850).

```
@misc{cruz2022fairgbm,
  doi = {10.48550/ARXIV.2209.07850},
  url = {https://arxiv.org/abs/2209.07850},
  author = {Cruz, Andr{\'{e}} F and Bel{\'{e}}m, Catarina and Bravo, Jo{\~{a}}o and Saleiro, Pedro and Bizarro, Pedro},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Computers and Society (cs.CY), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {FairGBM: Gradient Boosting with Fairness Constraints},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
