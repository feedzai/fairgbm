# FairGBM

> **Under Construction**; release date: 16th of September.

FairGBM is an easy-to-use and lightweight fairness-aware ML algorithm.
Specifically, it enables **constrained optimization** with Gradient Boosting models.

FairGBM builds upon the popular [LightGBM](https://github.com/microsoft/LightGBM) algorithm and adds customizable
constraints for group-wise fairness (_e.g._, equal opportunity, predictive equality) and other global goals (_e.g._,
specific Recall or FPR prediction targets).

## Install

FairGBM can be installed from [PyPI](https://pypi.org/project/fairgbm/) <!-- TODO: FairGBM pypi link -->

```pip install fairgbm```

or from GitHub

```
git clone --recurse-submodules https://github.com/feedzai/fairgbm.git
pip install fairgbm/python-package
```

## Examples

A short Python code snippet:
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

# Predict
Y_test_pred = fairgbm_clf.predict_proba(X_test)[:, -1]  # Compute continuous class probabilities (recommended)
# Y_test_pred = fairgbm_clf.predict(X_test)             # Or compute discrete class predictions
```

A more in-depth explanation and other usage examples can be found in the [**_examples folder_**](/examples).

For Python examples see the [**_notebooks folder_**](/examples/FairGBM-python-notebooks).


## Features

FairGBM enables you to train a GBM model to **minimize a loss function** (_e.g._, cross-entropy) **subject to fairness 
constraints** (_e.g._, equal opportunity).

Namely, you can target equality of performance metrics (FPR, FNR, or both) across instances from different protected groups (see [fairness constraints](#fairness-constraints) section).
Simultaneously (and optionally), you can add global constraints on specific metrics (see [global constraints](#global-constraints) section).

### Parameter list

List of FairGBM-specific parameters:
**TODO**: enumerate parameters and document each one;

| _Name_ | _Description_ | _Default_ |
|:------:|---------------|:---------:|
| `groupwise_constraint_type` | The type of fairness (group-wise equality) constraint to use (if any). | `FPR,FNR` |
| `global_constraint_type` | The type of global equality constraint to use (if any). | _None_ |
| ... | **TODO** |

Besides these FairGBM-specific parameters, you can use any core LightGBM parameter as well (_e.g._, 
`n_estimators`). The only exception is the `objective` parameter, as FairGBM only uses 
`objective="constrained_cross_entropy"` (a standard non-constrained objective will fallback to using standard LightGBM).

Please consult [this list](https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters) for a detailed 
view of all vanilla LightGBM parameters.


### Fairness constraints

You can use FairGBM to equalize the following metrics across protected groups:
- Equalize FNR (equivalent to equalizing TPR or Recall)
    - also known as _equal opportunity_ [(Hardt et al., 2016)](#external-references)
- Equalize FPR (equivalent to equalizing TNR or Specificity)
    - also known as _predictive equality_ [(Corbett-Davies et al., 2017)](#external-references)
- Equalize both FNR and FPR simultaneously
    - also known as _equal odds_ [(Hardt et al., 2016)](#external-references)

> Example for _equality of opportunity_ in college admissions:
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

You can also target specific TPR or FPR goals.
For example, in cases where high accuracy is trivially achieved (_e.g._, problems with high class imbalance),
you may want to maximize TPR with a constraint on FPR (_e.g._, "maximize TPR with at most 5% FPR").
You can achieve this by setting a constraint on `global FPR ≤ 0.05`.

You can simultaneously set constraints on group-wise metrics (fairness constraints) and constraints on global metrics.
[This notebook](/examples/FairGBM-python-notebooks) shows an example on a highly class imbalanced dataset that makes use of both group-level and global constraints.


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

For a more in-depth explanation of FairGBM please consult [the paper](#citing-fairgbm).


## Results

%% TODO: results and run-time comparisons against fairlearn, TFCO, and others


## Citing FairGBM

The paper is publicly available at this [arXiv link](https://arxiv.org/abs/2103.12715). _**TODO: update with correct link and reference**_

```
@article{cruz2022fairgbm,
  title={FairGBM: Gradient Boosting with Fairness Constraints},
  author={Cruz, Andr{\'e} F and Bel{\'e}m, Catarina and Bravo, Jo\~{a}o and Saleiro, Pedro and Bizarro, Pedro},
  journal={},  %%TBD
  year={2022}
}
```

### External References

[comment]: <> (FairGBM)
_Cruz, André, Catarina Belém, João Bravo, Pedro Saleiro, and Pedro Bizarro. "FairGBM: Gradient Boosting with Fairness Constraints." 2022._

[comment]: <> (Account-opening fraud dataset)
_Jesus, Sérgio, José Pombal, Duarte Alves, André Cruz, Pedro Saleiro, Rita P. Ribeiro, João Gama, and Pedro Bizarro. "bank-account-fraud: Tabular Dataset(s) for Fraud Detection under Multiple Bias Conditions." 2022._

[comment]: <> (Fairlearn)
_Agarwal, Alekh, Alina Beygelzimer, Miroslav Dudík, John Langford, and Hanna Wallach. "A reductions approach to fair classification." ICML, 2018._

[comment]: <> (TensorFlow constrained optimization)
_Cotter, Andrew, Heinrich Jiang, Maya R. Gupta, Serena Wang, Taman Narayan, Seungil You, and Karthik Sridharan. "Optimization with Non-Differentiable Constraints with Applications to Fairness, Recall, Churn, and Other Goals." JMLR, 2019._

[comment]: <> (LightGBM)
_Ke, Guolin, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. "Lightgbm: A highly efficient gradient boosting decision tree." NeurIPS, 2017._

[comment]: <> (Equality of opportunity)
_Hardt, Moritz, Eric Price, and Nati Srebro. "Equality of opportunity in supervised learning." NeurIPS, 2016._
