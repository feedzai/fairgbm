# FairGBM

> _Official release date: **16th of September**._

FairGBM is an easy-to-use and lightweight fairness-aware ML algorithm.

FairGBM builds upon the popular [LightGBM](https://github.com/microsoft/LightGBM) algorithm and adds customizable
constraints for group-wise fairness (_e.g._, equal opportunity, predictive equality) and other global goals (_e.g._,
specific Recall or FPR targets).

## Install

FairGBM can be installed from [PyPI](https://pypi.org/project/shap/)

```pip install fairgbm```

or from GitHub

```
git clone --recurse-submodules https://github.com/feedzai/fairgbm.git
pip install fairgbm/python-package
```

## Examples

%% TODO: short code snippets, plus enumerate and link to `examples` folder,
and code-up new python notebooks (current examples are mostly in C++ configs)

A more in-depth explanation and other usage examples can be found in the [examples folder](/examples).

## Technical Details

%% TODO: explain FairGBM technical details:
1. Lagrange method of multipliers
2. Gradient ascent/descent
3. Proxy losses (w/ python plot)
4. Re-iterate that the C++ code-base is made to have very high performance


## Results

%% TODO: results and run-time comparisons against fairlearn, TFCO, and others


## Citing FairGBM

```
@article{cruz2022fairgbm,
  title={FairGBM: Gradient Boosting with Fairness Constraints},
  author={Cruz, Andr{\'e} F and Bel{\'e}m, Catarina and Bravo, Jo\~{a}o and Saleiro, Pedro and Bizarro, Pedro},
  journal={},  %%TBD
  year={2022}
}
```
