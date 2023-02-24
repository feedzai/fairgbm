# Supplementary Materials for ICLR 2023

> These are the supplementary materials for the ICLR 2023 paper titled ["FairGBM: Gradient Boosting with Fairness Constraints"](https://arxiv.org/abs/2209.07850).

See the paper appendix for further details on the folder structure.

### Key files

FairGBM is based on iterative gradient descent (in function space) and ascent (w.r.t. Lagrange multipliers) steps of the proxy-Lagrangian and the Lagrangian respectively.

The main files to look out for are:
- the `ConstrainedCrossEntropy` class implemented at `code/src/objective/constrained_xentropy_objective.hpp`;
- and its superclass `ConstrainedObjectiveFunction` at `code/include/LightGBM/constrained_objective_function.h`:
- as well as all code under the C++ namespace `LightGBM::Constrained`.

### Trying it out

- [Recommended for **Linux** users] Follow the install instructions for the `fairgbm` python package on the [`main-fairgbm` branch](https://github.com/feedzai/fairgbm).
- [Recommended for **non-Linux** users] Or use the file `code/CMakeLists.txt` to compile the FairGBM project from source.
