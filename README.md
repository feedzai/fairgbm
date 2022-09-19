# FairGBM supplementary materials

See the paper appendix for further details on the folder structure.

### Key files

FairGBM is based on iterative gradient descent (in function space) and ascent (w.r.t. Lagrange multipliers) steps of the proxy-Lagrangian and the Lagrangian respectively.

The main files to look out for are:
- the `ConstrainedCrossEntropy` class implemented at `code/src/objective/constrained_xentropy_objective.hpp`;
- and its superclass `ConstrainedObjectiveFunction` at `code/include/LightGBM/constrained_objective_function.h`:
- as well as all code under the C++ namespace `LightGBM::Constrained`.

### Trying it out

- Follow the install instructions for the `fairgbm` python package in the `code` folder.
- Or use the file `code/CMakeLists.txt` to compile the FairGBM project from scratch.
  - The ACSIncome-Adult dataset can be obtained via the notebook at `notebooks/data.Adult.folktables.ipynb`.
  - You can adapt the config files under `hyperparameters/sampled-hyperparameters` to your local filesystem.
