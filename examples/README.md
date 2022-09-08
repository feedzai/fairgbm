# FairGBM Usage Examples

Just as with [vanilla LightGBM](https://github.com/microsoft/LightGBM/tree/master/examples), you can use FairGBM in multiple ways:
1. Using the Python API
1. Using the command line (and config files)
1. Using the C API

## Using Python

**[Recommended]** Using the sklearn-style API:

```python
# TODO
```


Using the standard LightGBM API:

```python
from fairgbm import Dataset, train

# Create train dataset with features (X), labels (Y), and sensitive attributes (S)
train_set = Dataset(X, label=Y, constraint_group=S)

# Example FairGBM parameters
fgbm_params = {
    "objective": "constrained_cross_entropy",   # This objective is FairGBM's entry-point
    "constraint_type": "FNR",   # Constraint on equal group-wise TPR (equal opportunity)
    "n_estimators": 200,
    "random_state": 42,
}

# Train FairGBM
fairgbm_clf = train(params=fgbm_params, train_set=train_set)

# Compute test predictions
y_test_pred = fairgbm_clf.predict(X_test)
# NOTE! FairGBM doesn't use sensitive attributes (S_test) to predict
```


## Using the command line

> TODO: minor explainer (compile and run pointing to a given configuration file)


## Using the C API

```c
// TODO
```
