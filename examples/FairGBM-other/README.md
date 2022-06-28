Binary Classification with Fairness Constraints
===============================================
> Example of training a binary classifier with fairness constraints.

Dataset source: https://github.com/propublica/compas-analysis/

***You must follow the [installation instructions](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)
for the following commands to work. The `lightgbm` binary must be built and available at the root of this project.***


Training
--------
Run the following command in this folder to train **FairGBM**:

```bash
"../../lightgbm" config=train.conf
```

To train the vanilla LightGBM on the same data use:
```bash
"../../lightgbm" config=train_unconstrained.conf
```

Prediction
----------

You should finish training first.

Run the following command in this folder to compute test predictions for **FairGBM**:

```bash
"../../lightgbm" config=predict.conf
```

To compute test predictions for the vanilla LightGBM use:
```bash
"../../lightgbm" config=predict_unconstrained.conf
```
