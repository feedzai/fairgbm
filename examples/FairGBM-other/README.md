Binary Classification with Fairness Constraints
===============================================
> Example of training a binary classifier with fairness constraints.

***You must follow the [installation instructions](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)
for the following commands to work. The `lightgbm` binary must be built and available at the root of this project.***

Dataset
-------

The BAF (Bank Account Fraud) dataset is a real-world dataset on banking fraud.
More details [here](https://github.com/feedzai/bank-account-fraud).

For this example we're using a sample of 10% of the BAF-Base dataset.
Fairness is measured as the FPR ratio between people aged under 50 and those aged 50 or above (as per the dataset's
[datasheet](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf)).


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
