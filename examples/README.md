# FairGBM Usage Examples

Just as with [vanilla LightGBM](https://github.com/microsoft/LightGBM/tree/master/examples), you can use FairGBM in multiple ways:
1. Using the Python API (**_recommended_**)
1. Using the command line (and config files)
1. Using the C API

## Using Python

**[Recommended]** Using the sklearn-style API:

```python
from fairgbm import FairGBMClassifier

# Instantiate
fairgbm_clf = FairGBMClassifier(
    "constraint_type"="FNR",    # constraint on equal group-wise TPR (equal opportunity)
    "n_estimators"=200,         # core parameters from vanilla LightGBM
    "random_state"=42,          # ...
)

# Train using features (X), labels (Y), and sensitive attributes (S)
fairgbm_clf.fit(X, Y, constraint_group=S)

# Predict
Y_test_pred = fairgbm_clf.predict_proba(X_test)[:, -1]  # Compute continuous class probabilities (recommended)
# Y_test_pred = fairgbm_clf.predict(X_test)             # Or compute discrete class predictions
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

To run FairGBM from the command line, you'll need to compile the project locally
```
git clone --recurse-submodules https://github.com/feedzai/fairgbm.git
cd fairgbm
cmake .
make -j4
```

and then call the compiled binary with your config file
```
"./lightgbm" config=your_config_file.txt
```

you can also add other parameters right from the command line
```
"./lightgbm" config=train.conf objective=constrained_cross_entropy
```

For further details see LightGBM's guide on [compiling locally](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#installation-guide) and [running from the command line](https://lightgbm.readthedocs.io/en/latest/Quick-Start.html#run-lightgbm).


## Using the C API
> Using the C API is only recommended for interoperability with other languages. 
> See [this repository](https://github.com/feedzai/feedzai-openml-java/blob/master/openml-lightgbm/lightgbm-provider/src/main/java/com/feedzai/openml/provider/lightgbm/LightGBMBinaryClassificationModelTrainer.java#L126) 
> for an example on training FairGBM/LightGBM from a Java code-base.

This is a barebones example of using FairGBM with the LightGBM C API.
For further details please consult LightGBM's [C API reference](https://lightgbm.readthedocs.io/en/latest/C-API.html).

```c
int main(int argc, char** argv) {

    // Construct dataset
    LightGBM::DatasetHandle datasetHandle;
    std::string datasetParameters = (
            "label_column=name:fraud_bool "
            "constraint_group_column=name:customer_age_category "
            "has_header=True");
    LightGBM::LGBM_DatasetCreateFromFile("examples/FairGBM/BAF-base.train", datasetParameters, &datasetHandle);
    
    // Construct GBM model
    LightGBM::BoosterHandle boosterHandle;
    std::string boosterParameters = ("objective=constrained_cross_entropy constraint_type=fpr");  // Add other parameters as needed
    LightGBM::LGBM_BoosterCreate(datasetHandle, parameters, &boosterHandle)
    
    // Train model
    int isFinished, numIterations = 100;
    for (int trainIteration = 0; trainIteration < numIterations; ++trainIteration) {
        int returnCodeLGBM = LightGBM::LGBM_BoosterUpdateOneIter(boosterHandle, &isFinished);
    }
}
```

If you're interested in looking under the proverbial C++ hood, you should start from the 
[`ConstrainedObjectiveFunction`](/include/LightGBM/constrained_objective_function.h) class.
Most FairGBM-specific classes are in the C++ namespace `LightGBM::Constrained`.


## Configuration files

FairGBM's config files functionality is forked from LightGBM.
Please consult [LightGBM's core parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters) to see how to set-up a config file.

Example FairGBM config files [here](/examples/FairGBM/train.conf) or [here](/examples/FairGBM-other/train.conf).

**Note**: Remember that to use the FairGBM classifier you must always set a `objective=constrained_cross_entropy`. This is not needed when using the Python `FairGBMClassifier` class as it's 
already taken care of.
