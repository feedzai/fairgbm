# FairGBM Python Examples

In this folder you can find several FairGBM usage examples.

| _File name_             | _Dataset (size)_ | _Description_ | _Fairness criterion_ |
|-------------------------|------------------|---------------|----------------------|
| [FairGBM_example_for_equalized_odds_[google_colab].ipynb](FairGBM_example_for_equalized_odds_[google_colab].ipynb) ([colab link](https://colab.research.google.com/github/AndreFCruz/fairgbm-fork/blob/add-colab-example/examples/FairGBM-python-notebooks/FairGBM_example_for_equalized_odds_%5Bgoogle_colab%5D.ipynb))| [UCI Adult](https://archive.ics.uci.edu/ml/datasets/adult) (49K) | Simple [*Colab* example](https://colab.research.google.com/github/AndreFCruz/fairgbm-fork/blob/add-colab-example/examples/FairGBM-python-notebooks/FairGBM_example_for_equalized_odds_%5Bgoogle_colab%5D.ipynb) to get up and running with FairGBM. | Equal TPR and FPR (_equalized odds_) |
| [UCI-Adult-example.ipynb](UCI-Adult-example.ipynb) | [UCI Adult](https://archive.ics.uci.edu/ml/datasets/adult) (49K) | Simple local example on the popular UCI Adult dataset. | Equal TPR (_equal opportunity_) |
| [UCI-Adult-example-with-hyperparameter-tuning.ipynb](UCI-Adult-example-with-hyperparameter-tuning.ipynb) | [UCI Adult](https://archive.ics.uci.edu/ml/datasets/adult) (49K) | Example with hyperparameter-tuning to obtain optimal `multiplier_learning_rate` parameter value. | Equal TPR and FPR (_equalized odds_) |

<!--
| [credit-card-fraud-example.ipynb](credit-card-fraud-example.ipynb) | [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (285K) | Highly class imbalanced dataset in which a trivial classifier achieves 99.8% accuracy. Show-cases the use of both fairness and global constraints! | Equal FPR (predictive equality) |
| [ACSIncome-example.ipynb](ACSIncome-example.ipynb) | [ACSIncome](https://github.com/zykls/folktables) (1.7M) | Large dataset for comparing run-times of popular fairness-aware algorithms. _Requires a sizeable amount of RAM memory (+32GB)._ | Equal FNR (equal opportunity) |
-->

Examples may require extra requirements listed in the `requirements.txt` file. Please run `pip install -r requirements.txt` to install them.
