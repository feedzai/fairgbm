# FairGBM Python Examples

In this folder you can find several FairGBM usage examples.

| _File name_             | _Dataset (size)_ | _Description_ | _Fairness criterion_ |
|-------------------------|------------------|---------------|----------------------|
| [UCI-Adult-example.ipynb](UCI-Adult-example.ipynb) | [UCI Adult](https://archive.ics.uci.edu/ml/datasets/adult) (49K) | Simple example on the popular UCI Adult dataset. | Equal FNR and TPR (equal odds) |
| [credit-card-fraud-example.ipynb](credit-card-fraud-example.ipynb) | [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (285K) | Highly class imbalanced dataset in which a trivial classifier achieves 99.8% accuracy. Show-cases the use of both fairness and global constraints! | Equal FPR (predictive equality) |
| [ACSIncome-example.ipynb](ACSIncome-example.ipynb) | [ACSIncome](https://github.com/zykls/folktables) (1.7M) | Large dataset for comparing run-times of popular fairness-aware algorithms. _Requires a sizeable amount of RAM memory (+32GB)._ | Equal FNR (equal opportunity) |

Running `pip install -r requirements.txt` installs all requirements for all examples.
You can also install each package manually as visible in the beginning of each example.
