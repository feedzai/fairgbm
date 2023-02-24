from pathlib import Path


def get_dataset_details(dataset_name: str, local_or_cluster: str = "local") -> dict:

    # Dict to hold dataset metadata
    metadata = dict()

    # Set data paths (according to whether you're using the local machine or the cluster)
    if local_or_cluster == "local":
        root_path = Path("/home/andre.cruz/Documents/fair-boosting/")
    elif local_or_cluster == "cluster":
        root_path = Path("/mnt/home/andre.cruz/fair-boosting/")
    else:
        raise ValueError(f"Expected local_or_cluster in ('local', 'cluster'), got '{local_or_cluster}'")

    data_path = root_path / "data" / dataset_name
    experiment_path = root_path / "experiments" / dataset_name
    columns_path = data_path / "cols_order.csv"

    metadata["root_path"] = root_path
    metadata["data_path"] = data_path
    metadata["experiment_path"] = experiment_path
    metadata["columns_path"] = columns_path

    # AOF (FairHO version) data paths
    if dataset_name == "AOF-FairHO":
        metadata["train_data_path"] = data_path / "pre-processed_train.header.csv"
        metadata["val_data_path"] = data_path / "pre-processed_validation.header.csv"
        metadata["test_data_path"] = data_path / "pre-processed_test.header.csv"

        metadata["label_col"] = "fraud_bool"
        metadata["sensitive_col"] = "age-group"

        # Target metric for evaluation
        metadata["target_metric"] = "fpr"
        metadata["target_value"] = 0.05
        
        # Performance metric to optimize
        metadata["perf_metric"] = "tpr"
        
        # Fairness metric
        metadata["fair_metric"] = "fpr_ratio"


    elif dataset_name == "AOF-Fairbench":
        metadata["train_data_path"] = data_path / "candidate_random_sample_handpicked_1_train_sanitized_one_hot.processed-for-lightgbm-cpp.TRAIN.csv"
        metadata["val_data_path"] = data_path / "candidate_random_sample_handpicked_1_train_sanitized_one_hot.processed-for-lightgbm-cpp.VAL.csv"
        metadata["test_data_path"] = data_path / "candidate_random_sample_handpicked_1_val_sanitized_one_hot.processed-for-lightgbm-cpp.TEST.csv"

        metadata["label_col"] = "fraud_bool"
        metadata["sensitive_col"] = "age-group"
        
        # Target metric for evaluation
        metadata["target_metric"] = "fpr"
        metadata["target_value"] = 0.05
        
        # Performance metric to optimize
        metadata["perf_metric"] = "tpr"
        
        # Fairness metric
        metadata["fair_metric"] = "fpr_ratio"


    elif dataset_name == "Adult-2021":
        metadata["train_data_path"] = data_path / "ACSIncome.train.header.csv"
        metadata["val_data_path"] = data_path / "ACSIncome.validation.header.csv"
        metadata["test_data_path"] = data_path / "ACSIncome.test.header.csv"

        metadata["label_col"] = "PINCP"
        metadata["sensitive_col"] = "SEX"

        # Target metric for evaluation
        metadata["target_metric"] = "threshold"
        metadata["target_value"] = 0.5

        # Performance metric to optimize
        metadata["perf_metric"] = "accuracy"

        # Fairness metric
        metadata["fair_metric"] = "fnr_ratio"    


    else:
        raise ValueError(f"Not configured for this dataset: {dataset_name}")
        
    print(f"Using dataset {dataset_name}!")
    return metadata
