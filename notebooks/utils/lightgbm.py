import lightgbm as lgb


def load_lightgbm_model(path: str):
    with open(path, "r") as file_in:
        model_str = file_in.read()

    # Replace all occurrences of "constrained_cross_entropy" with "cross_entropy"
    model_str = model_str.replace("constrained_cross_entropy", "cross_entropy")

    # Replace all occurrences of "constrained_recall_objective" with "cross_entropy"    -- should be fine :)
    model_str = model_str.replace("constrained_recall_objective", "cross_entropy")

    return lgb.Booster(model_str=model_str)
