import pandas as pd


def load_data(data_path: str, columns_order_path: str, index_col: int = 0, **kwargs) -> pd.DataFrame:
    col_names = pd.read_csv(columns_order_path).columns.to_list()
    return pd.read_csv(data_path, sep='\t', header=None, names=col_names, index_col=index_col, **kwargs)
