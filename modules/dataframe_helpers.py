import pandas as pd
from modules.financial_helpers import compute_cumulative_returns_pred, compute_sharpe_ratio

def count_largest_missing_interval(df: pd.DataFrame, column: str):
    # Step 1: Create a mask for missing values in the specified column.
    mask = df[column].isna()

    # Step 2: Identify consecutive groups of NaNs using cumsum on the inverted mask.
    # Use (~mask).cumsum() to create a unique group number for consecutive NaNs.
    # Cumsum on the inverted mask will assign an increasing integer to each entry that is not NaN,
    # and repeat the last integer for each NaN entry.
    groups = (~mask).cumsum()

    # Step 3: Filter only the groups corresponding to NaN values.
    # Grouping by the assigned integers will create several unitary groups for not NaN entries,
    # and the NaN ones, that repeat the same integer, will be grouped together in sequence.
    nan_groups = df[mask].groupby(groups).size()

    # Step 4: Find the maximum size of consecutive NaNs. If a group has size > 1 it means it's
    # composed of NaNs. So the largest sequence is the max number of NaNs in sequence.
    max_nan_sequence_length = nan_groups.max() if not nan_groups.empty else 0

    # Step 5: Get the index range for this sequence if it exists.
    if max_nan_sequence_length > 0:
        longest_nan_group = nan_groups.idxmax()
        start_date = df[mask & (groups == longest_nan_group)].index.min()
        end_date = df[mask & (groups == longest_nan_group)].index.max()
        print(f"The length of the largest consecutive sequence of missing values in {column} is: {max_nan_sequence_length}")
        print(f"The time range for this sequence is from {start_date} to {end_date}")
    else:
        print(f"No missing values found in the {column} time series.")
    print("\n")

    return

def compute_returns_for_prices(df: pd.DataFrame, column_names: list[str]):
    df = df.copy()
    for column in column_names:
        new_name = column.split("_")[1] if "_" in column else column
        df[f"Return_{new_name}"] = (df[column] / df[column].shift(1)) - 1
        df.dropna(inplace=True)
    return df

def compute_lags(df: pd.DataFrame, column_names: list[str], lags: int = 5):
    df = df.copy()
    
    for col in column_names:
        if "Return" in col:
            for i in range(lags - 1):
                df[f"{col}_lag_{i+1}"] = df[f"{col}"].shift(i+1)
    return df

def compute_targets(df: pd.DataFrame, stock_names: list[str]):
    df = df.copy()

    for col in stock_names:
        df[f"{col}_y"] = df[f"Return_{col}"].shift(-1) > 0
    return df

def preprocess_data(data: pd.DataFrame, stock_name: str, start_idx: int, end_idx: int) -> pd.DataFrame:
    """
    Preprocesses data for a given stock, ensuring that the last 5 days before the rolling window are included 
    for computing lags.
    
    Parameters:
        - data: The full DataFrame containing stock data.
        - stock_name: The stock name to filter columns.
        - start_idx: The starting index of the rolling window.
        - end_idx: The ending index of the rolling window.
    
    Returns:
        - Processed DataFrame containing only the rolling window data, but with correctly computed lags.
    """
    data = data.copy()
    
    # Ensure we can access the previous 5 days
    start_idx_with_lags = max(0, start_idx - 5)
    
    # Extract the relevant data (prior 5 days + current window)
    data_window = data.iloc[start_idx_with_lags:end_idx+1].copy()
    
    # Interpolation
    data_window = data_window.interpolate(method="linear")
    
    # Compute returns, lags, and targets
    data_window = compute_returns_for_prices(data_window, [f"Price_{stock_name}"])
    data_window = compute_lags(data_window, [f"Return_{stock_name}"])
    data_window = compute_targets(data_window, [stock_name])
    
    # Select relevant columns
    data_window = data_window[[col for col in data_window.columns.tolist() if stock_name in col or "median" in col]].copy()
    
    # Drop unnecessary columns
    data_window = data_window.drop(f"Price_{stock_name}", axis=1)
    
    # Drop NaNs, but only within the rolling window range
    # data_window = data_window.iloc[3:] if start_idx >= 3 else data_window  # Drop first 5 rows only when safe
    data_window = data_window.dropna(axis=0)
    return data_window

def preprocess_data_test(data: pd.DataFrame, stock_name: str) -> pd.DataFrame:
    data = data.copy()
    data = data.interpolate(method="linear")
    data = compute_returns_for_prices(data, [f"Price_{stock_name}"])
    data = compute_lags(data, [f"Return_{stock_name}"])
    data = compute_targets(data, [stock_name])
    data = data[[col for col in data.columns.tolist() if stock_name in col or "median" in col]].copy()
    data = data.drop(f"Price_{stock_name}", axis=1)
    data = data.dropna(axis=0)
    return data

def check_time_index_overlap(dataframes):
    # Convert each DataFrame's index to a set for faster intersection checking
    index_sets = [set(df.index) for df in dataframes]
    
    # Check each pair of DataFrames for overlap
    for i in range(len(index_sets)):
        for j in range(i + 1, len(index_sets)):
            if index_sets[i].intersection(index_sets[j]):
                print(f"Overlap found between DataFrame {i} and DataFrame {j}")
                return True

    print("No overlap found; all time indices are mutually exclusive.")
    return False

def get_sharpe_series_from_results(experiments, daily_returns):
    sharpes = []

    for experiment in experiments:
        _, exp_portfolio_returns = compute_cumulative_returns_pred(
            experiment["abev3"][1],
            experiment["bbdc3"][1],
            experiment["itsa3"][1],
            experiment["itub3"][1],
            experiment["wege3"][1],
            daily_returns
        )

        sharpe = compute_sharpe_ratio(exp_portfolio_returns)
        sharpes.append(sharpe)
        
    
    return sharpes

def get_metric_distribution_from_results(results, metric):
    metric_series = []
    for result in results:
        metric_sum = result["abev3"][0][metric] + result["bbdc3"][0][metric] + result["itub3"][0][metric] + result["itsa3"][0][metric] + result["wege3"][0][metric]
        metric_series.append(metric_sum / 5)

    return metric_series