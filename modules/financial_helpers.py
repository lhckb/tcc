import pandas as pd
import numpy as np

def compute_sharpe_ratio(portfolio_daily_returns):
    risk_free_daily_rate = ((1 + 0.10) ** (1 / 252)) - 1  # considering SELIC at an arbitrary avg of 10% year over 252 trading days

    avg_return = portfolio_daily_returns.mean()
    std_dev_return = portfolio_daily_returns.std(ddof=1)

    return (avg_return - risk_free_daily_rate) / std_dev_return

def compute_buyandhold_cumulative_returns(returns_series):
    # Define equal weights for each stock
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # Calculate daily portfolio returns
    # portfolio_returns = returns_series.dot(weights)
    portfolio_returns = (returns_series * weights).sum(axis=1)

    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Convert to DataFrame for plotting
    cumulative_returns_df = pd.DataFrame({'Cumulative Return': cumulative_returns}, index=returns_series.index)

    return cumulative_returns_df, portfolio_returns

def compute_cumulative_returns_pred(abev3_preds, bbdc3_preds, itsa3_preds, itub3_preds, wege3_preds, daily_returns):
    inclusion_mask = pd.DataFrame({
        'Return_abev3': pd.Series([list(val.values())[0] for val in abev3_preds], index=[list(val.keys())[0] for val in abev3_preds]),
        'Return_bbdc3': pd.Series([list(val.values())[0] for val in bbdc3_preds], index=[list(val.keys())[0] for val in bbdc3_preds]),
        'Return_itsa3': pd.Series([list(val.values())[0] for val in itsa3_preds], index=[list(val.keys())[0] for val in itsa3_preds]),
        'Return_itub3': pd.Series([list(val.values())[0] for val in itub3_preds], index=[list(val.keys())[0] for val in itub3_preds]),
        'Return_wege3': pd.Series([list(val.values())[0] for val in wege3_preds], index=[list(val.keys())[0] for val in wege3_preds]),
    }).shift(1)

    # Apply mask to daily returns
    adjusted_returns = daily_returns * inclusion_mask.shift(1)
    adjusted_returns = adjusted_returns.dropna(axis=0)

    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    # Adjust weights based on inclusion mask
    effective_weights = inclusion_mask * weights

    # Calculate daily portfolio returns with adjusted weights
    portfolio_returns = (adjusted_returns * effective_weights).sum(axis=1)

    # Step 6: Calculate cumulative returns
    cumulative_returns_pred = (1 + portfolio_returns).cumprod()

    # Convert to DataFrame for plotting
    cumulative_returns_pred_df = pd.DataFrame({'Cumulative Return': cumulative_returns_pred}, index=daily_returns.index)

    return cumulative_returns_pred_df, portfolio_returns # daily

def find_best_experiment(experiments, daily_returns):
    max_sharpe = float("-inf")
    best_results = None
    best_cuml_performance = None
    best_idx = -1

    for i, experiment in enumerate(experiments):
        cuml_pred_df, exp_portfolio_returns = compute_cumulative_returns_pred(
            experiment["abev3"][1],
            experiment["bbdc3"][1],
            experiment["itsa3"][1],
            experiment["itub3"][1],
            experiment["wege3"][1],
            daily_returns
        )

        sharpe = compute_sharpe_ratio(exp_portfolio_returns)
        if sharpe > max_sharpe:
            max_sharpe = sharpe
            best_results = experiment
            best_cuml_performance = cuml_pred_df
            best_idx = i

    print(f"Selected exp number {best_idx+1}")

    return best_results, best_cuml_performance, max_sharpe, best_idx

def get_mean_cuml_returns_across_exps(experiments, daily_returns):
    cuml_returns_across_exps = []

    for i, experiment in enumerate(experiments):
        cuml_pred_df, _ = compute_cumulative_returns_pred(
            experiment["abev3"][1],
            experiment["bbdc3"][1],
            experiment["itsa3"][1],
            experiment["itub3"][1],
            experiment["wege3"][1],
            daily_returns
        )

        cuml_returns_across_exps.append(cuml_pred_df.to_numpy()[-1])
    
    return np.mean(cuml_returns_across_exps), np.std(cuml_returns_across_exps, ddof=1)

def get_cuml_returns_dist(experiments, daily_returns):
    cuml_returns_across_exps = []

    for i, experiment in enumerate(experiments):
        cuml_pred_df, _ = compute_cumulative_returns_pred(
            experiment["abev3"][1],
            experiment["bbdc3"][1],
            experiment["itsa3"][1],
            experiment["itub3"][1],
            experiment["wege3"][1],
            daily_returns
        )

        cuml_returns_across_exps.append(cuml_pred_df.to_numpy()[-1][0])
    
    return cuml_returns_across_exps

