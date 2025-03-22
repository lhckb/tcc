import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import colorsys
from prettytable import PrettyTable
import numpy as np

def print_line_series(df: pd.DataFrame, title: str = "Series"):
    plt.figure(figsize=(15, 5))
    plt.title(title)
    plt.plot(df.Price, label="Preço (R$)")
    plt.legend()

def adjust_saturation(color, saturation_factor=0.5):
    """Modify the saturation of an RGB color."""
    rgb = mcolors.to_rgb(color)  # Convert to RGB
    h, l, s = colorsys.rgb_to_hls(*rgb)  # Convert to HLS
    new_rgb = colorsys.hls_to_rgb(h, l, s * saturation_factor)  # Modify saturation
    return new_rgb

sat = 0.45
model_colors = {
    "LSTM": adjust_saturation("blue", sat),
    "GRU": adjust_saturation("orange", sat),
    "RNN": adjust_saturation("green", sat),
    "RF": adjust_saturation("red", sat),
    "RG": adjust_saturation("purple", sat),
    "B&H": adjust_saturation("grey", sat),
    "Ensemble": adjust_saturation("yellow", sat),
}

def plot_many_returns_series(series, labels, index, colors, title="Portfolio Comparison between Buy & Hold and Model"):
    if len(series) > 4:
        raise Exception("Cannot handle more than 4 series")
    
    plt.figure(figsize=(25, 10))
    for i_pos, cuml_series in enumerate(series):
        plt.plot(index, cuml_series, label = labels[i_pos], linestyle='-', color=colors[i_pos])
    plt.title(title)
    plt.legend()

    for i_pos, cuml_series in enumerate(series):
        max_idx = -1

        plt.annotate(
            f"{cuml_series.to_numpy()[max_idx]}",
            (cuml_series.index[max_idx], cuml_series.to_numpy()[max_idx]),
            xytext=(cuml_series.index[max_idx], cuml_series.to_numpy()[max_idx] + 0.5),
            arrowprops=dict(facecolor=colors[i_pos], arrowstyle='->'),
            color=colors[i_pos],
        )

    plt.plot()

def plot_distribution_and_stddev(all_data, metric, models=["LSTM", "GRU", "RNN", "RF", "RG"]):
    fig, axes = plt.subplots(1, 2, figsize=(25, 10))

    labels = [models[i] for i in range(len(all_data))]

    # --- Boxplot ---
    box = axes[0].boxplot(
        all_data, 
        labels=labels,
        patch_artist=True  # Enable box fill colors
    )

    # Apply colors to boxplot
    for patch, model in zip(box['boxes'], labels):
        patch.set_facecolor(model_colors.get(model, "black"))  # Use black if model is not in dictionary

    axes[0].set_title(f'Boxplot of {metric}')

    # --- Bar Chart ---
    std_devs = [np.array(model).std() for model in all_data]
    bars = axes[1].bar(
        labels,
        std_devs, 
        color=[model_colors.get(model, "black") for model in labels]
    )

    for bar in bars:
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.4f}',
            ha='center', va='bottom'
        )

    axes[1].set_title(f'Standard Deviations of {metric}')

    plt.tight_layout()
    plt.show()

def print_statistics_table_for_series(all_data, models = ["LSTM", "GRU", "RNN", "RF", "RG"]):
    table = PrettyTable(["model", "mean", "median", "max", "min", "stddev", "full range"])

    for series, model in zip(all_data, models):
        table.add_row([
            model,
            np.array(series).mean(),
            np.median(np.array(series)),
            np.array(series).max(),
            np.array(series).min(),
            np.array(series).std(),
            np.array(series).max() - np.array(series).min()
        ])
    
    print(table)

def print_class_distributions_for_experiment(experiments, best_idx, model, stocks):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    fig.suptitle(f"Class distributions for best {model} experiment")

    for i, col in enumerate(stocks):
        best_pred_series = pd.Series(list(item.values())[0] for item in experiments[best_idx][col][1])
        bars = axes[i].bar(["False", "True"], best_pred_series.value_counts())
        for bar in bars:
            height = bar.get_height()
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height}',
                ha='center', va='bottom'
            )
        axes[i].set_title(f"class distribution for {col.upper()}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

def plot_mean_and_stddev(means, std_devs, metric, models=["LSTM", "GRU", "RNN", "RF", "RG", "Ensemble"]):
    fig, axes = plt.subplots(1, 2, figsize=(25, 10))

    labels = [models[i] for i in range(len(means))]

    # --- Mean Bar Chart ---
    bars1 = axes[0].bar(
        labels, 
        means, 
        color=[model_colors.get(model, "gray") for model in labels]
    )
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.4f}',
            ha='center', va='bottom'
        )

    axes[0].set_title(f'Mean of {metric}')

    # --- Std Dev Bar Chart ---
    bars = axes[1].bar(
        labels,
        std_devs, 
        color=[model_colors.get(model, "gray") for model in labels]  # Apply consistent colors
    )

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.4f}',
            ha='center', va='bottom'
        )

    axes[1].set_title(f'Standard Deviations of {metric}')

    plt.tight_layout()
    plt.show()