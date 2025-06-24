import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 300
plt.rcParams["boxplot.medianprops.color"] = "black"
import pandas as pd
import matplotlib.colors as mcolors
import colorsys
from prettytable import PrettyTable
import numpy as np
import matplotlib.patheffects as pe

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

sat = 1

# blue_shades = ["#F0F8FF", "#6CB4EE", "#0066b2", "#00308F"]
blue_shades = ["#B9F3FC", "#AEE2FF", "#93C6E7", "#578FCA"]

model_colors = {
    "LSTM": adjust_saturation(blue_shades[1], sat),
    "GRU": adjust_saturation(blue_shades[2], sat),
    "RNR": adjust_saturation(blue_shades[3], sat),
    "Comitê": adjust_saturation(blue_shades[0], sat),
    "FA": adjust_saturation("#C8FFD4", sat),
    "MA": adjust_saturation("#B1AFFF", sat),
    "B&H": adjust_saturation("grey", sat),
}

# def plot_many_returns_series(series, labels, index, colors, title="Portfolio Comparison between Buy & Hold and Model"):
#     if len(series) > 6:
#         raise Exception("Cannot handle more than 6 series")
    
#     plt.figure(figsize=(15, 7.5))
#     for i_pos, cuml_series in enumerate(series):
#         line = plt.plot(index, cuml_series, label = labels[i_pos], linestyle='-', color=colors[i_pos])
#         line[0].set_path_effects([pe.Stroke(linewidth=1.75, foreground='black'), pe.Normal()])
#     plt.title(title)
#     # plt.grid(True, axis='y', linestyle="--", linewidth=0.4, alpha=0.7, zorder=1)
#     plt.legend(fontsize=12, loc="upper left", facecolor="white", framealpha=1)

#     for i_pos, cuml_series in enumerate(series):
#         max_idx = -1
#         max_value = cuml_series.to_numpy()[max_idx]
#         max_x = cuml_series.index[max_idx]
        
#         # Traçar linha pontilhada
#         plt.axhline(y=max_value, color=colors[i_pos], linestyle='dotted', linewidth=1, zorder=0)
        
#         plt.annotate(
#             f"{max_value[0]:.4f}",
#             xy=(max_x, max_value),
#             xytext=(max_x + (index[-1] - index[0]) * 0.02, max_value),  # Deslocamento para a direita
#             ha='left', va='center',
#             color=colors[i_pos],
#             bbox=dict(facecolor='white', edgecolor=colors[i_pos], boxstyle='round,pad=0.3'),
#             path_effects=[pe.withStroke(linewidth=1.15, foreground="black")]
#         )

#     plt.plot()

def plot_many_returns_series(series, labels, index, colors, title="Portfolio Comparison between Buy & Hold and Model"):
    fig, ax = plt.subplots(figsize=(15, 7.5))
    
    for i_pos, cuml_series in enumerate(series):
        line = ax.plot(index, cuml_series, label=labels[i_pos], linestyle='-', color=colors[i_pos])
        line[0].set_path_effects([pe.Stroke(linewidth=1.7, foreground='grey'), pe.Normal()])

    ax.set_title(title)
    ax.legend(fontsize=12, loc="upper left", facecolor="white", framealpha=1)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.grid(True, axis='y', linestyle="--", linewidth=0.4, alpha=0.7, zorder=1)

    max_values = []

    for i_pos, cuml_series in enumerate(series):
        max_idx = -1
        max_value = cuml_series.to_numpy()[max_idx]
        max_x = cuml_series.index[max_idx]
        
        # ax.axhline(y=max_value, color=colors[i_pos], linestyle='dotted', linewidth=1, zorder=0)

        max_values.append(max_value[0])
    
    plt.margins(0)
    plt.show()

def plot_distribution_and_stddev(all_data, metric, models=["LSTM", "GRU", "RNR", "FA", "MA"], legend_loc="best", legend_size=12, benchmark_value=None, benchmark_color="red", benchmark_name="Benchmark"):
    plt.figure(figsize=(15, 10))

    labels = [models[i] for i in range(len(all_data))]

    # --- Boxplot ---
    box = plt.boxplot(
        all_data, 
        labels=labels,
        patch_artist=True  # Enable box fill colors
    )

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Apply colors to boxplot
    for patch, model in zip(box['boxes'], labels):
        patch.set_facecolor(model_colors.get(model, "black"))  # Use black if model is not in dictionary

    # Plot benchmark line if provided
    if benchmark_value is not None:
        plt.axhline(y=benchmark_value, color=benchmark_color, linestyle='--', linewidth=1.5, zorder=2)
        plt.text(
            x=0.5, 
            y=benchmark_value + 0.001, 
            s=benchmark_name, 
            color=benchmark_color, 
            fontsize=12, 
            ha='left', 
            va='bottom', 
            transform=ax.get_yaxis_transform()
        )

    plt.title(f'Boxplot de {metric}', fontsize=legend_size)
    plt.grid(True, axis='y', linestyle="--", linewidth=0.4, alpha=0.7, zorder=1)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=model_colors[model], markersize=legend_size, label=model) for model in models
    ]
    # plt.legend(handles=handles, loc=legend_loc, fontsize=legend_size)
    plt.xticks(range(1, len(labels) + 1), labels, fontsize=legend_size)
    plt.yticks(fontsize=legend_size)

    plt.tight_layout()
    plt.show()

def print_statistics_table_for_series(all_data, models = ["LSTM", "GRU", "RNR", "FA", "MA"]):
    table = PrettyTable(["modelo", "média", "mediana", "max", "min", "desvio padrão", "range"])

    for series, model in zip(all_data, models):
        table.add_row([
            model,
            f"{np.array(series).mean():.4f}",
            f"{np.median(np.array(series)):.4f}",
            f"{np.array(series).max():.4f}",
            f"{np.array(series).min():.4f}",
            f"{np.array(series).std():.4f}",
            f"{(np.array(series).max() - np.array(series).min()):.4f}"
        ])
    
    print(table)

def print_class_distributions_for_experiment(experiments, best_idx, model, stocks):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    fig.suptitle(f"Distribuição de classes para {model}")

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
        axes[i].set_title(f"Distribuição de classes para {col.upper()}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

def plot_mean_and_stddev(means, std_devs, metric, models=["Comitê", "LSTM", "GRU", "RNR", "FA", "MA"]):
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
    # axes[0].grid(True, axis='y', linestyle="--", linewidth=0.4, alpha=0.7)

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

def plot_grouped_boxplot(data, metric, stocks=["ABEV3", "BBDC3", "ITSA3", "ITUB3", "WEGE3"], 
                         models=["Comitê", "LSTM", "GRU", "RNR", "FA"], legend_size=12):
    """
    Plots a grouped boxplot where X-axis represents stocks and each stock has 5 grouped models.
    
    Parameters:
    - data: dict {stock: [list of 5 lists for models]}
    - metric: str, metric name for the title
    - stocks: list, stock names
    - models: list, model names
    - model_colors: dict, colors for each model
    - legend_size: int, font size for labels
    """

    num_stocks = len(stocks)
    num_models = len(models)
    
    # X positions for each stock, with small shifts for each model
    x_positions = np.arange(num_stocks)
    width = 0.1  # Width of each model box within a stock group
    
    plt.figure(figsize=(24, 8))

    # Plot each model's data
    for i, model in enumerate(models):
        model_data = [data[stock][i] for stock in stocks]  # Extract model-specific data across stocks
        positions = x_positions + (i - (num_models / 2)) * width  # Offset each model within the stock group
        box = plt.boxplot(model_data, positions=positions, widths=width, patch_artist=True)

        # Apply colors
        for patch in box['boxes']:
            patch.set_facecolor(model_colors[model])

    plt.xticks(x_positions, stocks, fontsize=legend_size)  # Set stock names as X-ticks
    plt.yticks(fontsize=legend_size)
    plt.grid(True, axis='y', linestyle="--", linewidth=0.5, alpha=0.7)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Create legend
    handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=model_colors[m], markersize=10, label=m) 
               for m in models]
    plt.legend(handles=handles, loc="upper left", fontsize=legend_size)

    plt.title(f'Boxplot de {metric} por Ação', fontsize=legend_size + 2)

    plt.tight_layout()
    plt.show()