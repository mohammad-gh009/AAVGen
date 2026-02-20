import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import gaussian_kde, spearmanr
from eval_utils import *

sns.set_theme(style="whitegrid", context="talk")  #
plt.rcParams["font.family"] = "Arial"


def read_all_csvs(path2dfs: str):
    all_files = os.listdir(path2dfs)
    all_csvs = [file for file in all_files if file.endswith(".csv")]
    aavdfs = {}
    aav_type = path2dfs.split("/")[-1]
    for csv_file in all_csvs:
        file_name = csv_file.replace(".csv", "")
        file_path = os.path.join(path2dfs, csv_file)
        aavdfs[file_name] = pd.read_csv(file_path)
        aavdfs[file_name]["aav_type"] = [aav_type] * len(aavdfs[file_name])
    return aavdfs


def scatter_plot_density(df, name):

    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)
    y_pred = df["y_pred"].values.flatten()
    y_true = df["y_true"].values.flatten()

    fig, ax = plt.subplots(figsize=(8, 6))

    xy = np.vstack([y_true, y_pred])
    kde = gaussian_kde(xy)
    density = kde(xy)

    spearman_corr, p_value = spearmanr(y_true, y_pred)

    scatter = ax.scatter(
        y_true, y_pred, c=density, cmap="plasma", alpha=0.7, s=20, edgecolors="none"
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Point density", rotation=270, labelpad=20)

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        label="Perfect prediction",
        linewidth=2,
    )

    textstr = f"ρ = {spearman_corr:.3f}"  # \np = {p_value:.3e}

    # Create text with box styling matching your legend example
    props = dict(
        boxstyle="round", facecolor="white", edgecolor="black", alpha=1.0, linewidth=1
    )

    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=props,
    )

    ax.set_xlabel("True scores")
    ax.set_ylabel("Predicted scores")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    aav2dfs = read_all_csvs(r"regressoins_results")

    for i in aav2dfs.keys():
        scatter_plot_density(aav2dfs[i], f"AAV2_{i}")
