import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from matplotlib.patches import Patch
from datasets import load_dataset, concatenate_datasets

# style
sns.set_style("whitegrid")
sns.set_theme(style="whitegrid", context="talk")  #
plt.rcParams["font.family"] = "Arial"
sns.set_theme(style="whitegrid")

map = {
    "pred_fitness": "Pred. fitness",
    "pred_kidney": "Pred. kidney",
    "pred_them": "Pred. thermostability",
}


# Function to remove outliers using IQR method
def remove_outliers_iqr(data):
    if len(data) == 0:
        return data
    data = np.array(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median = np.median(data)
    print(f"Median: {median}, Lower bound: {Q1}, Upper bound: {Q3}")
    return data[(data >= lower_bound) & (data <= upper_bound)]


def remove_outliers_iqr_spec(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        median = df_clean[col].median()
        print("for column:", col)
        print(f"Median: {median}, Lower bound: {Q1}, Upper bound: {Q3}")
        df_clean = df_clean[
            (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        ]
    return df_clean


def calculate_mae_sklearn(df, true_col="y_true", pred_col="y_pred"):
    return mean_absolute_error(df[true_col], df[pred_col])


def analyze_repetitive_sequences(df, step_size=1000):
    """
    Analyze repetitive sequences in the generate_seq column
    showing how repetition counts change as we examine larger subsets
    """

    subset_sizes = []
    repetitive_counts = []
    unique_counts = []
    repetition_percentages = []

    # Analyze subsets from step_size to total length
    total_rows = len(df)

    for subset_size in range(step_size, total_rows + 1, step_size):
        # Get the subset of data
        subset = df.iloc[:subset_size]
        seq_counts = subset["generate_seqs"].value_counts()

        # Count sequences that appear more than once (repetitive)
        repetitive = sum(seq_counts > 1)
        unique = len(seq_counts)
        repetition_percentage = (repetitive / unique) * 100 if unique > 0 else 0

        # Store results
        subset_sizes.append(subset_size)
        repetitive_counts.append(repetitive)
        unique_counts.append(unique)
        repetition_percentages.append(repetition_percentage)

    return subset_sizes, repetitive_counts, unique_counts, repetition_percentages


def compute_reward(x, a, seq_num, error):
    if x > seq_num + 4 * error:
        return "best"
    if seq_num + 4 * error > x > seq_num + error:
        return "good"
    elif seq_num + error > x > seq_num:
        return "uncertain"
    elif x < seq_num and x >= a / 2:
        return "uncertain"
    else:
        return "bad"


def plot_3d_scatter_seaborn(df):
    """3D scatter plot with seaborn color palette and styling"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = sns.color_palette("viridis", as_cmap=True)

    scatter = ax.scatter(
        df["pred_fitness"],
        df["pred_kidney"],
        df["pred_them"],
        c=df["combined_score"],
        cmap=colors,
        alpha=0.6,
        s=30,
    )
    ax.set_xlabel("Pred. fitness", fontsize=12, labelpad=10)
    ax.set_ylabel("Pred. kidney", fontsize=12, labelpad=10)
    ax.set_zlabel("Pred. thermostability", fontsize=12, labelpad=10)

    plt.colorbar(scatter, ax=ax, shrink=0.8, label="Combined score")
    plt.tight_layout()
    plt.show()


def analyze_top_performers_seaborn(df, top_n=100):
    """Find and visualize top performers using seaborn styling (2D only)"""
    if len(df) > 50_000:
        df = df.sample(3000, random_state=42)
    else:
        df = df.copy()

    # Calculate composite score
    df_analysis = df.copy()
    df_analysis["composite_score"] = (
        df_analysis["pred_fitness"]
        + df_analysis["pred_kidney"]
        + df_analysis["pred_them"]
    ) / 3

    # Identify top performers
    top_performers = df_analysis.nlargest(top_n, "composite_score")

    df_analysis["category"] = "Regular"
    df_analysis.loc[df_analysis.index.isin(top_performers.index), "category"] = (
        f"Top {top_n}"
    )
    colors = {"Regular": "#D2B48C", f"Top {top_n}": "#1f77b4"}

    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(131)
    sns.scatterplot(
        data=df_analysis,
        x="pred_fitness",
        y="pred_kidney",
        hue="category",
        palette=colors,
        alpha=0.6,
        ax=ax1,
        legend=False,
    )
    sns.regplot(
        data=df_analysis,
        x="pred_fitness",
        y="pred_kidney",
        scatter=False,
        color="red",
        ax=ax1,
    )
    # Calculate correlations
    spearman_corr, _ = spearmanr(
        df_analysis["pred_fitness"], df_analysis["pred_kidney"]
    )
    ax1.text(
        0.05,
        0.95,
        f"ρ = {spearman_corr:.3f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        ),
    )
    ax1.set_xlabel(map["pred_fitness"])
    ax1.set_ylabel(map["pred_kidney"])

    ax2 = fig.add_subplot(132)
    sns.scatterplot(
        data=df_analysis,
        x="pred_fitness",
        y="pred_them",
        hue="category",
        palette=colors,
        alpha=0.6,
        ax=ax2,
        legend=False,
    )
    sns.regplot(
        data=df_analysis,
        x="pred_fitness",
        y="pred_them",
        scatter=False,
        color="red",
        ax=ax2,
    )
    # Calculate correlations
    spearman_corr, _ = spearmanr(df_analysis["pred_fitness"], df_analysis["pred_them"])
    ax2.text(
        0.05,
        0.95,
        f"ρ = {spearman_corr:.3f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        ),
    )
    ax2.set_xlabel(map["pred_fitness"])
    ax2.set_ylabel(map["pred_them"])

    ax3 = fig.add_subplot(133)
    sns.scatterplot(
        data=df_analysis,
        x="pred_kidney",
        y="pred_them",
        hue="category",
        palette=colors,
        alpha=0.6,
        ax=ax3,
    )
    sns.regplot(
        data=df_analysis,
        x="pred_kidney",
        y="pred_them",
        scatter=False,
        color="red",
        ax=ax3,
    )
    # Calculate correlations
    spearman_corr, _ = spearmanr(df_analysis["pred_kidney"], df_analysis["pred_them"])
    ax3.text(
        0.05,
        0.95,
        f"ρ = {spearman_corr:.3f}",
        transform=ax3.transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        ),
    )
    ax3.set_xlabel(map["pred_kidney"])
    ax3.set_ylabel(map["pred_them"])

    handles, labels = ax3.get_legend_handles_labels()

    ax3.get_legend().remove()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.05),
        frameon=True,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()

    return top_performers


if __name__ == "__main":
    df = load_dataset("Moreza009/AAVGen-dataset-out", split="data").to_pandas()
    df_aav2_base = pd.read_csv(r"../../../datasets/aav2_bases.csv")
    wt_fitness = float(
        df_aav2_base[df_aav2_base["name"].str.split("_").str[0] == "fitness"]["score"]
    )
    wt_kidney = float(
        df_aav2_base[df_aav2_base["name"].str.split("_").str[0] == "kidney"]["score"]
    )
    wt_thermo = float(
        df_aav2_base[df_aav2_base["name"].str.split("_").str[0] == "thermostability"][
            "score"
        ]
    )
    wt_fitness_mae = float(
        df_aav2_base[df_aav2_base["name"].str.split("_").str[0] == "fitness"]["MAE"]
    )
    wt_kidney_mae = float(
        df_aav2_base[df_aav2_base["name"].str.split("_").str[0] == "kidney"]["MAE"]
    )
    wt_thermo_mae = float(
        df_aav2_base[df_aav2_base["name"].str.split("_").str[0] == "thermostability"][
            "MAE"
        ]
    )

    ds = load_dataset("Moreza009/AAV_datasets")
    all_datasets = [ds[key] for key in ds.keys()]
    combined_dataset = concatenate_datasets(all_datasets)
    df_com = combined_dataset.to_pandas()
    df_filtered = df_com[df_com["fitness_score"] >= 0]
    df_original = df_filtered.drop_duplicates(subset=["final_seq", "aav_type"])
    df_original.rename(columns={"final_seq": "generate_seqs"}, inplace=True)

    similarity_df = pd.read_parquet(
        r"../../../datasets/basic_similarity.parquet", engine="fastparquet"
    )
    df_cleaned = df[
        (df["is_in_high"] != 1)
        & (df["is_in_all"] != 1)
        & (df["is_wt_aav2"] != 1)
        & (df["is_wt_aav9"] != 1)
    ]

    # in the training set
    df_in = df.drop_duplicates(subset="generate_seqs")
    df_in = df_in.reset_index(drop=True)
    print(f"number of training set samples: {len(df_in)}")

    # Equal to WT
    Equal_WT = df[(df["is_wt_aav2"] == 1)]
    print(f"number of samples equel to WT: {len(Equal_WT)}")

    sizes, rep_counts, unique_counts, rep_percentages = analyze_repetitive_sequences(
        df_cleaned, 1000
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sizes, rep_percentages, "r-", linewidth=2, marker="s", markersize=3)
    ax.set_xlabel("Subset size (N = 1000)")
    ax.set_ylabel("Repetitive sequences (%)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Remove duplicate sequences based on 'generate_seqs' column
    df_unique = df_cleaned.drop_duplicates(subset="generate_seqs")
    df_unique = df_unique.reset_index(drop=True)

    # WT values and MAE for each prediction column
    column_params = {
        "pred_fitness": {
            "wt_value": wt_fitness,
            "mae": wt_fitness_mae,
            "label": "Fitness",
        },
        "pred_kidney": {
            "wt_value": wt_kidney,
            "mae": wt_kidney_mae,
            "label": "Kidney Tropism",
        },
        "pred_them": {
            "wt_value": wt_thermo,
            "mae": wt_thermo_mae,
            "label": "Thermostability",
        },
    }

    # Calculate reward distributions for each column
    reward_distributions = {}

    for col_name, params in column_params.items():
        rewards = []
        for x in df_unique[col_name].dropna():
            reward = compute_reward(
                x, params["wt_value"], params["wt_value"], params["mae"]
            )
            rewards.append(reward)

        reward_counts = pd.Series(rewards).value_counts()
        reward_distributions[col_name] = {
            "counts": reward_counts,
            "label": params["label"],
        }

    # Define colors for each reward category
    reward_colors = {
        "best": "#2E8B57",  # Sea green
        "good": "#87CEEB",  # Sky blue
        "uncertain": "#FFD700",  # Gold
        "bad": "#DC143C",  # Crimson
    }

    # Common reward categories across all datasets
    all_categories = ["best", "good", "uncertain", "bad"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (col_name, data) in enumerate(reward_distributions.items()):
        counts = data["counts"]
        label = data["label"]

        # Ensure all categories are present
        for category in all_categories:
            if category not in counts:
                counts[category] = 0

        sorted_counts = counts.reindex(all_categories)
        total = sum(sorted_counts.values)
        percentages = 100 * sorted_counts.values / total

        # Create pie chart
        wedges, texts = axes[i].pie(
            sorted_counts.values,
            colors=[reward_colors[cat] for cat in sorted_counts.index],
            startangle=90,
            wedgeprops={"linewidth": 2, "edgecolor": "white"},
        )

        # Collect label information for smart positioning
        label_info = []
        for j, (wedge, category, count, percentage) in enumerate(
            zip(wedges, sorted_counts.index, sorted_counts.values, percentages)
        ):
            ang = (wedge.theta2 + wedge.theta1) / 2
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            label_info.append(
                {
                    "wedge": wedge,
                    "category": category,
                    "percentage": percentage,
                    "angle": ang,
                    "x": x,
                    "y": y,
                }
            )

        # Group labels by side (left/right) and sort by angle
        left_labels = [info for info in label_info if info["x"] < 0]
        right_labels = [info for info in label_info if info["x"] >= 0]

        left_labels.sort(key=lambda x: x["angle"], reverse=True)
        right_labels.sort(key=lambda x: x["angle"])

        # Function to distribute labels vertically and horizontally to avoid overlap
        def distribute_labels(labels, side="right"):
            if not labels:
                return

            min_spacing = 0.4  # Minimum vertical spacing between labels
            min_horizontal = (
                0.5  # Minimum horizontal spacing to push labels further out
            )
            base_distance = 1.45

            # Calculate ideal positions
            positions = []
            x_positions = []
            for info in labels:
                ideal_y = base_distance * info["y"]
                ideal_x = base_distance * info["x"]
                positions.append(ideal_y)
                x_positions.append(ideal_x)

            # Adjust vertical positions to avoid overlap
            for idx in range(len(positions)):
                if idx > 0:
                    min_y = positions[idx - 1] + min_spacing
                    if positions[idx] < min_y:
                        positions[idx] = min_y

            # Adjust horizontal positions if labels are too close vertically
            for idx in range(len(positions)):
                if idx > 0:
                    y_diff = abs(positions[idx] - positions[idx - 1])
                    if (
                        y_diff < min_spacing * 1.5
                    ):  # If labels are relatively close vertically
                        # Push this label further out horizontally
                        extra_distance = min_horizontal * (
                            1 - y_diff / (min_spacing * 1.5)
                        )
                        x_positions[idx] += extra_distance * (
                            1 if side == "right" else -1
                        )

            # Center the group vertically if it drifted too much
            if len(positions) > 1:
                center_offset = (positions[0] + positions[-1]) / 2
                for idx in range(len(positions)):
                    positions[idx] -= center_offset * 0.3

            # Draw annotations
            for info, adjusted_y, adjusted_x in zip(labels, positions, x_positions):
                ha = "left" if side == "right" else "right"

                axes[i].annotate(
                    f"{info['category'].capitalize()}\n{info['percentage']:.2f}%",
                    xy=(info["x"], info["y"]),
                    xytext=(adjusted_x, adjusted_y),
                    ha=ha,
                    va="center",
                    arrowprops=dict(
                        arrowstyle="-",
                        color=reward_colors[info["category"]],
                        linewidth=1.5,
                        shrinkA=5,
                        shrinkB=0,
                    ),
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor=reward_colors[info["category"]],
                        linewidth=1.5,
                        alpha=0.95,
                    ),
                )

        # Distribute labels on each side
        distribute_labels(left_labels, side="left")
        distribute_labels(right_labels, side="right")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)

    plt.show()

    # Print summary statistics
    print("Reward Category Distributions:")
    print("=" * 50)
    for col_name, data in reward_distributions.items():
        print(f"\n{data['label']}:")
        total = sum(data["counts"].values)
        for category in all_categories:
            count = data["counts"].get(category, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {category.capitalize()}: {count} ({percentage:.2f}%)")

    dirs = os.listdir(rf"fULL_PATH_TO_\regressoins_results")  # USE FULL PATH

    for file_name in dirs:
        df_pr = pd.read_csv(
            rf"fULL_PATH_TO_\regressoins_results\{file_name}"
        )  # USE FULL PATH
        round = file_name.split(".")[0]
        wt_score = float(
            df_aav2_base[df_aav2_base["name"].str.split("_").str[0] == round]["score"]
        )
        mae = calculate_mae_sklearn(df_pr)

    data1 = [len(i) for i in df_original["generate_seqs"].tolist()]
    data2 = df_unique["length"].tolist()

    print("original")
    data1_clean = remove_outliers_iqr(data1)
    print("generated")
    data2_clean = remove_outliers_iqr(data2)

    print(
        f"Original data1: {len(data1)} points, Clean data1: {len(data1_clean)} points"
    )
    print(
        f"Original data2: {len(data2)} points, Clean data2: {len(data2_clean)} points"
    )

    df_plot = pd.DataFrame(
        {
            "Length": list(data1_clean) + list(data2_clean),
            "Source": ["Original"] * len(data1_clean)
            + ["Generated"] * len(data2_clean),
        }
    )

    # Plot
    plt.figure(figsize=(8, 6))
    ax = sns.histplot(
        data=df_plot,
        x="Length",
        hue="Source",
        bins=10,
        element="step",
        stat="count",
        common_norm=False,
        legend=False,
    )
    vline = plt.axvline(735, color="black", linestyle="--", linewidth=2)
    plt.xlabel("Length")
    plt.ylabel("Frequency")

    legend_elements = [
        Patch(facecolor=sns.color_palette()[0], label="Training set"),
        Patch(facecolor=sns.color_palette()[1], label="Generated"),
        plt.Line2D(
            [0], [0], color="black", linestyle="--", linewidth=2, label="Wild-type"
        ),
    ]
    ax.legend(handles=legend_elements)

    plt.show()

    df_unique["combined_score"] = (
        df_unique["pred_fitness"] + df_unique["pred_kidney"] + df_unique["pred_them"]
    ) / 3
    plot_3d_scatter_seaborn(df_unique)

    analyze_top_performers_seaborn(df_unique, top_n=100)

    df_align = similarity_df[
        ["identity_percent", "similarity_percent", "alignment_score"]
    ]
    df_clean = remove_outliers_iqr_spec(
        df_align, ["identity_percent", "similarity_percent", "alignment_score"]
    )

    print(f"Original data points: {len(df_align)}")
    print(f"After removing outliers: {len(df_clean)}")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df_clean["identity_percent"],
        df_clean["similarity_percent"],
        c=df_clean["alignment_score"],
        cmap="viridis",
        alpha=0.6,
        s=50,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Alignment score", rotation=270, labelpad=20)

    plt.xlabel("Identity (%)")
    plt.ylabel("Similarity (%)")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    edit_distances = similarity_df["edit_distance"].dropna()
    edit_distances = remove_outliers_iqr(edit_distances)
    plt.figure(figsize=(8, 6))
    plt.hist(edit_distances, bins=15, alpha=0.7, color="lightcoral", edgecolor="black")
    plt.xlabel("Edit distance")
    plt.ylabel("Frequency")

    plt.tight_layout()

    # Calculate median and IQR
    identity_median = similarity_df["edit_distance"].median()
    identity_q1 = similarity_df["edit_distance"].quantile(0.25)
    identity_q3 = similarity_df["edit_distance"].quantile(0.75)

    print(
        f"edit - Median: {identity_median:.2f}%, IQR: {identity_q1:.2f}–{identity_q3:.2f}%"
    )
