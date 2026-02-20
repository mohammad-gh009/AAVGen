import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np
from datasets import load_dataset

sns.set_style("whitegrid")
plt.figure(figsize=(9, 6))

map = {
    "pred_fitness": "Pred. fitness",
    "pred_kidney": "Pred. kidney",
    "pred_them": "Pred. thermostability",
    "median_rmsd": "Median RMSD",
}

mapper = {
    "median_rmsd_AAVGen": "Median RMSD (AAVGen)",
    "median_rmsd_random": "Median RMSD (Random)",
}


def clean_protein_sequence(seq):

    if pd.isna(seq):
        return ""
    seq = str(seq)
    seq_ascii = seq.encode("ascii", "ignore").decode("ascii")
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    seq_clean = "".join([c for c in seq_ascii.upper() if c in valid_aas])
    return seq_clean


def remove_outliers_iqr(data):
    if len(data) == 0:
        return data
    data = np.array(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]


def get_vp3(seq):
    return seq[216:]


def remove_outliers_iqr(df, columns, multiplier=1.5):
    """Remove outliers using IQR method across specified columns."""
    mask = pd.Series(True, index=df.index)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask &= (df[col] >= Q1 - multiplier * IQR) & (df[col] <= Q3 + multiplier * IQR)
    return df[mask]


def analyze_both_groups_combined(df, df_random):

    predictors = ["pred_fitness", "pred_kidney", "pred_them"]
    cols_to_check = predictors + ["median_rmsd"]

    df_low = remove_outliers_iqr(df[df["median_rmsd"] < 0.45].copy(), cols_to_check)
    df_high = remove_outliers_iqr(df[df["median_rmsd"] >= 0.45].copy(), cols_to_check)
    df_rand = remove_outliers_iqr(df_random.copy(), cols_to_check)

    df_low["group"] = "AAVGen (RMSD <0.45)"
    df_high["group"] = "AAVGen (RMSD ≥0.45)"
    df_rand["group"] = "Random"

    df_combined = pd.concat([df_low, df_high, df_rand])

    colors = {
        "AAVGen (RMSD <0.45)": "#4A90E2",
        "AAVGen (RMSD ≥0.45)": "#943126",
        "Random": "#E69F00",
    }

    group_data_map = {
        "AAVGen (RMSD <0.45)": df_low,
        "AAVGen (RMSD ≥0.45)": df_high,
        "Random": df_rand,
    }

    # --- Build stats dataframe ---
    records = []
    for group_name, group_data in group_data_map.items():
        for pred_col in predictors:
            corr, p_val = spearmanr(group_data[pred_col], group_data["median_rmsd"])
            records.append(
                {
                    "Group": group_name,
                    "Predictor": map[pred_col],
                    "Spearman ρ": round(corr, 4),
                    "p-value": round(p_val, 4),
                    f"Median {map[pred_col]}": round(group_data[pred_col].median(), 4),
                    "Median RMSD": round(group_data["median_rmsd"].median(), 4),
                    "n": len(group_data),
                }
            )

    df_stats = pd.DataFrame(records)
    print(df_stats.to_string(index=False))

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for pred_col, ax in zip(predictors, axes):
        sns.scatterplot(
            data=df_combined,
            x=pred_col,
            y="median_rmsd",
            hue="group",
            palette=colors,
            alpha=0.6,
            ax=ax,
            s=80,
        )

        for group_name, color in colors.items():
            group_data = group_data_map[group_name]
            if len(group_data) > 1:
                sns.regplot(
                    data=group_data,
                    x=pred_col,
                    y="median_rmsd",
                    scatter=False,
                    color=color,
                    ax=ax,
                    line_kws={"linewidth": 2},
                )

        ax.set_xlabel(map[pred_col])
        ax.set_ylabel(map["median_rmsd"])
        ax.legend().remove()

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=True,
    )

    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()

    return df_stats


def plot_length_vs_rmsd(df, df_random):

    df_plot = df.copy()
    df_random_plot = df_random.copy()

    df_plot["seq_length"] = df_plot["generate_seqs"].str.len()
    df_random_plot["seq_length"] = df_random_plot["generate_seqs"].str.len()

    cols_to_check = ["seq_length", "median_rmsd"]
    df_random_plot = remove_outliers_iqr(df_random_plot, cols_to_check)

    df_low = remove_outliers_iqr(
        df_plot[df_plot["median_rmsd"] < 0.45].copy(), cols_to_check
    )
    df_high = remove_outliers_iqr(
        df_plot[df_plot["median_rmsd"] >= 0.45].copy(), cols_to_check
    )

    df_low["group"] = "AAVGen (RMSD <0.45)"
    df_high["group"] = "AAVGen (RMSD ≥0.45)"
    df_random_plot["group"] = "Random"

    df_combined = pd.concat([df_low, df_high, df_random_plot])

    colors = {
        "AAVGen (RMSD <0.45)": "#4A90E2",
        "AAVGen (RMSD ≥0.45)": "#943126",
        "Random": "#E69F00",
    }

    markers = {
        "AAVGen (RMSD <0.45)": "o",
        "AAVGen (RMSD ≥0.45)": "s",
        "Random": "^",
    }

    alphas = {
        "AAVGen (RMSD <0.45)": 0.5,
        "AAVGen (RMSD ≥0.45)": 0.8,
        "Random": 0.4,
    }

    sizes = {
        "AAVGen (RMSD <0.45)": 80,
        "AAVGen (RMSD ≥0.45)": 100,
        "Random": 60,
    }

    group_data_map = {
        "AAVGen (RMSD <0.45)": df_low,
        "AAVGen (RMSD ≥0.45)": df_high,
        "Random": df_random_plot,
    }

    # --- Stats ---
    print("\n" + "=" * 60)
    print("SPEARMAN CORRELATION STATISTICS")
    print("=" * 60)
    for group_name, group_data in group_data_map.items():
        corr, p_val = spearmanr(group_data["seq_length"], group_data["median_rmsd"])
        print(
            f"{group_name:35s} | Spearman ρ = {corr:.4f} | p = {p_val:.4f} | n = {len(group_data)}"
        )
    print("=" * 60 + "\n")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 6))

    for group in colors.keys():
        group_data = df_combined[df_combined["group"] == group]

        if len(group_data) == 0:
            continue

        ax.scatter(
            group_data["seq_length"],
            group_data["median_rmsd"],
            c=colors[group],
            marker=markers[group],
            alpha=alphas[group],
            s=sizes[group],
            label=group,
            edgecolors="black",
            linewidth=0.5,
        )

        if len(group_data) > 5:
            z = np.polyfit(group_data["seq_length"], group_data["median_rmsd"], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(
                group_data["seq_length"].min(), group_data["seq_length"].max(), 50
            )
            ax.plot(
                x_trend,
                p(x_trend),
                color=colors[group],
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
            )

    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Median RMSD")

    legend = ax.legend(loc="best", frameon=True, framealpha=0.9)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.5)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return fig, ax


if __name__ == "__main__":
    df = load_dataset("Moreza009/AAVGen-dataset-out", split="data").to_pandas()
    df["generate_seqs"] = df["generate_seqs"].apply(clean_protein_sequence)
    df = df[
        (df["is_in_high"] != 1)
        & (df["is_in_all"] != 1)
        & (df["is_wt_aav2"] != 1)
        & (df["is_wt_aav9"] != 1)
    ]
    df.drop_duplicates(subset=["generate_seqs"], inplace=True)
    filtered_df = df[df["length"].isin(remove_outliers_iqr(df["length"]))]
    best = df[
        (df["mapped_fitness"].round(2) == 7.36)
        & (df["mapped_kidney"].round(2) == 3.5)
        & (df["mapped_thermo"].round(2) == 3.83)
    ]
    good = df[
        (df["mapped_fitness"].round(2) == 7.36)
        & (df["mapped_kidney"].round(2) == 2.63)
        & (df["mapped_thermo"].round(2) == 3.83)
    ]
    good_best_both = df[
        (df["mapped_fitness"].round(2) == 7.36)
        & (
            (df["mapped_kidney"].round(2) == 2.63)
            | (df["mapped_kidney"].round(2) == 3.5)
        )
        & (df["mapped_thermo"].round(2) == 3.83)
    ]
    good_best_both["combined_score"] = (
        good_best_both["pred_fitness"]
        + good_best_both["pred_kidney"]
        + good_best_both["pred_them"]
    ) / 3

    sample_500_random = good_best_both.sample(500, random_state=1234)
    sample_500_random["generate_seqs"] = sample_500_random["generate_seqs"].apply(
        get_vp3
    )
    df_500 = pd.read_csv(r"../../../datasets/sample_random_500.csv")
    df_folded = pd.read_csv(r"../../../datasets/rmsd_results_simple.csv")
    df_scores = pd.read_csv(r"../../../datasets/random_250_generated_evaluated.csv")
    df_folded_random = pd.read_csv(r"../../../datasets/rmsd_results_random.csv")

    df_folded["folder"] = df_folded["folder"].str.replace("_100", "", regex=False)
    df_folded["folder"] = df_folded["folder"].astype(int) - 1
    df_folded = df_folded.sort_values(["folder"], ascending=True, ignore_index=True)
    df_final = df_500.join(df_folded)
    df_final["median_rmsd"] = df_final[
        ["rmsd_model_0", "rmsd_model_1", "rmsd_model_2", "rmsd_model_3", "rmsd_model_4"]
    ].median(axis=1)

    df_folded_random["median_rmsd"] = df_folded_random[
        ["rmsd_model_0", "rmsd_model_1", "rmsd_model_2", "rmsd_model_3", "rmsd_model_4"]
    ].median(axis=1)
    df_folded_random["folder"] = (
        df_folded_random["folder"]
        .str.replace("random_", "", regex=False)
        .str.replace("_100", "", regex=False)
    )
    df_folded_random["folder"] = df_folded_random["folder"].astype(int) - 1
    df_folded_random = df_folded_random.sort_values(
        ["folder"], ascending=True, ignore_index=True
    )

    df_final_random = df_scores.join(df_folded_random)
    df_stats = analyze_both_groups_combined(df_final, df_final_random)

    plot_length_vs_rmsd(df_final, df_final_random)

    sns.kdeplot(
        data=df_final,
        x="median_rmsd",
        color="#0072B2",
        linewidth=3,
        fill=False,
        label=mapper["median_rmsd_AAVGen"],
    )
    sns.kdeplot(
        data=df_final_random,
        x="median_rmsd",
        color="#E69F00",
        linewidth=3,
        fill=False,
        label=mapper["median_rmsd_random"],
    )
    # histogram for median RMSD
    sns.histplot(
        data=df_final,
        x="median_rmsd",
        bins=50,
        color="#0072B2",
        alpha=0.2,
        stat="density",
    )
    sns.histplot(
        data=df_final_random,
        x="median_rmsd",
        bins=50,
        color="#E69F00",
        alpha=0.2,
        stat="density",
    )
    plt.xlabel("RMSD")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
