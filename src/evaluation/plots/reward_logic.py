import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import os 
import seaborn as sns
from matplotlib.lines import Line2D 
sns.set_theme(style="whitegrid", context="talk") # 
plt.rcParams['font.family'] = 'Arial'



def calculate_mae_sklearn(df, true_col='y_true', pred_col='y_pred'):
    return mean_absolute_error(df[true_col], df[pred_col])

def compute_reward(x, a, seq_num, error):
    if x > seq_num + 4*error:
        return a
    if seq_num + 4*error > x > seq_num + error:
        return 3*a/4
    elif seq_num + error > x > seq_num: 
        return a/2
    elif x < seq_num and x >= a/2:
        return a/2
    else:
        return x

def final_reward_mapper_plot(data, seq_num, error, file_name):
    a = -min(data)

    rewards = [compute_reward(x, a, seq_num, error) for x in data]

    # Create figure with larger font sizes
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()

    # -------- Histogram on primary axis -------- #
    n, bins, patches = ax.hist(data, bins='auto', alpha=0.7, edgecolor='black', linewidth=1.2, label='Histogram')

    from scipy import stats
    density = stats.gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 200)
    kde_line = ax.plot(x_range, density(x_range) * len(data) * (bins[1] - bins[0]), 
            color='darkblue', linewidth=3, alpha=0.8, label='KDE')[0]

    for patch in patches:
        if patch.get_x() + patch.get_width() > seq_num:
            patch.set_facecolor("#FF0000") 
            patch.set_alpha(0.8)
        else:
            patch.set_facecolor("#69EFE6")  
            patch.set_alpha(0.8)

    # Line styles
    line_styles = {
        'seq_num': ('#000000', '--', 3.0),
        'mean': ('#FF0000', '-', 4.0),
        'std': ('#555555', ':', 3.0),
        '2std': ('#888888', ':', 3.0),
    }

    # Vertical line for WT fitness score
    wt_line = ax.axvline(x=seq_num, color=line_styles['seq_num'][0], 
               linestyle=line_styles['seq_num'][1], 
               linewidth=line_styles['seq_num'][2], label='WT Fitness')

    ax.text(seq_num, ax.get_ylim()[0] - 0.005 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
            f'{seq_num:.2f}', 
            ha='center', va='top', fontsize=15, fontweight='normal',
            color=line_styles['seq_num'][0], zorder=10)

    # -------- Independent Reward Plot -------- #
    x_min = min(data)
    x_max = max(data)
    x_reward_range = np.linspace(x_min, x_max, 1000)
    
    reward_values = [compute_reward(x, a, seq_num, error) for x in x_reward_range]
    
    # Plot the reward function
    reward_line = ax2.plot(x_reward_range, reward_values, alpha=0.8, color='black', 
             linewidth=3, label='Reward Function')[0]
    

    ax.set_xlabel('Predicted values', fontsize=20, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=20, fontweight='bold', color='blue')
    ax2.set_ylabel('Reward', fontsize=20, fontweight='bold', color='black')
    
    ax.set_facecolor("#FFFFFF")

    ax.tick_params(axis='y', colors='blue')
    ax2.tick_params(axis='y', colors='black')

    lines = [ wt_line, reward_line] 
    labels = [ 'Wild-type', 'Reward Logic'] 
    legend = ax.legend(lines, labels, loc='upper left', framealpha=1.0)

    ax.grid(False)
    ax2.grid(False)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__": 
    dirs = os.listdir(fr"fULL_PATH_TO_\regressoins_results") # USE FULL PATH 
    df_aav2_base = pd.read_csv(r"../../../datasets/aav2_bases.csv")
    for file_name in dirs: 
        df_pr = pd.read_csv(rf"fULL_PATH_TO_\regressoins_results\{file_name}") # USE FULL PATH 
        round = file_name.split(".")[0]
        wt_score = float(df_aav2_base[df_aav2_base["name"].str.split("_").str[0] == round]["score"])
        mae = calculate_mae_sklearn(df_pr)
        final_reward_mapper_plot(np.array(df_pr["y_pred"].tolist()) ,wt_score, mae, round)