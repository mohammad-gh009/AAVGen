import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns

checkpoint = "checkpoint-2000"
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["font.family"] = "Arial"

def plot_final_reward(json_path, window_size=100):

    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

    data = json_data["log_history"]

    steps = [entry["step"] for entry in data if "step" in entry and "reward" in entry]
    rewards = [
        entry["reward"] for entry in data if "step" in entry and "reward" in entry
    ]

    Q1 = np.percentile(rewards, 25)
    Q3 = np.percentile(rewards, 75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_rewards = [x for x in rewards if lower_bound <= x <= upper_bound]
    filtered_steps = [
        x for x, y in zip(steps, rewards) if lower_bound <= y <= upper_bound
    ]

    df = pd.DataFrame({"Step": filtered_steps, "Reward": filtered_rewards})

    df["Moving_Avg"] = df["Reward"].rolling(window=window_size).mean()

    plt.figure(figsize=(12, 6))

    plt.plot(df["Step"], df["Reward"], alpha=0.3, label="Raw Rewards", color="#5DABD6")

    plt.plot(
        df["Step"],
        df["Moving_Avg"],
        label=f"{window_size}-Step Moving Average (MA)",
        color="#FF0000",
        linewidth=2,
    )

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()


def plot_training_metrics(json_path, window_size=10):
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

    data = json_data["log_history"]

    metrics = {}
    for entry in data:
        for key, value in entry.items():
            if key != "step" and key != "epoch":
                if key.startswith("rewards/") and key.endswith("/mean"):
                    if key not in metrics:
                        metrics[key] = {"steps": [], "values": []}
                    if "step" in entry:
                        metrics[key]["steps"].append(entry["step"])
                        metrics[key]["values"].append(value)

    for metric_name, metric_data in metrics.items():
        plt.figure(figsize=(12, 6))

        steps = metric_data["steps"]
        values = metric_data["values"]

        plt.plot(steps, values, alpha=0.3, label="Raw Rewards", color="#5DABD6")

        if window_size > 1 and len(values) >= window_size:
            moving_avg = np.convolve(
                values, np.ones(window_size) / window_size, mode="valid"
            )
            padded_moving_avg = np.concatenate(
                [np.full(window_size - 1, np.nan), moving_avg]
            )

            plt.plot(
                steps,
                padded_moving_avg,
                label=f"MA ({window_size} steps)",
                color="#FF0000",
                linewidth=2,
            )

        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend(loc="lower right")
        plt.tight_layout()

        metric_name = metric_name.replace("/", "_")

        plt.show()

    plt.close("all")


if __name__ == "__main__":

    plot_final_reward(r"trainer_states\gspo\gspo_trainer_state.json", window_size=100)
    plot_training_metrics(
        r"trainer_states\gspo\gspo_trainer_state.json", window_size=100
    )
