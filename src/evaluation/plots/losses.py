import json
import matplotlib.pyplot as plt
import seaborn as sns
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["font.family"] = "Arial"

sns.set_theme(style="whitegrid", context="talk")  #
plt.rcParams["font.family"] = "Arial"

# Regression models training and validation losses


sns.set_theme(style="whitegrid", context="talk")  #
plt.rcParams["font.family"] = "Arial"


def plot_training_loss(file_path, name):

    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)

    data = json_data["log_history"]

    steps = [entry["step"] for entry in data if "step" in entry and "loss" in entry]
    steps_ev = [
        entry["step"] for entry in data if "step" in entry and "eval_loss" in entry
    ]
    losses_t = [
        entry["loss"] / 8 for entry in data if "step" in entry and "loss" in entry
    ]  # here 8 is the gradient accumulation steps used in training, the trainer_state.json doesn't handle it automatically
    losses_e = [
        entry["eval_loss"] for entry in data if "step" in entry and "eval_loss" in entry
    ]

    plt.plot(steps, losses_t, marker="o", label="Training Loss", color="#4C72B0")
    plt.plot(steps_ev, losses_e, marker="o", label="Evaluation Loss", color="#DD8452")

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    # plt.title(f"{name} Training and Evaluation Loss")
    plt.legend()
    plt.tight_layout()

    plt.show()


def read_all_jsons(path2dfs: str):
    all_files = os.listdir(path2dfs)
    all_jsons = [file for file in all_files if file.endswith(".json")]
    for csv_file in all_jsons:
        file_name = csv_file.replace(".json", "").split("-")[0]
        file_path = os.path.join(path2dfs, csv_file)
        plot_training_loss(file_path, file_name)


if __name__ == "__main__": 
    aav2jsons = read_all_jsons(r"trainer_states")
