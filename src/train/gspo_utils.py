################### IMPORTS
import torch
import os
import csv
from sklearn.metrics import mean_absolute_error

device = "cuda" if torch.cuda.is_available() else "cpu"
########### util functions ###########

def is_in_all_fn(seq, df):
    if seq in df["final_seq"].tolist():
        return 1
    else:
        return 0


def is_in_high_fn(seq, df):
    if seq in df[df["fitness_score"] >= 0]["final_seq"].tolist():
        return 1
    else:
        return 0



def log_results(
    prompts,
    completions,
    rewards,
    preds,
    lengths,
    is_in_all,
    is_in_high,
    trail_out_path,
    file_ext,
):
    # Construct output strings
    outputs = []
    for i in range(len(prompts)):
        output = f"""{'-'*20}\n\nResponse:\n{completions[i]}\n\nScore:  {rewards[i]}\nPreds_real:  {preds[i]}\nis_in_all:  {is_in_all[i]}\nis_in_high:  {is_in_high[i]}\nlength:  {lengths[i]}"""
        outputs.append(output)

    combined_out = "combined_out:\n" + "\n\n".join(outputs)
    print(combined_out)

    # Prepare filename
    filename = f"{trail_out_path}{file_ext}"

    # Write results to CSV
    try:
        file_exists = os.path.exists(filename)
        with open(filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header if new file or empty
            if not file_exists or os.stat(filename).st_size == 0:
                writer.writerow(
                    [
                        "gen_seqs",
                        "reward",
                        "fitness_score",
                        "seq_len",
                        "is_in_all",
                        "is_in_high",
                    ]
                )

            for completion, reward, pred, length, in_all, in_high in zip(
                completions, rewards, preds, lengths, is_in_all, is_in_high
            ):
                writer.writerow([completion, reward, pred, length, in_all, in_high])

    except Exception as e:
        print(f"Failed to write to CSV file: {e}")


def log_other_results(prompts, rewards, preds, objective, trail_out_path, file_ext):
    # Construct output strings
    outputs = []
    for i in range(len(prompts)):
        output = f"""Score:  {rewards[i]}\nPreds_real:  {preds[i]}"""
        outputs.append(output)

    combined_out = f"{'-'*20}\n\n{objective}:\n" + "\n\n".join(outputs)
    print(combined_out)

    # Prepare filename
    filename = f"{trail_out_path}__{objective}{file_ext}"

    # Write results to CSV
    try:
        file_exists = os.path.exists(filename)
        with open(filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header if new file or empty
            if not file_exists or os.stat(filename).st_size == 0:
                writer.writerow(["reward", "pred"])

            for reward, pred in zip(rewards, preds):
                writer.writerow([reward, pred])

    except Exception as e:
        print(f"Failed to write to CSV file: {e}")


def calculate_mae_sklearn(df, true_col="y_true", pred_col="y_pred"):
    return mean_absolute_error(df[true_col], df[pred_col])


def util_compute_reward(x, a, seq_num, error):
    if x > seq_num + 4*error:
        return a
    if seq_num + 4*error>x > seq_num + error:
        return 3*a/4
    elif seq_num + error > x > seq_num : 
        return a/2
    elif x < seq_num and x>=a/2:
        return a/2
    else:
        return x


def final_reward_mapper(data, seq_num, error, df):
    max_d = -min(df)
    rewards = [util_compute_reward(x, max_d, seq_num, error) for x in data]
    return rewards


def get_from_esm(sequences: list, esm_model, esm_tokenizer) -> list:
    tokenized = esm_tokenizer(
        sequences, truncation=True, padding=True, max_length=755, return_tensors="pt"
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    esm_model.to(device)
    esm_model.eval()
    with torch.no_grad():
        outputs = esm_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = outputs.logits.squeeze().tolist()
    return predictions
