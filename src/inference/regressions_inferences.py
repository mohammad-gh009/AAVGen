import argparse
import torch 
import pandas as pd
import os
import time
import csv
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser(description="Run ESM reward model predictions on sequences.")
    parser.add_argument("--df_name", type=str, required=True, help="Name of the input CSV file (without .csv extension)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for ESM inference (default: 512)")
    return parser.parse_args()

def replace_redundant(text: str) -> str:
    return text.replace("\n", "")

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

def load_esm_model(model_path, device, num_labels=1):
    model = EsmForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    return model.to(device).eval()

def get_from_esm(sequences: list, esm_model, tokenizer, device, batch_size=512) -> list:
    sequences = [replace_redundant(seq) for seq in sequences]
    all_predictions = []
    esm_model.to(device).eval()
    total_batches = (len(sequences) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(sequences), batch_size), total=total_batches, desc="Processing sequences"):
        batch_sequences = sequences[i:i + batch_size]
        tokenized = tokenizer(
            batch_sequences,
            truncation=True,
            padding=True,
            max_length=755,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        with torch.no_grad():
            outputs = esm_model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.squeeze()
            batch_predictions = predictions.cpu().tolist()
            if isinstance(batch_predictions[0], list):
                for pred in batch_predictions:
                    all_predictions.extend(pred)
            else:
                all_predictions.extend(batch_predictions)

    return all_predictions

def make_reward_fn(model, tokenizer, device, batch_size):
    def reward_fn(completions):
        return get_from_esm(completions, model, tokenizer, device, batch_size)
    return reward_fn

def main():
    args = parse_args()
    df_name = args.df_name
    batch_size = args.batch_size

    print(f"start time ==================================== >> {time.time()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = pd.read_csv(f"../../datasets/{df_name}.csv")
    NAME = f"../../datasets/{df_name}_out.csv"

    esm_fitness = "Moreza009/AAV-Fitness"
    esm_thermo = "Moreza009/AAV-Thermostability"
    esm_kidney = "Moreza009/AAV-Kidney-Tropism"
    tok_path = "facebook/esm2_t6_8M_UR50D"

    print("Loading and preprocessing dataset...")

    aav9 = "MAADGYLPDWLEDNLSEGIREWWALKPGAPQPKANQQHQDNARGLVLPGYKYLGPGNGLDKGEPVNAADAAALEHDKAYDQQLKAGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRLLEPLGLVEEAAKTAPGKKRPVEQSPQEPDSSAGIGKSGAQPAKKRLNFGQTGDTESVPDPQPIGEPPAAPSGVGSLTMASGGGAPVADNNEGADGVGSSSGNWHCDSQWLGDRVITTSTRTWALPTYNNHLYKQISNSTSGGSSNDNAYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTDNNGVKTIANNLTSTVQVFTDSDYQLPYVLGSAHEGCLPPFPADVFMIPQYGYLTLNDGSQAVGRSSFYCLEYFPSQMLRTGNNFQFSYEFENVPFHSSYAHSQSLDRLMNPLIDQYLYYLSKTINGSGQNQQTLKFSVAGPSNMAVQGRNYIPGPSYRQQRVSTTVTQNNNSEFAWPGASSWALNGRNSLMNPGPAMASHKEGEDRFFPLSGSLIFGKQGTGRDNVDADKVMITNEEEIKTTNPVATESYGQVATNHQSAQAQAQTGWVQNQGILPGMVWQDRDVYLQGPIWAKIPHTDGNFHPSPLMGGFGMKHPPPQILIKNTPVPADPPTAFNKDKLNSFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYYKSNNVEFAVNTEGVYSEPRPIGTRYLTRNL"
    insert_idx = 588

    def insert_sequence(row):
        return aav9[:insert_idx] + row['AA'] + aav9[insert_idx:]

    aav2dfs = read_all_csvs("../../datasets/AAV2/processed")
    aav9dfs = read_all_csvs("../../datasets/AAV9/processed")

    for i in aav9dfs.keys():
        aav9dfs[i]['AA'] = aav9dfs[i].apply(insert_sequence, axis=1)
        aav9dfs[i] = aav9dfs[i].rename(columns={"AA": "final_seq", f"{i}".replace("AAV9_", ""): "fitness_score"})

    final_df = pd.concat(list(aav9dfs.values()) + list(aav2dfs.values()))
    final_df = final_df[["final_seq", "fitness_score", "aav_type"]]
    unique_all_df = final_df.drop_duplicates(subset=['final_seq'], keep='first')[["final_seq", "fitness_score", "aav_type"]].reset_index(drop=True)

    print("Loading ESM models...")
    esm_tokenizer = AutoTokenizer.from_pretrained(tok_path)
    fitness_model = load_esm_model(esm_fitness, device)
    thermo_model = load_esm_model(esm_thermo, device)
    kidney_model = load_esm_model(esm_kidney, device)

    fitness_reward = make_reward_fn(fitness_model, esm_tokenizer, device, batch_size)
    kidney_reward = make_reward_fn(kidney_model, esm_tokenizer, device, batch_size)
    thermo_reward = make_reward_fn(thermo_model, esm_tokenizer, device, batch_size)

    print("Running predictions...")
    input_seqs = df["generate_seqs"].tolist()
    df["pred_fitness"] = fitness_reward(input_seqs)
    df["pred_kidney"] = kidney_reward(input_seqs)
    df["pred_them"] = thermo_reward(input_seqs)

    print(f"Saving to {NAME}...")
    df.to_csv(NAME, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
    print("Done.")

if __name__ == "__main__":
    main()