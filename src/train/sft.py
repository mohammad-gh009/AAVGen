from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from datasets import Dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import time

# ========================================
start = time.time()
# PARAMS
model_name_or_path = "nferruz/ProtGPT2"

output_dir = "../../model/sft"
os.makedirs(output_dir, exist_ok=True)

# Training hyperparameters
batch_size = 4
gradient_accumulation = 4
per_device_eval_batch_size = 8
learning_rate = 1e-4
num_epochs = 3
logging_steps = 50
save_steps = 100
max_seq_length = 300
save_total_limit = 3
warmup_ratio = 0.01
weight_decay = 0.01
test_size = 0.2
eval_on_start = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, max_length=max_seq_length)
tokenizer.pad_token = tokenizer.eos_token

print("Loading and preprocessing dataset...")


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


aav2dfs = read_all_csvs("/home/u111169/wrkdir/mgh/all/main_dataframes/AAV2")
aav9dfs = read_all_csvs("/home/u111169/wrkdir/mgh/all/main_dataframes/AAV9")

aav9 = "MAADGYLPDWLEDNLSEGIREWWALKPGAPQPKANQQHQDNARGLVLPGYKYLGPGNGLDKGEPVNAADAAALEHDKAYDQQLKAGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRLLEPLGLVEEAAKTAPGKKRPVEQSPQEPDSSAGIGKSGAQPAKKRLNFGQTGDTESVPDPQPIGEPPAAPSGVGSLTMASGGGAPVADNNEGADGVGSSSGNWHCDSQWLGDRVITTSTRTWALPTYNNHLYKQISNSTSGGSSNDNAYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTDNNGVKTIANNLTSTVQVFTDSDYQLPYVLGSAHEGCLPPFPADVFMIPQYGYLTLNDGSQAVGRSSFYCLEYFPSQMLRTGNNFQFSYEFENVPFHSSYAHSQSLDRLMNPLIDQYLYYLSKTINGSGQNQQTLKFSVAGPSNMAVQGRNYIPGPSYRQQRVSTTVTQNNNSEFAWPGASSWALNGRNSLMNPGPAMASHKEGEDRFFPLSGSLIFGKQGTGRDNVDADKVMITNEEEIKTTNPVATESYGQVATNHQSAQAQAQTGWVQNQGILPGMVWQDRDVYLQGPIWAKIPHTDGNFHPSPLMGGFGMKHPPPQILIKNTPVPADPPTAFNKDKLNSFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYYKSNNVEFAVNTEGVYSEPRPIGTRYLTRNL"
insert_idx = 588


def insert_sequence(row):
    insert_seq = row["AA"]
    new_seq = aav9[:insert_idx] + insert_seq + aav9[insert_idx:]
    return new_seq


for i in aav9dfs.keys():
    aav9dfs[i]["AA"] = aav9dfs[i].apply(insert_sequence, axis=1)
    aav9dfs[i] = aav9dfs[i].rename(
        columns={"AA": "final_seq", f"{i}".replace("AAV9_", ""): "fitness_score"}
    )


final_df_lists = list(aav9dfs.values()) + list(aav2dfs.values())
final_df = pd.concat(final_df_lists)
final_df = final_df[["final_seq", "fitness_score", "aav_type"]]
final_df = final_df[final_df["fitness_score"] >= 0]
unique_df = final_df.drop_duplicates(subset=["final_seq"], keep="first")[
    ["final_seq", "aav_type"]
].reset_index(drop=True)

unique_df.rename(columns={"final_seq": "text"}, inplace=True)


def convert_to_fasta(seq):
    c = 0
    out = tokenizer.eos_token + "\n"
    for i in seq:
        c += 1
        out += i
        if c % 60 == 0 and c != 0:
            out += "\n"
    return out + "\n" + tokenizer.eos_token


unique_df["text"] = unique_df.apply(lambda x: convert_to_fasta(x["text"]), axis=1)

train_df, val_df = train_test_split(
    unique_df, stratify=unique_df["aav_type"], test_size=0.2, random_state=1234
)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)


print("Loading model ...")


model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.config.use_cache = False


print("Tokenizing dataset...")


def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding=False,
        truncation=True,
        max_length=max_seq_length,
    )


train_set = train_dataset.map(tokenize_function, batched=True)
train_set = train_set.remove_columns(["text", "aav_type"])
train_set.set_format(type="torch", columns=["input_ids", "attention_mask"])


val_set = val_dataset.map(tokenize_function, batched=True)
val_set = val_set.remove_columns(["text", "aav_type"])
val_set.set_format(type="torch", columns=["input_ids", "attention_mask"])

print("Setting up training arguments...")

training_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir=f"{output_dir}/logs",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation,
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    num_train_epochs=num_epochs,
    save_steps=save_steps,
    fp16=True,
    save_total_limit=save_total_limit,
    warmup_ratio=warmup_ratio,
    optim="adamw_torch",
    report_to="none",
    save_strategy="steps",
    torch_empty_cache_steps=save_steps,
    weight_decay=weight_decay,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    seed=42,
    data_seed=42,
    do_eval=False,
    lr_scheduler_type="linear",
)


print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_set,
    args=training_args,
)


print("Starting training...")
trainer.train()

print("Training complete. Model saved to:", output_dir)

end = time.time()


print(
    f"################                 total training time : {(end-start)/3600} hours"
)
