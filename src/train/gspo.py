import time
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch
from trl import GRPOConfig, GRPOTrainer
import time
import os
from transformers import (
    AutoTokenizer,
    EsmForSequenceClassification,
)
from datasets import Dataset
import os
from gspo_utils import *

start = time.time()
print(f"start time ==================================== >> {start}")


######################################
##########   PARMS & PATHS
######################################
rutime_name = "AAVGen"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

max_prompt_length = 1
max_completion_length = 754
max_seq_length = 755
per_device_train_batch_size = 4
gradient_accumulation_steps = 8
num_generations = 32
torch_empty_cache_steps = 5
num_train_epochs = 5  #################
learning_rate = 2e-6
save_total_limit = 5
logging_steps = 1
warmup_steps = 50
data_init_number = 500  # number of rows in the dataset that we want to do grpo on
FILE_EXT = ".csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

fitness_path = "../../datasets/AAV2/processed/production_main_merged_final.csv"
thermo_path = "../../datasets/AAV2/processed/Thermostability.csv"
kidney_path = "../../datasets/AAV2/processed/Kidney_Tropism.csv"
aav2_base_path = "/home/u111169/wrkdir/mgh/all/aav2_bases.csv"
esm_fitness = "Moreza009/AAV-Fitness"
esm_thermo = "Moreza009/AAV-Thermostability"
esm_kidney = "Moreza009/AAV-Kidney-Tropism"
gen_model = "Moreza009/AAVGen"

tok_path = "facebook/esm2_t6_8M_UR50D"

df_aav2_base = pd.read_csv(aav2_base_path)
df_fitness = pd.read_csv(fitness_path)
df_thermo = pd.read_csv(thermo_path)
df_kidney = pd.read_csv(kidney_path)


df_pred_fitness = pd.read_csv(f"fitness.csv")
df_pred_kidney = pd.read_csv(f"kidney.csv")
df_pred_thermo = pd.read_csv(f"thermo.csv")

wt_fitness = float(
    df_aav2_base[df_aav2_base["name"].str.split("_").str[0] == "fitness"]["score"]
)
wt_thermo = float(
    df_aav2_base[df_aav2_base["name"].str.split("_").str[0] == "thermostability"]["score"]
)
wt_kidney = float(
    df_aav2_base[df_aav2_base["name"].str.split("_").str[0] == "kidney"]["score"]
)

mae_fitness = calculate_mae_sklearn(df_pred_fitness)
mae_kidney = calculate_mae_sklearn(df_pred_kidney)
mae_thermo = calculate_mae_sklearn(df_pred_thermo)


TRAIL_OUT_PATH = (
    f"../../outs/{rutime_name}_{time.strftime('%Y%m%d-%H%M%S')}"
)



out_dir = f"../../gspo/{rutime_name}_lr{learning_rate}_num_gen{num_generations}_t{time.strftime('%Y%m%d-%H%M%S')}"

os.mkdir(out_dir)
OUTPUT_DIR = out_dir
logging_dir = f"{OUTPUT_DIR}/log/"


fitness_model = (
    EsmForSequenceClassification.from_pretrained(esm_fitness, num_labels=1)
    .to("cpu")
    .eval()
)
thermo_model = (
    EsmForSequenceClassification.from_pretrained(esm_thermo, num_labels=1)
    .to("cpu")
    .eval()
)
kidney_model = (
    EsmForSequenceClassification.from_pretrained(esm_kidney, num_labels=1)
    .to("cpu")
    .eval()
)

esm_tokenizer = AutoTokenizer.from_pretrained(tok_path)


######################################
##########   Load Generative Model
######################################
model = AutoModelForCausalLM.from_pretrained(gen_model)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(
    gen_model, padding=False, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

def replace_redundant(text):
    text = text.replace("\n", "")
    text = text.replace(tokenizer.eos_token, "")
    return text


######################################
##########   Reward Functions
######################################


def fitness_reward(prompts, completions, **kwargs) -> list:

    coms = [replace_redundant(com) for com in completions]
    preds = get_from_esm(coms, fitness_model, esm_tokenizer)
    rewards = final_reward_mapper(
        preds, wt_fitness, mae_fitness, df_pred_fitness["y_pred"]
    )
    is_in_all = [is_in_all_fn(com, df_fitness) for com in coms]
    is_in_high = [is_in_high_fn(com, df_fitness) for com in coms]
    lengths = [len(com) for com in coms]

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    log_results(
        prompts,
        completions,
        rewards,
        preds,
        lengths,
        is_in_all,
        is_in_high,
        TRAIL_OUT_PATH,
        FILE_EXT,
    )

    return rewards


def kidney_reward(prompts, completions, **kwargs) -> list:

    coms = [replace_redundant(com) for com in completions]
    preds = get_from_esm(coms, kidney_model, esm_tokenizer)
    rewards = final_reward_mapper(
        preds, wt_kidney, mae_kidney, df_pred_kidney["y_pred"]
    )

    return rewards


def thermo_reward(prompts, completions, **kwargs) -> list:
    
    coms = [replace_redundant(com) for com in completions]
    preds = get_from_esm(coms, thermo_model, esm_tokenizer)
    rewards = final_reward_mapper(
        preds, wt_thermo, mae_thermo, df_pred_thermo["y_pred"]
    )

    return rewards


def length2tails_reward(
    prompts, completions, **kwargs
):  
    target_length = 735
    sigma = 3
    rewards = []
    for seq in completions:
        seq = replace_redundant(seq)
        length = len(seq)
        reward = np.exp(-((length - target_length) ** 2) / (2 * sigma**2))
        reverse_reward = 1 - reward
        rewards.append(reverse_reward)
    return rewards


def repeated_in_batch_reward(prompts, completions, **kwargs) -> list: 
    rewards = []
    for com in completions :

        if completions.count(com) > 1:
            rewards.append(0)
        else: 
            rewards.append(1)
            
    return rewards
######################################
##########   BUILD DATAFRAME
######################################
df = pd.DataFrame({"prompt": [ esm_tokenizer.eos_token+ "\n" +"M"] * data_init_number})#
dataset = Dataset.from_pandas(df)

print("dataframes are loaded and dataset is made. ")

######################################
##########   training loop
######################################
print("config . . . ")
grpo_config = {
    # Standard Training Arguments
    "output_dir": OUTPUT_DIR,
    "per_device_train_batch_size": per_device_train_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "learning_rate": learning_rate,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,
    "optim": "adamw_torch",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "num_train_epochs": num_train_epochs,
    "warmup_steps": warmup_steps,
    "logging_dir": logging_dir,
    "logging_strategy": "steps",
    "logging_first_step": False,
    "logging_steps": logging_steps,
    "save_strategy": "epoch",
    "save_total_limit": save_total_limit,
    "save_safetensors": True,
    "seed": 42,
    "data_seed": 42,
    "fp16": True,
    "resume_from_checkpoint": None,
    "report_to": "none",
    "torch_empty_cache_steps": torch_empty_cache_steps,
    "gradient_checkpointing": True,
    "importance_sampling_level": 'sequence',
    "reward_weights": [1.0, 1.0, 1.0, 0.1 , 0.1],
    # Generation Parameters
    "max_prompt_length": max_prompt_length,
    "num_generations": num_generations,
    "max_completion_length": max_completion_length,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": None,
    "repetition_penalty": 1.0,
}

training_args = GRPOConfig(**grpo_config)
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[fitness_reward, kidney_reward, thermo_reward, length2tails_reward , repeated_in_batch_reward],
    args=training_args,
    train_dataset=dataset,
)

print("ready to train . . . ")
trainer.train()#


end = time.time()
print(f"end time ==================================== >> {end}")
print(f"total time ==================================== >> {(end - start)/3600} hours")