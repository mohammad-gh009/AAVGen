################### IMPORTS
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import mean_absolute_error
from transformers import (
    AutoTokenizer,
    EsmForSequenceClassification,
)
import torch

aav2seq="MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"
esm_fitness = "/home/u111169/wrkdir/mgh/aav/checkpoints/new_final/fitness_checkpoint-14100"
esm_thermo = "/home/u111169/wrkdir/mgh/aav/checkpoints/new_final/thermostability_checkpoint-37500"
esm_kidney = "/home/u111169/wrkdir/mgh/aav/checkpoints/new_final/kidney_checkpoint-26600"

df_pred_fitness = pd.read_csv(f"/home/u111169/wrkdir/mgh/aav/evaluation/new_final/kidney_aav2_final_26600/_evaluted_df.csv")
df_pred_kidney = pd.read_csv(f"/home/u111169/wrkdir/mgh/aav/evaluation/new_final/fitness_aav2_final_final_14100/_evaluted_df.csv")
df_pred_thermo = pd.read_csv(f"/home/u111169/wrkdir/mgh/aav/evaluation/new_final/thermo_final_37500/_evaluted_df.csv")

tok_path = "/home/u111169/blkdir/mgh/aav/models/models--facebook--esm2_t6_8M_UR50D/snapshots/c731040fcd8d73dceaa04b0a8e6329b345b0f5df"

device = "cuda" if torch.cuda.is_available() else "cpu"
esm_tokenizer = AutoTokenizer.from_pretrained(tok_path)


def calculate_mae_sklearn(df, true_col="y_true", pred_col="y_pred"):
    return mean_absolute_error(df[true_col], df[pred_col])

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

mae_fitness = calculate_mae_sklearn(df_pred_fitness)
mae_kidney = calculate_mae_sklearn(df_pred_kidney)
mae_thermo = calculate_mae_sklearn(df_pred_thermo)

aav2_dict = {}
for checkpoint_path in [esm_kidney , esm_thermo , esm_fitness] : 
    model = (
        EsmForSequenceClassification.from_pretrained(checkpoint_path, num_labels=1)
        .to("cpu")
        .eval()
    )
    aav2_dict[checkpoint_path.split("/")[-1]] = get_from_esm([aav2seq], model, esm_tokenizer)
    

aav2_dict

final_df = pd.DataFrame({"name": aav2_dict.keys() , "score": aav2_dict.values() , "MAE":[mae_kidney , mae_thermo , mae_fitness]})
final_df.to_csv("aav2_bases.csv" , index = False)


