# Logic: 3 kinds of rewards. 
# 1 length and simple ones. like (in all or in goods) (done)
# 2. Three reward functions. Fitness, Thermostability and kidney  (done)
# 3. information retrieved from alignments.  (done)

from Bio import SeqIO, Align, pairwise2
from Bio.Seq import Seq
from Bio.SeqUtils import ProtParam, molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Align import PairwiseAligner, substitution_matrices
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import entropy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from Bio.SeqRecord import SeqRecord
from io import StringIO
from Bio import AlignIO
import subprocess
from math import log2
import json 
import torch 
from datasets import Dataset
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import (
    AutoTokenizer,
    EsmForSequenceClassification,
)
print(f"start time ==================================== >> {time.time()}")

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("/home/u111169/wrkdir/mgh/all/evaluation/AAV2-final-500_000-first-250_000-for-analysis.csv")

fitness_path = "/home/u111169/wrkdir/mgh/aav/dataset/production_main_merged_final.csv"
thermo_path = "/home/u111169/wrkdir/mgh/aav/dataset/Thermostability.csv"
kidney_path = "/home/u111169/wrkdir/mgh/aav/dataset/Kidney_Tropism.csv"
aav2_base_path = "/home/u111169/wrkdir/mgh/all/aav2_bases.csv"
esm_fitness = "/home/u111169/wrkdir/mgh/aav/checkpoints/new_final/fitness_checkpoint-14100"
esm_thermo = "/home/u111169/wrkdir/mgh/aav/checkpoints/new_final/thermostability_checkpoint-37500"
esm_kidney = "/home/u111169/wrkdir/mgh/aav/checkpoints/new_final/kidney_checkpoint-26600"
tok_path = "/home/u111169/blkdir/mgh/aav/models/models--facebook--esm2_t6_8M_UR50D/snapshots/c731040fcd8d73dceaa04b0a8e6329b345b0f5df"


print("Loading and preprocessing dataset...")

def read_all_csvs(path2dfs:str):
    all_files = os.listdir(path2dfs)
    all_csvs = [file for file in all_files if file.endswith(".csv")]
    aavdfs = {}
    aav_type = path2dfs.split("/")[-1]
    for csv_file in all_csvs: 
        file_name = csv_file.replace(".csv" , "")
        file_path = os.path.join(path2dfs , csv_file)
        aavdfs[file_name] = pd.read_csv(file_path)
        aavdfs[file_name]["aav_type"] = [aav_type]*len(aavdfs[file_name])
    return aavdfs

aav2dfs = read_all_csvs("/home/u111169/wrkdir/mgh/all/main_dataframes/AAV2")
aav9dfs = read_all_csvs("/home/u111169/wrkdir/mgh/all/main_dataframes/AAV9")

aav9 = "MAADGYLPDWLEDNLSEGIREWWALKPGAPQPKANQQHQDNARGLVLPGYKYLGPGNGLDKGEPVNAADAAALEHDKAYDQQLKAGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRLLEPLGLVEEAAKTAPGKKRPVEQSPQEPDSSAGIGKSGAQPAKKRLNFGQTGDTESVPDPQPIGEPPAAPSGVGSLTMASGGGAPVADNNEGADGVGSSSGNWHCDSQWLGDRVITTSTRTWALPTYNNHLYKQISNSTSGGSSNDNAYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTDNNGVKTIANNLTSTVQVFTDSDYQLPYVLGSAHEGCLPPFPADVFMIPQYGYLTLNDGSQAVGRSSFYCLEYFPSQMLRTGNNFQFSYEFENVPFHSSYAHSQSLDRLMNPLIDQYLYYLSKTINGSGQNQQTLKFSVAGPSNMAVQGRNYIPGPSYRQQRVSTTVTQNNNSEFAWPGASSWALNGRNSLMNPGPAMASHKEGEDRFFPLSGSLIFGKQGTGRDNVDADKVMITNEEEIKTTNPVATESYGQVATNHQSAQAQAQTGWVQNQGILPGMVWQDRDVYLQGPIWAKIPHTDGNFHPSPLMGGFGMKHPPPQILIKNTPVPADPPTAFNKDKLNSFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYYKSNNVEFAVNTEGVYSEPRPIGTRYLTRNL"
aav2 ="MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"
insert_pos = 588  
insert_idx = insert_pos  

def insert_sequence(row):
    insert_seq = row['AA']
    new_seq = aav9[:insert_idx] + insert_seq + aav9[insert_idx:]
    return new_seq

for i in aav9dfs.keys(): 
    aav9dfs[i]['AA'] = aav9dfs[i].apply(insert_sequence, axis=1)
    aav9dfs[i] = aav9dfs[i].rename(columns ={"AA":"final_seq" , f"{i}".replace("AAV9_", ""):"fitness_score"})

final_df_lists = list(aav9dfs.values())+ list(aav2dfs.values())
final_df = pd.concat(final_df_lists)
final_df = final_df[["final_seq" ,"fitness_score", "aav_type" ]]

unique_all_df = final_df.drop_duplicates(subset=['final_seq'], keep='first')[["final_seq","fitness_score", "aav_type"]].reset_index(drop = True)

######################################
##########   Evaluation Functions
######################################
def replace_redundant(text: str) -> str:
    return text.replace("\n", "")

def get_lengths(completions) -> list:
    return [len(replace_redundant(com)) for com in completions]


def is_in_all_fn(seq, df):
    return int(replace_redundant(seq) in df["final_seq"].tolist())


def is_in_high_fn(seq, df):
    return int(replace_redundant(seq) in df[df["fitness_score"] >= 0]["final_seq"].tolist())


####################################################################################################



def load_esm_model(model_path, num_labels=1, device= device):
    model = EsmForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    return model.to(device).eval()


# Load models once
fitness_model = load_esm_model(esm_fitness)
thermo_model = load_esm_model(esm_thermo)
kidney_model = load_esm_model(esm_kidney)
esm_tokenizer = AutoTokenizer.from_pretrained(tok_path)


def get_from_esm(sequences: list, esm_model, tokenizer, device= device, batch_size=512) -> list:
    sequences = [replace_redundant(seq) for seq in sequences]
    all_predictions = []
    
    # Move model to device once
    esm_model.to(device).eval()
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    
    # Process sequences in batches
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
            
            # Convert to list and flatten if needed
            batch_predictions = predictions.cpu().tolist()
            
            # Handle single item vs batch
            if isinstance(batch_predictions[0], list):
                # If nested, flatten it
                for pred in batch_predictions:
                    all_predictions.extend(pred)
            else:
                # If already flat, extend normally
                all_predictions.extend(batch_predictions)
    
    return all_predictions


# Generic reward function factory
def make_reward_fn(model):
    def reward_fn(completions):
        return get_from_esm(completions, model, esm_tokenizer)
    return reward_fn


# Define specific reward functions
fitness_reward = make_reward_fn(fitness_model)
kidney_reward = make_reward_fn(kidney_model)
thermo_reward = make_reward_fn(thermo_model)


input_seqs = df["generate_seqs"].tolist()
df["pred_fitness"] = fitness_reward(input_seqs)
df["pred_kidney"] = kidney_reward(input_seqs)
df["pred_them"] = thermo_reward(input_seqs)


df["length"] = get_lengths(input_seqs)
df["is_in_all"] = [is_in_all_fn(seq ,unique_all_df ) for seq in input_seqs]
df["is_in_high"] = [is_in_high_fn(seq ,unique_all_df)for seq in input_seqs]

df["is_wt_aav2"] = [1 if seq == aav2 else 0 for seq in input_seqs]
df["is_wt_aav9"] = [1 if seq == aav9 else 0 for seq in input_seqs]

df.to_csv("basics_and_models_eval_results_250_000.csv", index = False)