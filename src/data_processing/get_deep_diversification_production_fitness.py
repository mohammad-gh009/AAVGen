import sys 
import pandas as pd 
import os
import numpy as np
aav2seq = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"

df = pd.read_csv("../../datasets/AAV2/Deep_diversification/allseqs_20191230.csv.zip")


def insertion(text: str, letter: str) -> str:
    return text[:560] + letter + text[588:]

df["final_seq"] = df.apply(lambda row:insertion(aav2seq,row["sequence"]  ) , axis=1)
final_df = df[["final_seq" , "viral_selection"]]
final_df.rename(columns={"viral_selection": "fitness_score"}, inplace=True)

final_df['fitness_score'] = final_df['fitness_score'].replace([np.inf,  -np.inf], np.nan)
final_dataset = final_df.dropna(subset=['fitness_score'])

output_dir = "..\..\datasets\AAV2\processed"
os.makedirs(output_dir, exist_ok=True)

final_df.to_csv(os.path.join(output_dir, "Deep_diversification_data.csv"), index=False)

print("dataset is ready !!!!")