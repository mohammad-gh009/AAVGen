import pandas as pd 
import os 
import sys 
# sys.path.append("/dataframes")
df1 = pd.read_csv("../../datasets/AAV2/Deep_diversification_data.csv")
df2 = pd.read_csv("../../datasets/AAV2/Production_Fitness.csv")

df2 = df2[["final_seq", "fitness_score"]]
df = pd.concat([df1 , df2], axis= 0 )
df = df.drop_duplicates(subset=['final_seq']).reset_index(drop=True)

#save the final dataset
output_dir = "../../datasets/AAV2/processed"
os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, "production_main_merged_final.csv"), index=False)