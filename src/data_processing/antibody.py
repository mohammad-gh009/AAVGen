import load_dataframes
import compute_selection
from util import *
import pandas as pd 
import os
import numpy as np

pd_idx = pd.IndexSlice
antibody_counts = load_dataframes.load_antibody_df()
packaging_counts = load_dataframes.load_packaging_df()


antibody_selection_df= compute_selection.compute_antibody_selection(
            ab_counts=antibody_counts, package_counts=packaging_counts, wt_norm=True)
antibody_selection_df.head()


antibody_df = (
    antibody_selection_df.mean(axis=1)  
    .apply(np.log2)                     
    .dropna()                          
    .to_frame('fitness_score')           
)

antibody_df.query("lib_type.isin(['ins'])")[['fitness_score']].reset_index()[['abs_pos','fitness_score',  'aa']]
# make the insertion dataset
insertion_pos_df = antibody_df.query("lib_type.isin(['ins'])")[['fitness_score']].reset_index()[['abs_pos', 'aa', 'fitness_score']]
insertion_pos_df["final_seq"] = insertion_pos_df.apply(lambda row:insertion(aav2seq ,row["abs_pos"] ,row["aa"]  ) , axis=1)

# make the substitution and deletion dataset
sub_del_df = antibody_df.query("lib_type.isin(['sub','del'])")[['fitness_score']].reset_index()[['abs_pos', 'aa','fitness_score']]
sub_del_df["final_seq"] = sub_del_df.apply(lambda row:sub_del_fun(aav2seq ,row["abs_pos"] ,row["aa"]  ) , axis=1)

final_dataset = pd.concat([insertion_pos_df, sub_del_df] , axis=0)

final_dataset['fitness_score'] = final_dataset['fitness_score'].replace([np.inf,  -np.inf], np.nan)
final_dataset = final_dataset.dropna(subset=['fitness_score'])

# save the final dataset
output_dir = "..\..\datasets\AAV2\processed"
os.makedirs(output_dir, exist_ok=True)

final_dataset.to_csv(os.path.join(output_dir, "Antibody.csv"), index=False)


print("dataset is ready !!!!")