import compute_selection
import load_dataframes
import pandas as pd
import numpy as np
import os
from util import *
pd_idx =pd.IndexSlice

packaging_counts = load_dataframes.load_packaging_df()


# load aa package
package_aa_full_sum = compute_selection.compute_packaging_selection(
    packaging_counts,level='aa', wt_norm=True, sum_measurements=True, for_plotting=True)

# make the insertion dataset
insertion_pos_df = package_aa_full_sum.query("lib_type.isin(['ins'])").xs('CMV', level=0,axis=1).reset_index()[['abs_pos', 'aa','0']]
insertion_pos_df["final_seq"] = insertion_pos_df.apply(lambda row:insertion(aav2seq ,row["abs_pos"] ,row["aa"]  ) , axis=1)

# make the substitution and deletion dataset
sub_del_df = package_aa_full_sum.query("lib_type.isin(['sub','del'])").xs('CMV', level=0,axis=1).reset_index()[['abs_pos', 'aa','0']]
sub_del_df["final_seq"] = sub_del_df.apply(lambda row:sub_del_fun(aav2seq ,row["abs_pos"] ,row["aa"]  ) , axis=1)

# concatenate two datasts above 
final_dataset = pd.concat([insertion_pos_df, sub_del_df] , axis=0)
final_dataset.rename(columns={"0":"fitness_score"} , inplace=True)
final_dataset["fitness_score"]= final_dataset["fitness_score"].apply(np.log2)

final_dataset['fitness_score'] = final_dataset['fitness_score'].replace([np.inf,  -np.inf], np.nan)
final_dataset = final_dataset.dropna(subset=['fitness_score'])
# save the final dataset
output_dir = "..\..\datasets\AAV2\processed"
os.makedirs(output_dir, exist_ok=True)

final_dataset.to_csv(os.path.join(output_dir, "landscape_production_fitness.csv"), index=False)

print("dataset is ready !!!!")