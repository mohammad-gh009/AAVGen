import load_dataframes
import compute_selection
from util import *
import pandas as pd 
import os
import numpy as np

pd_idx = pd.IndexSlice

package_data = load_dataframes.load_packaging_df()
tm_data = load_dataframes.load_thermostability_df()
tm_selection_dict = compute_selection.compute_tm_selection(package_data, tm_data,wt_norm=True,sum_all=True)

tma_aa_selection = tm_selection_dict['aa_selection']
tma_aa_mean = tma_aa_selection.loc[:,pd_idx[:,['67']]].groupby(level='virus', axis=1).mean()

# make the insertion dataset
insertion_pos_df = tma_aa_mean.query("lib_type.isin(['ins'])")[["CMV2"]].reset_index()[['abs_pos', 'aa',"CMV2"]]
insertion_pos_df["final_seq"] = insertion_pos_df.apply(lambda row:insertion(aav2seq ,row["abs_pos"] ,row["aa"]  ) , axis=1)


# make the substitution and deletion dataset
sub_del_df = tma_aa_mean.query("lib_type.isin(['sub','del'])")[["CMV2"]].reset_index()[['abs_pos', 'aa',"CMV2"]]
sub_del_df["final_seq"] = sub_del_df.apply(lambda row:sub_del_fun(aav2seq ,row["abs_pos"] ,row["aa"]  ) , axis=1)

# concatenate two datasts above 
final_dataset = pd.concat([insertion_pos_df, sub_del_df] , axis=0)
final_dataset.rename(columns={"CMV2":"fitness_score"} , inplace=True)
final_dataset["fitness_score"]= final_dataset["fitness_score"].apply(np.log2)

final_dataset['fitness_score'] = final_dataset['fitness_score'].replace([np.inf,  -np.inf], np.nan)
final_dataset = final_dataset.dropna(subset=['fitness_score'])

#save the final dataset
output_dir = "..\..\datasets\AAV2\processed"
os.makedirs(output_dir, exist_ok=True)

final_dataset.to_csv(os.path.join(output_dir, "Thermostability.csv"), index=False)

print("dataset is ready !!!!")