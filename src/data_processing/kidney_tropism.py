import load_dataframes
import compute_selection
from util import *
import pandas as pd 
pd_idx = pd.IndexSlice
import numpy as np
import os




package_counts = load_dataframes.load_packaging_df()
package_counts.sort_index(inplace=True)
mouse_counts = load_dataframes.load_mouse_df()

### filter out mutations with low counts and compute selection for mouse data
mouse_counts_filtered = mouse_counts.where(
        (mouse_counts > 10) & (mouse_counts <31000), np.nan)

package_selection = compute_selection.compute_packaging_selection(package_counts, level='aa')

new_columns = pd.MultiIndex.from_tuples(
    [('', '', '', '', col[0],col[1]) for col in package_selection.columns],
    names=['organ', '', '', '', '',''])
package_selection.columns = new_columns

new_mouse_selection = compute_selection.compute_mouse_selections(
        package_counts=package_counts, 
        mouse_counts=mouse_counts_filtered, 
        wt_norm=True,drop=True,
        drop_tile=False,
        return_freq=False)
mouse_aa_selection =  new_mouse_selection['aa_selection']


mouse_aa_selection_fitness = mouse_aa_selection.T.groupby(level='organ').median().replace(
        [np.inf,-np.inf,0],np.nan).apply(np.log2).T.dropna()


mouse_aa_selection_fitness = mouse_aa_selection_fitness["kidney"].reset_index()
holder = mouse_aa_selection_fitness.reset_index()
insertion_df = holder[holder["lib_type"]=="ins"]
sub_del_df = holder[holder["lib_type"]!="ins"]


insertion_df["final_seq"] = insertion_df.apply(lambda row:insertion(aav2seq ,row["abs_pos"] ,row["aa"]  ) , axis=1)
sub_del_df["final_seq"] = sub_del_df.apply(lambda row:sub_del_fun(aav2seq ,row["abs_pos"] ,row["aa"]  ) , axis=1)
final_dataset = pd.concat([insertion_df, sub_del_df] , axis=0)
final_dataset.rename(columns={"kidney":"fitness_score"} , inplace=True)
final_dataset['fitness_score'] = final_dataset['fitness_score'].replace([np.inf,  -np.inf], np.nan)
final_dataset = final_dataset.dropna(subset=['fitness_score'])

output_dir = "..\..\datasets\AAV2\processed"
os.makedirs(output_dir, exist_ok=True)

final_dataset.to_csv(os.path.join(output_dir, "Kidney_Tropism.csv"), index=False)
print("dataset is ready !!!!")