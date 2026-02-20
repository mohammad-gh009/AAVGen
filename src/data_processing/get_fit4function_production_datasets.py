import pandas as pd 
import numpy as np 
import os 

output_dir = "../../datasets/AAV9/processed"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("../../datasets/AAV9/fit4function_library_screens.csv")


df[['Production1', 'Production2']] = df[['Production1', 'Production2']].replace([np.inf, -np.inf], np.nan)
df_prod = (
    df[['AA', 'Production1', 'Production2']]
    .dropna()
    .assign(Production=lambda x: x[['Production1', 'Production2']].mean(axis=1))
    [['AA', 'Production']]
)

df_prod.to_csv(os.path.join(output_dir, "AAV9_Production.csv"), index=False)

for col in df.columns : 
    if col == "AA" or  col == "Production1" or col == "Production2": 
        pass
    else: 
        df[col] = df[col].replace([np.inf,  -np.inf], np.nan)
        df_new = df[["AA", col]]
        df_new = df_new.dropna()
        df_new.to_csv(os.path.join(output_dir, f"AAV9_{col}.csv"), index=False)

print("datasets are ready !!!!")