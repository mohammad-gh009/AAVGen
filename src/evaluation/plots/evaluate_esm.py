from eval_utils import *


experiment = "fitness_aav2"
model_path = f"/home/u111169/wrkdir/mgh/aav/checkpoints/esm-2_8m-fitness_aav2_0_2_valid/checkpoint
out_path = f"/home/u111169/wrkdir/mgh/aav/evaluation/{experiment}"
tok_path = "/home/u111169/blkdir/mgh/aav/models/models--facebook--esm2_t6_8M_UR50D/snapshots/c731040fcd8d73dceaa04b0a8e6329b345b0f5df"
test_data_path = f"/home/u111169/wrkdir/mgh/aav/dataset/hf_datasets/val_{experiment}_0_2_valid"


y_true, y_pred = evaluate_model(model_path, tok_path, test_data_path)
print("============================================================")
print("============================================================")
print("============================================================")
print("============================================================")
print("Evaluation Metrics:")
metrics = calculate_metrics(y_pred, y_true)
metrics2 = calculate_correlation_stats(y_pred, y_true)


final_df = pd.DataFrame({"y_true": y_true, "y_pred":y_pred })
final_df.to_csv(f"{out_path}_evaluted_df.csv" , index =False)