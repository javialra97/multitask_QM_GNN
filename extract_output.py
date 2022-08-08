import os
import pandas as pd

cross_val_dir = "cross_val"


def extract_from_line_list(line_list):
    mae_eact, mae_er, rmse_eact, rmse_er = None, None, None, None
    for line in reversed(line_list):
        if "MAE for 5-fold cross-validation" in line:
            mae_eact = float(line.split()[-4])
            mae_er = float(line.split()[-1])
        elif "RMSE for 5-fold cross-validation" in line:
            rmse_eact = float(line.split()[-5])
            rmse_er = float(line.split()[-1])

    return mae_eact, mae_er, rmse_eact, rmse_er


mae_eact_list = []
mae_er_list = []
rmse_eact_list = []
rmse_er_list = []

for dir in sorted(os.listdir(cross_val_dir)):
    try:
        sample_size = int(dir.split("_")[0])
    except ValueError:
        sample_size = 3200
    row_mae_eact, row_mae_er, row_rmse_eact, row_rmse_er = [], [], [], []
    path = os.path.join(cross_val_dir, dir)
    column_names = ["sample_size"]
    for dir2 in sorted(os.listdir(path)):
        path2 = os.path.join(path, dir2)
        with open(os.path.join(path2, "output.log"), "r") as f:
            line_list = f.readlines()
        mae_eact, mae_er, rmse_eact, rmse_er = extract_from_line_list(line_list)
        row_mae_eact.append(mae_eact)
        row_mae_er.append(mae_er)
        row_rmse_eact.append(rmse_eact)
        row_rmse_er.append(rmse_er)
        column_names.append(dir2)
    mae_eact_list.append([sample_size] + row_mae_eact)
    mae_er_list.append([sample_size] + row_mae_er)
    rmse_eact_list.append([sample_size] + row_rmse_eact)
    rmse_er_list.append([sample_size] + row_rmse_er)
    print(dir2, mae_eact, rmse_eact, mae_er, rmse_er)

df_mae_eact = pd.DataFrame(mae_eact_list, columns=column_names)
df_mae_er = pd.DataFrame(mae_er_list, columns=column_names)
df_rmse_eact = pd.DataFrame(rmse_eact_list, columns=column_names)
df_rmse_er = pd.DataFrame(rmse_er_list, columns=column_names)

df_mae_eact.to_csv("mae_eact.csv")
df_mae_er.to_csv("mae_er.csv")
df_rmse_eact.to_csv("rmse_eact.csv")
df_rmse_er.to_csv("rmse_er.csv")
