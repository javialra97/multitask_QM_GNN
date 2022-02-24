import os
import pandas as pd

cross_val_dir = 'cross_val'

def extract_from_line_list(line_list):
    mae, rmse = None, None
    for line in reversed(line_list):
        if 'MAE for 5-fold cross-validation' in line:
            mae = float(line.split()[-1])
        elif 'RMSE for 5-fold cross-validation' in line:
            rmse = float(line.split()[-1])
            break

    return mae, rmse

mae_list = []
rmse_list = []

for dir in sorted(os.listdir(cross_val_dir)):
    try:
        sample_size = int(dir.split('_')[0])
    except ValueError:
        sample_size = 800
    row_mae, row_rmse = [], []
    path = os.path.join(cross_val_dir, dir)
    column_names = ['sample_size']
    for dir2 in sorted(os.listdir(path)):
        path2 = os.path.join(path, dir2)
        with open(os.path.join(path2, 'output.log'), 'r') as f:
            line_list = f.readlines()
        mae, rmse = extract_from_line_list(line_list)
        row_mae.append(mae)
        row_rmse.append(rmse)
        column_names.append(dir2)
    mae_list.append([sample_size] + row_mae)
    rmse_list.append([sample_size] + row_rmse)
    print(dir2, mae, rmse)

df_mae = pd.DataFrame(mae_list, columns = column_names)
df_rmse = pd.DataFrame(rmse_list, columns = column_names) 

df_mae.to_csv('mae.csv')
df_rmse.to_csv('rmse.csv')
