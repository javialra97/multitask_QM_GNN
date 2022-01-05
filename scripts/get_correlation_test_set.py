import numpy as np
import pandas as pd
from time import sleep

df_test = pd.read_csv("cross_val_final_full/GNN/test_predicted_3.csv")

df_true = pd.read_csv("datasets/e2_sn2_regression_eq.csv")

df_test_list = df_test.values.tolist()

df_true_list = df_true.values.tolist()

correlation_list = []

for i in range(len(df_true_list)):
    for j in range(len(df_test_list)):
        if df_true_list[i][1] == df_test_list[j][1]:
            correlation_list.append([df_true_list[i][1], df_true_list[i][-1], df_test_list[j][-1]])

correlation_df = pd.DataFrame(correlation_list, columns=['reaction_id', 'activation_energy', "predicted"])

print(correlation_df.head())
print(len(correlation_df))

correlation_df.to_csv("cross_test_sn2_e2_GNN.csv")
