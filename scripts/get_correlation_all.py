import numpy as np
import pandas as pd
from time import sleep


def get_dataframe(test_file, true_file):
    df_test = pd.read_csv(test_file)
    df_true = pd.read_csv(true_file)
    df_test_list = df_test.values.tolist()
    df_true_list = df_true.values.tolist()
    correlation_list = []

    for j in range(len(df_true_list)):
        for k in range(len(df_test_list)):
            if df_true_list[j][1] == df_test_list[k][1]:
                correlation_list.append([df_true_list[j][1], df_true_list[j][-1], df_test_list[k][-1]])

    correlation_df = pd.DataFrame(correlation_list, columns=['reaction_id', 'activation_energy', "predicted"])

    return correlation_df


true_file = "datasets/Hong_dataset_filtered.csv"
folder_locations = ["cross_val_QM/Hong_charges_fukuis/"]

i = 0

for location in folder_locations:
    test_files = [location + "test_predicted_1.csv",
                  location + "test_predicted_2.csv",
                  location + "test_predicted_3.csv",
                  location + "test_predicted_4.csv",
                  ]

    correlation_df_tmp = get_dataframe(location + "test_predicted_0.csv", true_file)

    for test_file in test_files:
        tmp = get_dataframe(test_file, true_file)
        print(tmp.head())
        correlation_df_tmp = correlation_df_tmp.append(tmp)

    if i == 0:
        correlation_df = correlation_df_tmp
    else:
        correlation_df = correlation_df.append(correlation_df_tmp)

    i += 1


print(len(correlation_df))
correlation_df.to_csv("correlation_charges_fukuis_spin_dens.csv")
