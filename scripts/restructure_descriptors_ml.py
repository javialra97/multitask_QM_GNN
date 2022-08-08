import numpy as np
import pandas as pd


def convert_str_to_array(string):
    string = string.strip('"[]\n')
    string_list = string.split()

    try:
        string_list = list(map(float, string_list))
    except:
        string_list = np.array([])

    return np.array(string_list)


descriptors = pd.read_csv("../descriptors/descriptors_ml.csv")

descriptors["partial_charge"] = descriptors["partial_charge"].apply(
    lambda x: convert_str_to_array(x)
)
descriptors["fukui_neu"] = descriptors["fukui_neu"].apply(
    lambda x: convert_str_to_array(x)
)
descriptors["fukui_elec"] = descriptors["fukui_elec"].apply(
    lambda x: convert_str_to_array(x)
)
descriptors["NMR"] = descriptors["NMR"].apply(lambda x: convert_str_to_array(x))
descriptors["bond_order"] = descriptors["bond_order"].apply(
    lambda x: convert_str_to_array(x)
)
descriptors["bond_length"] = descriptors["bond_length"].apply(
    lambda x: convert_str_to_array(x)
)

descriptors.to_pickle("../descriptors/descriptors_ml.pkl")
descriptors.to_csv("../descriptors/test_ml.csv")
