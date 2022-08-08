import numpy as np
import pandas as pd


def convert_str_to_array(string):
    string = string.strip('"[]\n')
    string_list = string.split(",")

    try:
        string_list = list(map(float, string_list))
    except:
        string_list = np.array([])

    return np.array(string_list)


def get_mo_weight_array(string, index):
    mo_string = string.split("], [")[index]

    return convert_str_to_array(mo_string)


descriptors = pd.read_csv("../descriptors/descriptors_Lowdin.csv")

descriptors["partial_charge"] = descriptors["charges"].apply(
    lambda x: convert_str_to_array(x)
)
descriptors["fukui_neu"] = descriptors["fukui_neu"].apply(
    lambda x: convert_str_to_array(x)
)
descriptors["fukui_elec"] = descriptors["fukui_elec"].apply(
    lambda x: convert_str_to_array(x)
)
descriptors["NMR"] = descriptors["nmr"].apply(lambda x: convert_str_to_array(x))
descriptors["bond_order"] = descriptors["bond_order"].apply(
    lambda x: convert_str_to_array(x)
)
descriptors["bond_length"] = descriptors["bond_length"].apply(
    lambda x: convert_str_to_array(x)
)

descriptors["HOMO-4"] = descriptors["mo_weights"].apply(
    lambda x: get_mo_weight_array(x, 0)
)
descriptors["HOMO-3"] = descriptors["mo_weights"].apply(
    lambda x: get_mo_weight_array(x, 1)
)
descriptors["HOMO-2"] = descriptors["mo_weights"].apply(
    lambda x: get_mo_weight_array(x, 2)
)
descriptors["HOMO-1"] = descriptors["mo_weights"].apply(
    lambda x: get_mo_weight_array(x, 3)
)
descriptors["HOMO"] = descriptors["mo_weights"].apply(
    lambda x: get_mo_weight_array(x, 4)
)
descriptors["LUMO"] = descriptors["mo_weights"].apply(
    lambda x: get_mo_weight_array(x, 5)
)
descriptors["LUMO+1"] = descriptors["mo_weights"].apply(
    lambda x: get_mo_weight_array(x, 6)
)
descriptors["LUMO+2"] = descriptors["mo_weights"].apply(
    lambda x: get_mo_weight_array(x, 7)
)
descriptors["LUMO+3"] = descriptors["mo_weights"].apply(
    lambda x: get_mo_weight_array(x, 8)
)
descriptors["LUMO+4"] = descriptors["mo_weights"].apply(
    lambda x: get_mo_weight_array(x, 9)
)

descriptors = descriptors[
    [
        "smiles",
        "partial_charge",
        "fukui_neu",
        "fukui_elec",
        "NMR",
        "bond_order",
        "bond_length",
        "HOMO-4",
        "HOMO-3",
        "HOMO-2",
        "HOMO-1",
        "HOMO",
        "LUMO",
        "LUMO+1",
        "LUMO+2",
        "LUMO+3",
        "LUMO+4",
    ]
]

descriptors.to_pickle("../descriptors/descriptors.pkl")
descriptors.to_csv("../descriptors/test.csv")
