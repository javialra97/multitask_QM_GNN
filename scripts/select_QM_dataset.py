import numpy as np
import pandas as pd
from time import sleep


def check_desc_avail(smiles, smiles_list):
    for molecule in smiles.split("."):
        if molecule not in smiles_list:
            return "not_avail"
        else:
            continue

    return "avail"


df = pd.read_csv("../datasets/e2_sn2_regression_eq.csv")
descriptors = pd.read_csv("../descriptors/descriptors_Lowdin.csv")

smiles_list = descriptors.smiles.values.tolist()

df["reactant_desc_avail"] = df["rxn_smiles"].apply(
    lambda x: check_desc_avail(x, smiles_list)
)

df = df[df["reactant_desc_avail"] != "not_avail"]

df2 = df[["reaction_id", "rxn_smiles", "reaction_core", "activation_energy"]]

print(df2.head())
print(len(df2))

df2.to_csv("../datasets/e2_sn2_regression_Lowdin.csv")
