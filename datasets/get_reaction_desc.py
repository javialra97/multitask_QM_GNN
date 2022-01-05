import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import Chem
import pickle

df = pd.read_csv("final_dataset_mol_desc_mos.csv")

df["smiles"] = df["rxn_smiles"].apply(lambda x: x.split(">")[0] + ">>" + x.split(">")[-1])

df = df[["smiles", "G", "DE_RP", "G_alt1", "G_alt2"]]

df = df.rename(columns={"G_alt1": "G*", "G_alt2": "G**"})

print(df.head())

df.to_csv("reaction_desc_mos.csv")
df.to_pickle("reaction_desc_mos.pickle")
