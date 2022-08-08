import rdkit
import rdkit.Chem as Chem
import numpy as np
import pandas as pd
from time import sleep


def get_edits(smiles, preference):
    if "." not in smiles:
        return None
    # print(smiles)

    cand_beta_c = []
    configs = []
    amap = {}
    mol = Chem.MolFromSmiles(smiles)

    for atom in mol.GetAtoms():
        amap[atom.GetAtomMapNum() - 1] = atom.GetIdx()
    try:
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() == -1:
                if (
                    atom.GetSymbol() == "F"
                    or atom.GetSymbol() == "Cl"
                    or atom.GetSymbol() == "Br"
                    or atom.GetSymbol() == "H"
                ):
                    nucleophile = atom.GetAtomMapNum() - 1

        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() == 0:
                if (
                    atom.GetSymbol() == "F"
                    or atom.GetSymbol() == "Cl"
                    or atom.GetSymbol() == "Br"
                ):
                    leaving_group = atom.GetAtomMapNum() - 1

        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "C" and leaving_group in list(
                map(lambda x: x.GetAtomMapNum() - 1, atom.GetNeighbors())
            ):
                alpha_c = atom.GetAtomMapNum() - 1

        for atom in mol.GetAtoms():
            if (
                atom.GetSymbol() == "C"
                and alpha_c
                in list(map(lambda x: x.GetAtomMapNum() - 1, atom.GetNeighbors()))
                and 3.0
                not in list(map(lambda x: x.GetBondTypeAsDouble(), atom.GetBonds()))
            ):
                if (
                    len(list(map(lambda x: x.GetAtomMapNum() - 1, atom.GetNeighbors())))
                    > 1
                ):
                    beta_c = atom.GetAtomMapNum() - 1
                else:
                    cand_beta_c.append(atom.GetAtomMapNum() - 1)

        try:
            print(beta_c, cand_beta_c)
        except:
            print("\n \n \n")
            beta_c = cand_beta_c[0]

        # print(leaving_group, alpha_c, beta_c, nucleophile)

        if preference == "sn2":
            configs.append(
                get_substitution_product(leaving_group, alpha_c, nucleophile)
            )
            # configs.append(get_elimination_product(leaving_group, alpha_c, beta_c))
        elif preference == "e2":
            configs.append(
                get_elimination_product(leaving_group, alpha_c, beta_c, nucleophile)
            )
            # configs.append(get_substitution_product(leaving_group, alpha_c, nucleophile))

        return configs

    except UnboundLocalError:
        print("WTF")


def get_substitution_product(leaving_group, alpha_c, nucleophile):
    return [leaving_group, alpha_c, nucleophile]


def get_elimination_product(leaving_group, alpha_c, beta_c, nucleophile):
    return [leaving_group, alpha_c, beta_c, nucleophile]


def correct_smiles(smiles):
    if "." in smiles:
        pass
    else:
        mol = Chem.MolFromSmiles(smiles)
        number = len(list(map(lambda x: x.GetAtomMapNum() - 1, mol.GetAtoms()))) + 1
        smiles = "[Br-:" + str(number) + "]." + smiles
        # print(smiles)

    smiles1 = smiles.split(".")[0]
    smiles2 = smiles.split(".")[1]
    mol = Chem.MolFromSmiles(smiles1)
    if len(mol.GetAtoms()) < 2:
        return smiles2 + "." + smiles1
    else:
        return smiles1 + "." + smiles2


df = pd.read_csv("activation_energies_smiles_numbered_mp2_full.csv")

print(df["smiles"].head())

df["smiles"] = df["smiles"].apply(lambda x: correct_smiles(x))

df_list = df.values.tolist()
configs_list = []
problematic_cases = []
errors = 0
num_e2 = 0

for i in range(len(df_list)):
    if df_list[i][-2] == "e2":
        num_e2 += 1
        # continue
    # else:
    configs = get_edits(df_list[i][-3], df_list[i][-2])
    if configs:
        print(configs)
        try:
            if len(configs[0]) == len(set(configs[0])):
                configs_list.append([i, df_list[i][-3], configs, df_list[i][-1]])
        except:
            continue
    else:
        errors += 1
        problematic_cases.append([i, df_list[i]])
        # print(df_list[i][-2])


for i in range(len(problematic_cases)):
    print(problematic_cases[i])

print(errors)

config_df = pd.DataFrame(
    configs_list,
    columns=["reaction_id", "rxn_smiles", "bond_edits", "activation_energy"],
)

nonsensical_barriers = config_df[config_df["activation_energy"] > 1000]

print(nonsensical_barriers)

print(num_e2)
print(len(config_df))
config_df.to_csv("data_test_regression_mp2_final.csv")
