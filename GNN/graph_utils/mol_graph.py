import rdkit.Chem as Chem
import numpy as np
import pandas as pd
import os
import tensorflow as tf

elem_list = [
    "C",
    "O",
    "N",
    "F",
    "Br",
    "Cl",
    "S",
    "Si",
    "B",
    "I",
    "K",
    "Na",
    "P",
    "Mg",
    "Li",
    "Al",
    "H",
]

atom_fdim_geo = len(elem_list) + 6 + 6 + 6 + 1

bond_fdim = 6
max_nb = 10

qm_descriptors = None
reaction_descriptors = None


def initialize_qm_descriptors(df=None, path=None):
    global qm_descriptors
    if path is not None:
        qm_descriptors = pd.read_pickle(path).set_index("smiles")
    elif df is not None:
        qm_descriptors = df


def initialize_reaction_descriptors(df=None, path=None):
    global reaction_descriptors
    if path is not None:
        reaction_descriptors = pd.read_pickle(path).set_index("smiles")
    elif df is not None:
        reaction_descriptors = df.set_index("smiles")


def get_atom_classes():
    atom_classes = {}
    token = 0
    for e in elem_list:  # element
        for d in [0, 1, 2, 3, 4, 5]:  # degree
            for ev in [1, 2, 3, 4, 5, 6]:  # explicit valence
                for iv in [0, 1, 2, 3, 4, 5]:  # inexplicit valence
                    atom_classes[str((e, d, ev, iv))] = token
                    token += 1
    return atom_classes


def rbf_expansion(expanded, mu=0, delta=0.01, kmax=8):
    k = np.arange(0, kmax)
    return np.exp(-((expanded - (mu + delta * k)) ** 2) / delta)


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(
        onek_encoding_unk(atom.GetSymbol(), elem_list)
        + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + onek_encoding_unk(atom.GetExplicitValence(), [1, 2, 3, 4, 5, 6])
        + onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
        + [atom.GetIsAromatic()],
        dtype=np.float32,
    )


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array(
        [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing(),
        ],
        dtype=np.float32,
    )


def get_non_qm_features_p(p_smiles, fatom_index_r):
    p_mol = Chem.MolFromSmiles(p_smiles)
    if not p_mol:
        raise ValueError("Could not parse smiles string:", p_smiles)

    fatom_index_p = {
        a.GetIntProp("molAtomMapNumber") - 1: a.GetIdx() for a in p_mol.GetAtoms()
    }
    fbond_index = {
        "{}-{}".format(
            *sorted(
                [
                    b.GetBeginAtom().GetIntProp("molAtomMapNumber") - 1,
                    b.GetEndAtom().GetIntProp("molAtomMapNumber") - 1,
                ]
            )
        ): b.GetIdx()
        for b in p_mol.GetBonds()
    }

    n_atoms = p_mol.GetNumAtoms()
    n_bonds = max(p_mol.GetNumBonds(), 1)
    fatoms_geo = np.zeros((n_atoms, atom_fdim_geo))
    fbonds = np.zeros((n_bonds, bond_fdim))

    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)

    for map_idx in fatom_index_p:
        fatoms_geo[fatom_index_r[map_idx], :] = atom_features(
            p_mol.GetAtomWithIdx(fatom_index_p[map_idx])
        )

    for bond in p_mol.GetBonds():
        a1i, a2i = bond.GetBeginAtom().GetIntProp(
            "molAtomMapNumber"
        ), bond.GetEndAtom().GetIntProp("molAtomMapNumber")
        idx = fbond_index["{}-{}".format(*sorted([a1i - 1, a2i - 1]))]
        a1 = fatom_index_r[a1i - 1]
        a2 = fatom_index_r[a2i - 1]

        a1i = fatom_index_p[a1i - 1]
        a2i = fatom_index_p[a2i - 1]

        if num_nbs[a1] == max_nb or num_nbs[a2] == max_nb:
            raise Exception(p_smiles)
        atom_nb[a1, num_nbs[a1]] = a2
        atom_nb[a2, num_nbs[a2]] = a1
        bond_nb[a1, num_nbs[a1]] = idx
        bond_nb[a2, num_nbs[a2]] = idx
        num_nbs[a1] += 1
        num_nbs[a2] += 1

        fbonds[idx, :6] = bond_features(bond)

    return fatoms_geo, fbonds, atom_nb, bond_nb, num_nbs


def get_non_qm_features_r(r_smiles):
    r_mol = Chem.MolFromSmiles(r_smiles)
    if not r_mol:
        raise ValueError("Could not parse smiles string:", r_mol)

    fatom_index = {
        a.GetIntProp("molAtomMapNumber") - 1: a.GetIdx() for a in r_mol.GetAtoms()
    }
    fbond_index = {
        "{}-{}".format(
            *sorted(
                [
                    b.GetBeginAtom().GetIntProp("molAtomMapNumber") - 1,
                    b.GetEndAtom().GetIntProp("molAtomMapNumber") - 1,
                ]
            )
        ): b.GetIdx()
        for b in r_mol.GetBonds()
    }

    n_atoms = r_mol.GetNumAtoms()
    n_bonds = max(r_mol.GetNumBonds(), 1)
    fatoms_geo = np.zeros((n_atoms, atom_fdim_geo))
    fbonds = np.zeros((n_bonds, bond_fdim))

    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)

    for map_idx in fatom_index:
        fatoms_geo[fatom_index[map_idx], :] = atom_features(
            r_mol.GetAtomWithIdx(fatom_index[map_idx])
        )

    for bond in r_mol.GetBonds():
        a1i, a2i = bond.GetBeginAtom().GetIntProp(
            "molAtomMapNumber"
        ), bond.GetEndAtom().GetIntProp("molAtomMapNumber")
        idx = fbond_index["{}-{}".format(*sorted([a1i - 1, a2i - 1]))]
        a1 = fatom_index[a1i - 1]
        a2 = fatom_index[a2i - 1]

        if num_nbs[a1] == max_nb or num_nbs[a2] == max_nb:
            raise Exception(r_smiles)
        atom_nb[a1, num_nbs[a1]] = a2
        atom_nb[a2, num_nbs[a2]] = a1
        bond_nb[a1, num_nbs[a1]] = idx
        bond_nb[a2, num_nbs[a2]] = idx
        num_nbs[a1] += 1
        num_nbs[a2] += 1

        fbonds[idx, :6] = bond_features(bond)

    return (
        fatoms_geo,
        fbonds,
        atom_nb,
        bond_nb,
        num_nbs,
        fatom_index,
        fbond_index,
        n_atoms,
    )


def _mol2graph(
    rs,
    selected_atom_descriptors,
    selected_bond_descriptors,
    selected_reaction_descriptors,
    ps,
    core=[],
):
    atom_fdim_qm = 20 * len(selected_atom_descriptors)
    reaction_fdim_qm = len(selected_reaction_descriptors)

    (
        fatoms_geo,
        fbonds,
        atom_nb,
        bond_nb,
        num_nbs,
        fatom_index,
        fbond_index,
        n_atoms,
    ) = get_non_qm_features_r(rs)
    fatoms_geo_p, fbonds_p, atom_nb_p, bond_nb_p, num_nbs_p = get_non_qm_features_p(
        ps, fatom_index
    )

    fatoms_qm = np.zeros((n_atoms, atom_fdim_qm))
    freaction_qm = np.zeros((reaction_fdim_qm,))
    core_mask = np.zeros((n_atoms,), dtype=np.int32)

    for smiles in rs.split("."):

        mol = Chem.MolFromSmiles(smiles)
        fatom_index_mol = {
            a.GetIntProp("molAtomMapNumber") - 1: a.GetIdx() for a in mol.GetAtoms()
        }

        if "none" not in selected_bond_descriptors:
            qm_series = qm_descriptors.loc[smiles]

            bond_index = np.expand_dims(qm_series["bond_order_matrix"], -1)
            bond_index = np.apply_along_axis(
                rbf_expansion, -1, bond_index, 0.5, 0.125, 20
            )

            bond_distance = np.expand_dims(qm_series["distance_matrix"], -1)
            bond_distance = np.apply_along_axis(
                rbf_expansion, -1, bond_distance, 0.5, 0.10, 20
            )

        if "none" not in selected_atom_descriptors:
            qm_series = qm_descriptors.loc[smiles]

            partial_charge = qm_series["partial_charge"].reshape(-1, 1)
            partial_charge = np.apply_along_axis(
                rbf_expansion, -1, partial_charge, -6.0, 0.6, 20
            )

            fukui_elec = qm_series["fukui_elec"].reshape(-1, 1)
            fukui_elec = np.apply_along_axis(
                rbf_expansion, -1, fukui_elec, -6.0, 0.6, 20
            )

            fukui_neu = qm_series["fukui_neu"].reshape(-1, 1)
            fukui_neu = np.apply_along_axis(rbf_expansion, -1, fukui_neu, -6.0, 0.6, 20)

            spin_dens = qm_series["spin_dens"].reshape(-1, 1)
            spin_dens = np.apply_along_axis(rbf_expansion, -1, spin_dens, -6.0, 0.6, 20)

            spin_dens_triplet = qm_series["spin_dens_triplet"].reshape(-1, 1)
            spin_dens_triplet = np.apply_along_axis(
                rbf_expansion, -1, spin_dens_triplet, -6.0, 0.6, 20
            )

            nmr = qm_series["NMR"].reshape(-1, 1)
            nmr = np.apply_along_axis(rbf_expansion, -1, nmr, -6.0, 0.6, 20)

            if (
                "sasa" in selected_atom_descriptors
                or "pint" in selected_atom_descriptors
            ):
                sasa = qm_series["sasa"].reshape(-1, 1)
                sasa = np.apply_along_axis(rbf_expansion, -1, sasa, -6.0, 0.6, 20)

                pint = qm_series["pint"].reshape(-1, 1)
                pint = np.apply_along_axis(rbf_expansion, -1, pint, -6.0, 0.6, 20)

            selected_atom_descriptors = list(set(selected_atom_descriptors))
            selected_atom_descriptors.sort()

            atom_qm_descriptor = None

            for descriptor in selected_atom_descriptors:
                if atom_qm_descriptor is not None:
                    atom_qm_descriptor = np.concatenate(
                        [atom_qm_descriptor, locals()[descriptor]], axis=-1
                    )
                else:
                    atom_qm_descriptor = locals()[descriptor]

        for map_idx in fatom_index_mol:
            if "none" not in selected_atom_descriptors:
                fatoms_qm[fatom_index[map_idx], :] = atom_qm_descriptor[
                    fatom_index_mol[map_idx], :
                ]
            if fatom_index[map_idx] in core:
                core_mask[fatom_index[map_idx]] = 1

        for bond in mol.GetBonds():
            a1i, a2i = bond.GetBeginAtom().GetIntProp(
                "molAtomMapNumber"
            ), bond.GetEndAtom().GetIntProp("molAtomMapNumber")
            idx = fbond_index["{}-{}".format(*sorted([a1i - 1, a2i - 1]))]
            a1 = fatom_index[a1i - 1]
            a2 = fatom_index[a2i - 1]

            a1i = fatom_index_mol[a1i - 1]
            a2i = fatom_index_mol[a2i - 1]

            if num_nbs[a1] == max_nb or num_nbs[a2] == max_nb:
                raise Exception(smiles)
            atom_nb[a1, num_nbs[a1]] = a2
            atom_nb[a2, num_nbs[a2]] = a1
            bond_nb[a1, num_nbs[a1]] = idx
            bond_nb[a2, num_nbs[a2]] = idx
            num_nbs[a1] += 1
            num_nbs[a2] += 1

            if "bond_order" in selected_bond_descriptors:
                fbonds[idx, :20] = bond_index[a1i, a2i]
            if "bond_length" in selected_bond_descriptors:
                fbonds[idx, 20:] = bond_distance[a1i, a2i]

    selected_reaction_descriptors = list(set(selected_reaction_descriptors))
    selected_reaction_descriptors.sort()

    if "none" not in selected_reaction_descriptors:
        rxns = rs + ">>" + ps
        reaction_descriptor_series = reaction_descriptors.loc[rxns]
        for idx, descriptor in enumerate(selected_reaction_descriptors):
            freaction_qm[idx] = reaction_descriptor_series[descriptor]

    return (
        fatoms_geo,
        fatoms_qm,
        fbonds,
        atom_nb,
        bond_nb,
        num_nbs,
        core_mask,
        freaction_qm,
        fatoms_geo_p,
        fbonds_p,
        atom_nb_p,
        bond_nb_p,
        num_nbs_p,
    )


def smiles2graph_pr(
    r_smiles,
    p_smiles,
    selected_atom_descriptors=["partial_charge", "fukui_elec", "fukui_neu", "nmr"],
    selected_bond_descriptors=["bond_order", "bond_length"],
    selected_reaction_descriptors=["G", "E_r", "G_alt1", "G_alt2"],
    core_buffer=0,
):
    rs_core = _get_reacting_core(r_smiles, p_smiles, core_buffer)
    features = _mol2graph(
        r_smiles,
        selected_atom_descriptors,
        selected_bond_descriptors,
        selected_reaction_descriptors,
        p_smiles,
        core=rs_core,
    )

    return features, r_smiles


def _get_reacting_core(rs, p, buffer):
    """
    use molAtomMapNumber of molecules
    buffer: neighbor to be considered as reacting center
    return: atomidx of reacting core
    """
    r_mols = Chem.MolFromSmiles(rs)
    p_mol = Chem.MolFromSmiles(p)

    rs_dict = {a.GetIntProp("molAtomMapNumber"): a for a in r_mols.GetAtoms()}
    p_dict = {a.GetIntProp("molAtomMapNumber"): a for a in p_mol.GetAtoms()}

    rs_reactants = []
    for r_smiles in rs.split("."):
        for a in Chem.MolFromSmiles(r_smiles).GetAtoms():
            if a.GetIntProp("molAtomMapNumber") in p_dict:
                rs_reactants.append(r_smiles)
                break
    rs_reactants = ".".join(rs_reactants)

    core_mapnum = set()
    for a_map in p_dict:
        a_neighbor_in_p = set(
            [a.GetIntProp("molAtomMapNumber") for a in p_dict[a_map].GetNeighbors()]
        )
        a_neighbor_in_rs = set(
            [a.GetIntProp("molAtomMapNumber") for a in rs_dict[a_map].GetNeighbors()]
        )
        if a_neighbor_in_p != a_neighbor_in_rs:
            core_mapnum.add(a_map)
        else:
            for a_neighbor in a_neighbor_in_p:
                b_in_p = p_mol.GetBondBetweenAtoms(
                    p_dict[a_neighbor].GetIdx(), p_dict[a_map].GetIdx()
                )
                b_in_r = r_mols.GetBondBetweenAtoms(
                    rs_dict[a_neighbor].GetIdx(), rs_dict[a_map].GetIdx()
                )
                if b_in_p.GetBondType() != b_in_r.GetBondType():
                    core_mapnum.add(a_map)

    core_rs = _get_buffer(r_mols, [rs_dict[a].GetIdx() for a in core_mapnum], buffer)

    fatom_index = {
        a.GetIntProp("molAtomMapNumber") - 1: a.GetIdx()
        for a in Chem.MolFromSmiles(rs_reactants).GetAtoms()
    }

    core_rs = [fatom_index[x] for x in core_rs]

    return core_rs


def _get_buffer(m, cores, buffer):
    neighbors = set(cores)

    for i in range(buffer):
        neighbors_temp = list(neighbors)
        for c in neighbors_temp:
            neighbors.update([n.GetIdx() for n in m.GetAtomWithIdx(c).GetNeighbors()])

    neighbors = [
        m.GetAtomWithIdx(x).GetIntProp("molAtomMapNumber") - 1 for x in neighbors
    ]

    return neighbors


def pack2D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i, 0:n, 0:m] = arr
    return a


def pack2D_withidx(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M, 2))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i, 0:n, 0:m, 0] = i
        a[i, 0:n, 0:m, 1] = arr
    return a


def pack1D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        a[i, 0:n] = arr
    return a


def get_mask(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        for j in range(arr.shape[0]):
            a[i][j] = 1
    return a


if __name__ == "__main__":
    graph = _mol2graph("c1cccnc1")
