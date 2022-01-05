import rdkit.Chem as Chem
import numpy as np

elem_list = ['C', 'O', 'N', 'F', 'Br', 'Cl', 'S',
             'Si', 'B', 'I', 'K', 'Na', 'P', 'Mg', 'Li', 'Al', 'H']

atom_fdim_geo = len(elem_list) + 6 + 6 + 6 + 1
atom_fdim_qm = 10 + 10 + 10 + 10

bond_fdim_geo = 6
bond_fdim_qm = 25 + 40
max_nb = 10


def get_atom_classes():
    atom_classes = {}
    token = 0
    for e in elem_list:     #element
        for d in [0, 1, 2, 3, 4, 5]:    #degree
            for ev in [1, 2, 3, 4, 5, 6]:   #explicit valence
                for iv in [0, 1, 2, 3, 4, 5]:  #inexplicit valence
                    atom_classes[str((e, d, ev, iv))] = token
                    token += 1
    return atom_classes


def rbf_expansion(expanded, mu=0, delta=0.01, kmax=8):
    k = np.arange(0, kmax)
    return np.exp(-(expanded - (mu + delta * k))**2 / delta)


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list)
                    + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                    + onek_encoding_unk(atom.GetExplicitValence(), [1, 2, 3, 4, 5, 6])
                    + onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
                    + [atom.GetIsAromatic()], dtype=np.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array(
        [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
         bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)


def _mol2graph(rs, selected_atom_descriptors, selected_reaction_descriptors, core=[]):
    mol_rs = Chem.MolFromSmiles(rs)
    if not mol_rs:
        raise ValueError("Could not parse smiles string:", rs)

    fatom_index = {a.GetIntProp('molAtomMapNumber') - 1: a.GetIdx() for a in mol_rs.GetAtoms()}
    fbond_index = {'{}-{}'.format(*sorted([b.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1,
                                          b.GetEndAtom().GetIntProp('molAtomMapNumber') - 1])): b.GetIdx()
                   for b in mol_rs.GetBonds()}

    n_atoms = mol_rs.GetNumAtoms()
    n_bonds = max(mol_rs.GetNumBonds(), 1)
    fatoms_geo = np.zeros((n_atoms, atom_fdim_geo))
    fbonds_geo = np.zeros((n_bonds, bond_fdim_geo))

    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)
    core_mask = np.zeros((n_atoms,), dtype=np.int32)

    for smiles in rs.split('.'):

        mol = Chem.MolFromSmiles(smiles)
        fatom_index_mol = {a.GetIntProp('molAtomMapNumber') - 1: a.GetIdx() for a in mol.GetAtoms()}

        for map_idx in fatom_index_mol:
            fatoms_geo[fatom_index[map_idx], :] = atom_features(mol_rs.GetAtomWithIdx(fatom_index[map_idx]))
            if fatom_index[map_idx] in core:
                core_mask[fatom_index[map_idx]] = 1

        for bond in mol.GetBonds():
            a1i, a2i = bond.GetBeginAtom().GetIntProp('molAtomMapNumber'), \
                       bond.GetEndAtom().GetIntProp('molAtomMapNumber')
            idx = fbond_index['{}-{}'.format(*sorted([a1i-1, a2i-1]))]
            a1 = fatom_index[a1i-1]
            a2 = fatom_index[a2i-1]

            a1i = fatom_index_mol[a1i-1]
            a2i = fatom_index_mol[a2i-1]

            if num_nbs[a1] == max_nb or num_nbs[a2] == max_nb:
                raise Exception(smiles)
            atom_nb[a1, num_nbs[a1]] = a2
            atom_nb[a2, num_nbs[a2]] = a1
            bond_nb[a1, num_nbs[a1]] = idx
            bond_nb[a2, num_nbs[a2]] = idx
            num_nbs[a1] += 1
            num_nbs[a2] += 1

            fbonds_geo[idx, :] = bond_features(bond)

    return fatoms_geo, fbonds_geo, atom_nb, bond_nb, num_nbs, core_mask


def smiles2graph_pr(r_smiles, p_smiles, selected_atom_descriptors=["partial_charge", "fukui_elec", "fukui_neu", "nmr"],
                    selected_reaction_descriptors=["G", "DE_RP", "G*", "G**"], core_buffer=0):
    rs, rs_core, p_core = _get_reacting_core(r_smiles, p_smiles, core_buffer)
    rs_features = _mol2graph(r_smiles, selected_atom_descriptors, selected_reaction_descriptors, core=rs_core)

    return rs_features, r_smiles


def _get_reacting_core(rs, p, buffer):
    '''
    use molAtomMapNumber of molecules
    buffer: neighbor to be considered as reacting center
    return: atomidx of reacting core
    '''
    r_mols = Chem.MolFromSmiles(rs)
    p_mol = Chem.MolFromSmiles(p)

    rs_dict = {a.GetIntProp('molAtomMapNumber'): a for a in r_mols.GetAtoms()}
    p_dict = {a.GetIntProp('molAtomMapNumber'): a for a in p_mol.GetAtoms()}

    rs_reactants = []
    for r_smiles in rs.split('.'):
        for a in Chem.MolFromSmiles(r_smiles).GetAtoms():
            if a.GetIntProp('molAtomMapNumber') in p_dict:
                rs_reactants.append(r_smiles)
                break
    rs_reactants = '.'.join(rs_reactants)

    core_mapnum = set()
    for a_map in p_dict:
        # FIXME chiral change
        # if str(p_dict[a_map].GetChiralTag()) != str(rs_dict[a_map].GetChiralTag()):
        #    core_mapnum.add(a_map)

        a_neighbor_in_p = set([a.GetIntProp('molAtomMapNumber') for a in p_dict[a_map].GetNeighbors()])
        a_neighbor_in_rs = set([a.GetIntProp('molAtomMapNumber') for a in rs_dict[a_map].GetNeighbors()])
        if a_neighbor_in_p != a_neighbor_in_rs:
            core_mapnum.add(a_map)
        else:
            for a_neighbor in a_neighbor_in_p:
                b_in_p = p_mol.GetBondBetweenAtoms(p_dict[a_neighbor].GetIdx(), p_dict[a_map].GetIdx())
                b_in_r = r_mols.GetBondBetweenAtoms(rs_dict[a_neighbor].GetIdx(), rs_dict[a_map].GetIdx())
                if b_in_p.GetBondType() != b_in_r.GetBondType():
                    core_mapnum.add(a_map)

    core_rs = _get_buffer(r_mols, [rs_dict[a].GetIdx() for a in core_mapnum], buffer)
    core_p = _get_buffer(p_mol, [p_dict[a].GetIdx() for a in core_mapnum], buffer)

    fatom_index = \
        {a.GetIntProp('molAtomMapNumber') - 1: a.GetIdx() for a in Chem.MolFromSmiles(rs_reactants).GetAtoms()}

    core_rs = [fatom_index[x] for x in core_rs]
    core_p = [fatom_index[x] for x in core_p]

    return rs_reactants, core_rs, core_p


def _get_reacting_core(rs, p, buffer):
    '''
    use molAtomMapNumber of molecules
    buffer: neighbor to be considered as reacting center
    return: atomidx of reacting core
    '''
    r_mols = Chem.MolFromSmiles(rs)
    p_mol = Chem.MolFromSmiles(p)

    rs_dict = {a.GetIntProp('molAtomMapNumber'): a for a in r_mols.GetAtoms()}
    try:
        p_dict = {a.GetIntProp('molAtomMapNumber'): a for a in p_mol.GetAtoms()}
    except:
        print(p)
        for a in p_mol.GetAtoms():
            print(a, a.GetIntProp('molAtomMapNumber'))

    rs_reactants = []
    for r_smiles in rs.split('.'):
        for a in Chem.MolFromSmiles(r_smiles).GetAtoms():
            if a.GetIntProp('molAtomMapNumber') in p_dict:
                rs_reactants.append(r_smiles)
                break
    rs_reactants = '.'.join(rs_reactants)

    core_mapnum = set()
    for a_map in p_dict:
        # FIXME chiral change
        # if str(p_dict[a_map].GetChiralTag()) != str(rs_dict[a_map].GetChiralTag()):
        #    core_mapnum.add(a_map)

        a_neighbor_in_p = set([a.GetIntProp('molAtomMapNumber') for a in p_dict[a_map].GetNeighbors()])
        a_neighbor_in_rs = set([a.GetIntProp('molAtomMapNumber') for a in rs_dict[a_map].GetNeighbors()])
        if a_neighbor_in_p != a_neighbor_in_rs:
            core_mapnum.add(a_map)
        else:
            for a_neighbor in a_neighbor_in_p:
                b_in_p = p_mol.GetBondBetweenAtoms(p_dict[a_neighbor].GetIdx(), p_dict[a_map].GetIdx())
                b_in_r = r_mols.GetBondBetweenAtoms(rs_dict[a_neighbor].GetIdx(), rs_dict[a_map].GetIdx())
                if b_in_p.GetBondType() != b_in_r.GetBondType():
                    core_mapnum.add(a_map)

    core_rs = _get_buffer(r_mols, [rs_dict[a].GetIdx() for a in core_mapnum], buffer)
    core_p = _get_buffer(p_mol, [p_dict[a].GetIdx() for a in core_mapnum], buffer)

    fatom_index = \
        {a.GetIntProp('molAtomMapNumber') - 1: a.GetIdx() for a in Chem.MolFromSmiles(rs_reactants).GetAtoms()}

    core_rs = [fatom_index[x] for x in core_rs]
    core_p = [fatom_index[x] for x in core_p]

    return rs_reactants, core_rs, core_p


def _get_buffer(m, cores, buffer):
    neighbors = set(cores)

    for i in range(buffer):
        neighbors_temp = list(neighbors)
        for c in neighbors_temp:
            neighbors.update([n.GetIdx() for n in m.GetAtomWithIdx(c).GetNeighbors()])

    neighbors = [m.GetAtomWithIdx(x).GetIntProp('molAtomMapNumber') - 1 for x in neighbors]

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


def smiles2graph_list(smiles_list, idxfunc=lambda x: x.GetIdx()):
    res = list(map(lambda x: smiles2graph(x, idxfunc), smiles_list))
    fatom_list, fbond_list, gatom_list, gbond_list, nb_list = zip(*res)
    return pack2D(fatom_list), pack2D(fbond_list), pack2D_withidx(gatom_list), pack2D_withidx(gbond_list), pack1D(
        nb_list), get_mask(fatom_list)


def get_bond_edits(reactant_smi, product_smi):
    reactants = Chem.MolFromSmiles(reactant_smi)
    products = Chem.MolFromSmiles(product_smi)
    conserved_maps = [a.GetAtomMapNum() for a in reactants.GetAtoms() if a.GetAtomMapNum()]
    bond_changes = set()

    bonds_prev = {}
    for bond in reactants.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()])
        bonds_prev['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
    bonds_new = {}
    for bond in products.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()])
        if (nums[0] not in conserved_maps) or (nums[1] not in conserved_maps): continue
        bonds_new['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()

    for bond in bonds_prev:
        if bond not in bonds_new:
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], 0.0))  # lost bond
        else:
            if bonds_prev[bond] != bonds_new[bond]:
                bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))  # changed bond
    for bond in bonds_new:
        if bond not in bonds_prev:
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))  # new bond

    return bond_changes


if __name__ == "__main__":
    # np.set_printoptions(threshold='nan')
    #"[Br:1][Br:2].[OH:3][c:4]1[cH:5][cH:6][cH:7][cH:8][c:9]1[F:10]")
    #print(graph)
    #print(smiles2graph_pr("[Br:1][c:5]1[c:4]([OH:3])[c:9]([F:10])[cH:8][cH:7][cH:6]1", "[Br:1][Br:2].[OH:3][c:4]1[cH:5][cH:6][cH:7][cH:8][c:9]1[F:10]", core_buffer=0))
    #print(smiles2graph_list(["[Br:2][c:5]1[c:4]([OH:3])[c:9]([F:10])[cH:8][cH:7][cH:6]1", "[Br:1][Br:2].[OH:3][c:4]1[cH:5][cH:6][cH:7][cH:8][c:9]1[F:10"]))
    graph = smiles2graph_pr("[CH3:1][C@@H:2]([NH2:3])[CH2:4][Cl:5].[F-:6]", "[3, 4, 1", core_buffer=0)
