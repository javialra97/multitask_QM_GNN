import os
import pickle

import pandas as pd
from rdkit import Chem
from qmdesc import ReactivityDescriptorHandler
from tqdm import tqdm

from .post_process import check_chemprop_out, normalize_atom_descs, normalize_reaction_descs, reaction_to_reactants


def predict_atom_descs(args, normalize=True):
    # predict descriptors for reactants in the reactions
    reactivity_data = pd.read_csv(args.data_path, index_col=0)
    reactants = reaction_to_reactants(reactivity_data['smiles'].tolist())

    print('Predicting descriptors for reactants...')

    handler = ReactivityDescriptorHandler()
    descs = []
    for smiles in tqdm(reactants):
        descs.append(handler.predict(smiles))

    df = pd.DataFrame(descs)

    invalid = check_chemprop_out(df)
    # FIXME remove invalid molecules from reaction dataset
    print(invalid)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    df.to_pickle(os.path.join(args.output_dir, 'reactants_descriptors.pickle'))
    save_dir = args.model_dir

    if not normalize:
        return df

    if not args.predict:
        df, scalers = normalize_atom_descs(df)
        pickle.dump(scalers, open(os.path.join(save_dir, 'scalers.pickle'), 'wb'))
    else:
        scalers = pickle.load(open(os.path.join(save_dir, 'scalers.pickle'), 'rb'))
        df, _ = normalize_atom_descs(df, scalers=scalers)

    df.to_pickle(os.path.join(args.output_dir, 'reactants_descriptors_norm.pickle'))

    return df


def predict_reaction_descs(args, normalize=True):
    pass
