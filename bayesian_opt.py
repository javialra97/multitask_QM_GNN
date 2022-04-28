import os
import pickle
import argparse

from GNN.WLN.data_loading import Graph_DataLoader as dataloader
from GNN.WLN.models import WLNRegressor as regressor
from process_descs import predict_atom_descs, predict_reaction_descs
from process_descs import min_max_normalize_atom_descs, reaction_to_reactants, min_max_normalize_reaction_descs
from GNN.graph_utils.mol_graph import initialize_qm_descriptors, initialize_reaction_descriptors

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
import pandas as pd

from types import SimpleNamespace
from tqdm import tqdm
from utils import lr_multiply_ratio, create_logger, scale_targets

from functools import partial
from hyperopt import fmin, hp, tpe

"""Optimizes hyperparameters using Bayesian optimization."""

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=None, type=str,
                        help='Path to reaction data')
    parser.add_argument('--model_dir', default='bayesian_opt',
                        help='path to the checkpoint file of the trained model')
    parser.add_argument('--atom_desc_path', default=None,
                        help='path to the file storing the atom-condensed descriptors (must be provided when using QM_GNN model)')
    parser.add_argument('--reaction_desc_path', default=None,
                        help='path to the file storing the reaction descriptors (must be provided when using QM_GNN model)')
    parser.add_argument('--select_atom_descriptors', nargs='+',
                        default=['partial_charge', 'fukui_elec', 'fukui_neu', 'nmr'],
                        help='(Optional) Selection of atom-condensed descriptors to feed to the (ml_QM_)GNN model')
    parser.add_argument('--select_reaction_descriptors', nargs='+',
                        default=['G', 'G_alt1', 'G_alt2'],
                        help='(Optional) Selection of reaction descriptors to feed to the (ml_)QM_GNN model')
    parser.add_argument('--select_bond_descriptors', nargs='+', default=['none'],
                        help='(Optional) Selection of bond descriptors to feed to the (ml_)QM_GNN model')

    cl_args = parser.parse_args()
    return cl_args


def objective(args0, df_reactions, rxn_smiles_column, target_column1, target_column2, model_dir, qmdf, df_reaction_desc, select_atom_descriptors, 
            select_bond_descriptors, select_reaction_descriptors, logger, k_fold, selec_batch_size):
    
    args = SimpleNamespace(**args0)

    k_fold_arange = np.linspace(0, len(df_reactions), k_fold + 1).astype(int)

    rmse_activation_energy_list = []
    rmse_reaction_energy_list = []

    for i in range(k_fold):
        # split data for fold
        if df_reactions is not None:
            valid = df_reactions[k_fold_arange[i] : k_fold_arange[i + 1]]
            train = df_reactions[~df_reactions.reaction_id.isin(valid.reaction_id)]

        # process training data
        train_rxn_id = train["reaction_id"].values
        train_smiles = (
            train[f"{rxn_smiles_column}"].str.split(">", expand=True)[0].values
        )
        train_product = (
            train[f"{rxn_smiles_column}"].str.split(">", expand=True)[2].values
        )
        train_activation_energy = train[f"{target_column1}"].values
        train_reaction_energy = train[f"{target_column2}"].values

        # scale target values based on target distribution in the training set
        activation_energy_scaler = scale_targets(train_activation_energy.copy())
        reaction_energy_scaler = scale_targets(train_reaction_energy.copy())

        train_activation_energy_scaled = (
            train[f"{target_column1}"]
            .apply(lambda x: activation_energy_scaler.transform([[x]])[0][0])
            .values
        )
        train_reaction_energy_scaled = (
            train[f"{target_column2}"]
            .apply(lambda x: reaction_energy_scaler.transform([[x]])[0][0])
            .values
        ) 

        # process validation data
        valid_rxn_id = valid["reaction_id"].values
        valid_smiles = (
            valid[f"{rxn_smiles_column}"].str.split(">", expand=True)[0].values
        )
        valid_product = (
            valid[f"{rxn_smiles_column}"].str.split(">", expand=True)[2].values
        )

        valid_activation_energy_scaled = (
            valid[f"{target_column1}"]
            .apply(lambda x: activation_energy_scaler.transform([[x]])[0][0])
            .values
        )
        valid_reaction_energy_scaled = (
            valid[f"{target_column2}"]
            .apply(lambda x: reaction_energy_scaler.transform([[x]])[0][0])
            .values
        )

        if "none" not in select_atom_descriptors or "none" not in select_bond_descriptors:
            train_reactants = reaction_to_reactants(train["rxn_smiles"].tolist())
            qmdf_temp, _ = min_max_normalize_atom_descs(qmdf.copy(), train_smiles=train_reactants)
            initialize_qm_descriptors(df=qmdf_temp)
        if "none" not in select_reaction_descriptors:
            df_reaction_desc_temp, _ = min_max_normalize_reaction_descs(
                df_reaction_desc.copy(), train_smiles=train["rxn_smiles"].tolist()
            )
            initialize_reaction_descriptors(df=df_reaction_desc_temp)

        # set up dataloaders for training and validation sets
        train_gen = dataloader(
            train_smiles,
            train_product,
            train_rxn_id,
            train_activation_energy_scaled,
            train_reaction_energy_scaled,
            selec_batch_size,
            select_atom_descriptors,
            select_bond_descriptors,
            select_reaction_descriptors,
        )
        train_steps = np.ceil(len(train_smiles) / selec_batch_size).astype(int)
        valid_gen = dataloader(
            valid_smiles,
            valid_product,
            valid_rxn_id,
            valid_activation_energy_scaled,
            valid_reaction_energy_scaled,
            selec_batch_size,
            select_atom_descriptors,
            select_bond_descriptors,
            select_reaction_descriptors,
        )
        valid_steps = np.ceil(len(valid_smiles) / selec_batch_size).astype(int)

        # set up tensorflow model
        model = regressor(
            50,
            int(args.depth),
            select_atom_descriptors,
            select_reaction_descriptors,
            args.w_atom,
            args.w_reaction,
            int(args.depth_mol_ffn),
            int(args.hidden_size_multiplier)
        )

        opt = tf.keras.optimizers.Adam(learning_rate=args.ini_lr, clipnorm=5)
        model.compile(
            optimizer=opt,
            loss={'activation_energy': "mean_squared_error", 'reaction_energy': "mean_squared_error"},
        )

        save_name = os.path.join(model_dir, f"best_model_{i}.hdf5")
        checkpoint = ModelCheckpoint(
            save_name, monitor="val_loss", save_best_only=True, save_weights_only=True
        )
        reduce_lr = LearningRateScheduler(
            lr_multiply_ratio(args.ini_lr, args.lr_ratio), verbose=0
        )

        callbacks = [checkpoint, reduce_lr]

        # run training and save weights
        print(f"training the {i}th iteration")
        hist = model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            epochs=100,
            validation_data=valid_gen,
            validation_steps=valid_steps,
            callbacks=callbacks,
            use_multiprocessing=True,
            workers=6,
            verbose=0
        )

        with open(os.path.join(model_dir, f"history_{i}.pickle"), "wb") as hist_pickle:
            pickle.dump(hist.history, hist_pickle)

        model.load_weights(save_name)

        activation_energies_predicted = []
        reaction_energies_predicted = []
        mse_activation_energy = 0
        mse_reaction_energy = 0

        for x, y in tqdm(valid_gen, total=int(len(valid_smiles) / selec_batch_size)):
            out = model.predict_on_batch(x)
            for y_output, y_true in zip(out['activation_energy'], y['activation_energy']):
                activation_energy_predicted = activation_energy_scaler.inverse_transform([y_output])[0][0]
                y_true_unscaled = activation_energy_scaler.inverse_transform([[y_true]])[0][0]
                activation_energies_predicted.append(activation_energy_predicted)
                mse_activation_energy += (activation_energy_predicted - y_true_unscaled) ** 2 / int(len(valid_smiles))
            for y_output, y_true in zip(out['reaction_energy'], y['reaction_energy']):
                reaction_energy_predicted = reaction_energy_scaler.inverse_transform([y_output])[0][0]
                y_true_unscaled = reaction_energy_scaler.inverse_transform([[y_true]])[0][0]
                reaction_energies_predicted.append(reaction_energy_predicted)
                mse_reaction_energy += (reaction_energy_predicted - y_true_unscaled) ** 2 / int(len(valid_smiles))

        rmse_reaction_energy = np.sqrt(mse_reaction_energy)
        rmse_activation_energy = np.sqrt(mse_activation_energy)

        rmse_activation_energy_list.append(rmse_activation_energy)
        rmse_reaction_energy_list.append(rmse_reaction_energy)

    logger.info(f"Selected Hyperparameters: {args0}")
    logger.info(f"RMSE amounts to - activation energy: {np.mean(np.array(rmse_activation_energy_list))} - reaction energy: {np.mean(np.array(rmse_reaction_energy_list))} \n")
    
    return np.mean(np.array(rmse_activation_energy_list))


def gnn_bayesian(data_path, random_state, model_dir, qm_pred, atom_desc_path, 
                reaction_desc_path, select_atom_descriptors, select_bond_descriptors, 
                select_reaction_descriptors, logger):
    
    df = pd.read_csv(data_path, index_col=0)
    df = df.sample(frac=0.8, random_state=random_state) 

    if "none" not in select_atom_descriptors or "none" not in select_bond_descriptors:
        if qm_pred:
            # TODO: complete this
            qmdf = predict_atom_descs(None, normalize=False)
        else:
            qmdf = pd.read_pickle(atom_desc_path)

    else:
        qmdf = None
    if "none" not in select_reaction_descriptors:
        if qm_pred:
            raise NotImplementedError
        else:
            df_reaction_desc = pd.read_pickle(reaction_desc_path)
    else:
        df_reaction_desc = None

    space = {
        'depth': hp.quniform('depth', low=2, high=6, q=1),
        'w_atom': hp.quniform('w_atom', low=1, high=7, q=0.5),
        'w_reaction': hp.quniform('w_reaction', low=1, high=7, q=0.5),
        'ini_lr': hp.loguniform('ini_lr', low=-10, high=-5),
        'lr_ratio': hp.quniform('lr_ratio', low=0.9, high=0.99, q=0.01),
        'depth_mol_ffn': hp.quniform('depth_mol_ffn', low=1, high=4, q=1),
        'hidden_size_multiplier': hp.quniform('hidden_size_multiplier', low=0, high=20, q=10)
    }

    fmin_objective = partial(objective, df_reactions=df,  rxn_smiles_column='rxn_smiles', target_column1='DG_TS', target_column2='G_r',
                        model_dir=model_dir, qmdf=qmdf, df_reaction_desc=df_reaction_desc, select_atom_descriptors=select_atom_descriptors, 
                        select_bond_descriptors=select_bond_descriptors, select_reaction_descriptors=select_reaction_descriptors, logger=logger,
                        k_fold=4, selec_batch_size=10)    

    best = fmin(fmin_objective, space, algo=tpe.suggest, max_evals=64)
    logger.info(best)


if __name__ == '__main__':
    cl_args = parse_command_line_args()
    if not os.path.isdir(cl_args.model_dir):
        os.mkdir(cl_args.model_dir)
    logger = create_logger(cl_args.model_dir)
    gnn_bayesian(cl_args.data_path, 1, cl_args.model_dir, False, 
                cl_args.atom_desc_path, cl_args.reaction_desc_path, cl_args.select_atom_descriptors, 
                cl_args.select_bond_descriptors, cl_args.select_reaction_descriptors, logger)
