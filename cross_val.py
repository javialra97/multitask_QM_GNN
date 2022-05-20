import os
import pickle

from GNN.WLN.data_loading import Graph_DataLoader as dataloader
from GNN.WLN.models import WLNRegressor as regressor
from process_descs import predict_atom_descs, predict_reaction_descs
from process_descs import reaction_to_reactants, normalize_atom_descs, normalize_reaction_descs
from GNN.graph_utils.mol_graph import initialize_qm_descriptors, initialize_reaction_descriptors

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
import pandas as pd
import math

from tqdm import tqdm
from utils import lr_multiply_ratio, parse_args, create_logger, scale_targets


args = parse_args(cross_val=True)

logger = create_logger(name=args.model_dir)

if "none" not in args.select_atom_descriptors or "none" not in args.select_bond_descriptors:
    if args.qm_pred:
        logger.info(f"Predicting atom-level descriptors")
        qmdf = predict_atom_descs(args, normalize=False)
    else:
        qmdf = pd.read_pickle(args.atom_desc_path)
    qmdf.to_csv(os.path.join(args.model_dir, "atom_descriptors.csv"))
    logger.info(f"The considered atom-level descriptors are: {args.select_atom_descriptors}")
    logger.info(f"The considered bond descriptors are: {args.select_bond_descriptors}")
if "none" not in args.select_reaction_descriptors:
    if args.qm_pred:
        raise NotImplementedError
    else:
        df_reaction_desc = pd.read_pickle(args.reaction_desc_path)
        df_reaction_desc.to_csv(os.path.join(args.model_dir, "reaction_descriptors"))
        logger.info(f"The considered reaction descriptors are: {args.select_reaction_descriptors}")

if args.data_path is not None:
    df = pd.read_csv(args.data_path, index_col=0)
    df = df.sample(frac=1, random_state=args.random_state)
# selective sampling
elif args.train_valid_set_path is not None and args.test_set_path is not None:
    df = pd.read_csv(args.train_valid_set_path, index_col=0)
    df = df.sample(frac=1, random_state=args.random_state)
    test = pd.read_csv(args.test_set_path, index_col=0)
else:
    raise Exception("Paths are not provided correctly!")

# split df into k_fold groups
k_fold_arange = np.linspace(0, len(df), args.k_fold + 1).astype(int)

# create lists to store metrics for each fold
rmse_activation_energy_list = []
mae_activation_energy_list = []
rmse_reaction_energy_list = []
mae_reaction_energy_list = []

for i in range(args.k_fold):
    # split data for fold
    if args.data_path is not None:
        test = df[k_fold_arange[i] : k_fold_arange[i + 1]]
        valid = df[~df.reaction_id.isin(test.reaction_id)].sample(
            frac=1 / (args.k_fold - 1), random_state=args.random_state
        )
        train = df[
            ~(
                df.reaction_id.isin(test.reaction_id)
                | df.reaction_id.isin(valid.reaction_id)
            )
        ]
    elif args.train_valid_set_path is not None:
        valid = df.sample(frac=1 / (args.k_fold - 1), random_state=args.random_state)
        train = df[~(df.reaction_id.isin(valid.reaction_id))]

    # downsample training and validation sets in case args.sample keyword has been selected
    if args.sample:
        try:
            train = train.sample(n=args.sample, random_state=args.random_state)
            valid = valid.sample(n=math.ceil(int(args.sample) / 4))
        except Exception:
            pass

    logger.info(
        f" \n Size train set: {len(train)} \n Size validation set: {len(valid)} \n Size test set: {len(test)} \n"
    )

    # process training data
    train_rxn_id = train["reaction_id"].values
    train_smiles = (
        train[f"{args.rxn_smiles_column}"].str.split(">", expand=True)[0].values
    )
    train_product = (
        train[f"{args.rxn_smiles_column}"].str.split(">", expand=True)[2].values
    )
    train_activation_energy = train[f"{args.target_column1}"].values
    train_reaction_energy = train[f"{args.target_column2}"].values

    # scale target values based on target distribution in the training set
    activation_energy_scaler = scale_targets(train_activation_energy.copy())
    reaction_energy_scaler = scale_targets(train_reaction_energy.copy())

    train_activation_energy_scaled = (
        train[f"{args.target_column1}"]
        .apply(lambda x: activation_energy_scaler.transform([[x]])[0][0])
        .values
    )
    train_reaction_energy_scaled = (
        train[f"{args.target_column2}"]
        .apply(lambda x: reaction_energy_scaler.transform([[x]])[0][0])
        .values
    ) 

    # process validation data
    valid_rxn_id = valid["reaction_id"].values
    valid_smiles = (
        valid[f"{args.rxn_smiles_column}"].str.split(">", expand=True)[0].values
    )
    valid_product = (
        valid[f"{args.rxn_smiles_column}"].str.split(">", expand=True)[2].values
    )
    valid_activation_energy = valid[f"{args.target_column1}"].values
    valid_reaction_energy = valid[f"{args.target_column2}"].values

    valid_activation_energy_scaled = (
        valid[f"{args.target_column1}"]
        .apply(lambda x: activation_energy_scaler.transform([[x]])[0][0])
        .values
    )
    valid_reaction_energy_scaled = (
        valid[f"{args.target_column2}"]
        .apply(lambda x: reaction_energy_scaler.transform([[x]])[0][0])
        .values
    )

    # set up the atom- and reaction-level descriptors
    if "none" not in args.select_atom_descriptors or "none" not in args.select_bond_descriptors:
        train_reactants = reaction_to_reactants(train["rxn_smiles"].tolist())
        qmdf_tmp, _ = normalize_atom_descs(qmdf.copy(), train_smiles=train_reactants)
        initialize_qm_descriptors(df=qmdf_tmp)
        qmdf_tmp.to_csv('scaled_atom_descs.csv')
    if "none" not in args.select_reaction_descriptors:
        df_reaction_desc_tmp, _ = normalize_reaction_descs(
            df_reaction_desc.copy(), train_smiles=train["rxn_smiles"].tolist()
        )
        initialize_reaction_descriptors(df=df_reaction_desc_tmp)
        df_reaction_desc_tmp.to_csv('scaled_reaction_descs.csv')

    # set up dataloaders for training and validation sets
    train_gen = dataloader(
        train_smiles,
        train_product,
        train_rxn_id,
        train_activation_energy_scaled,
        train_reaction_energy_scaled,
        args.selec_batch_size,
        args.select_atom_descriptors,
        args.select_bond_descriptors,
        args.select_reaction_descriptors,
    )
    train_steps = np.ceil(len(train_smiles) / args.selec_batch_size).astype(int)
    valid_gen = dataloader(
        valid_smiles,
        valid_product,
        valid_rxn_id,
        valid_activation_energy_scaled,
        valid_reaction_energy_scaled,
        args.selec_batch_size,
        args.select_atom_descriptors,
        args.select_bond_descriptors,
        args.select_reaction_descriptors,
    )
    valid_steps = np.ceil(len(valid_smiles) / args.selec_batch_size).astype(int)

    # set up tensorflow model
    model = regressor(
        args.feature,
        args.depth,
        args.select_atom_descriptors,
        args.select_reaction_descriptors,
        args.w_atom,
        args.w_reaction,
        args.depth_mol_ffn,
        args.hidden_size_multiplier
    )
    opt = tf.keras.optimizers.Adam(learning_rate=args.ini_lr, clipnorm=5)
    model.compile(
        optimizer=opt,
        loss={'activation_energy': "mean_squared_error", 'reaction_energy': "mean_squared_error"}
    )

    save_name = os.path.join(args.model_dir, f"best_model_{i}.hdf5")
    checkpoint = ModelCheckpoint(
        save_name, monitor="val_loss", save_best_only=True, save_weights_only=True
    )
    reduce_lr = LearningRateScheduler(
        lr_multiply_ratio(args.ini_lr, args.lr_ratio), verbose=1
    )

    callbacks = [checkpoint, reduce_lr]

    # run training and save weights
    print(f"training the {i}th iteration")
    hist = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=args.selec_epochs,
        validation_data=valid_gen,
        validation_steps=valid_steps,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=args.workers,
    )

    with open(os.path.join(args.model_dir, f"history_{i}.pickle"), "wb") as hist_pickle:
        pickle.dump(hist.history, hist_pickle)

    model.load_weights(save_name)

    # process testing data
    test_rxn_id = test["reaction_id"].values
    test_smiles = (
        test[f"{args.rxn_smiles_column}"].str.split(">", expand=True)[0].values
    )
    test_product = (
        test[f"{args.rxn_smiles_column}"].str.split(">", expand=True)[2].values
    )
    test_activation_energy = test[f"{args.target_column1}"].values
    test_reaction_energy = test[f"{args.target_column2}"].values

    # set up dataloader for testing data
    test_gen = dataloader(
        test_smiles,
        test_product,
        test_rxn_id,
        test_activation_energy,
        test_reaction_energy,
        args.selec_batch_size,
        args.select_atom_descriptors,
        args.select_bond_descriptors,
        args.select_reaction_descriptors,
        shuffle=False,
    )
    test_steps = np.ceil(len(test_smiles) / args.selec_batch_size).astype(int)

    # run model for testing set, save predictions and store metrics
    activation_energies_predicted = []
    reaction_energies_predicted = []
    mse_activation_energy = 0
    mse_reaction_energy = 0
    mae_activation_energy = 0
    mae_reaction_energy = 0
    
    for x, y in tqdm(test_gen, total=int(len(test_smiles) / args.selec_batch_size)):
        out = model.predict_on_batch(x)
        # activation_energy_out = np.reshape(out['activation_energy'], [-1])
        # reaction_energy_out = np.reshape(out['reaction_energy'], [-1])
        for y_output, y_true in zip(out['activation_energy'], y['activation_energy']):
            activation_energy_predicted = activation_energy_scaler.inverse_transform([y_output])[0][0]
            activation_energies_predicted.append(activation_energy_predicted)
            mae_activation_energy += abs(activation_energy_predicted - y_true) / int(len(test_smiles))
            mse_activation_energy += (activation_energy_predicted - y_true) ** 2 / int(len(test_smiles))
        for y_output, y_true in zip(out['reaction_energy'], y['reaction_energy']):
            reaction_energy_predicted = reaction_energy_scaler.inverse_transform([y_output])[0][0]
            reaction_energies_predicted.append(reaction_energy_predicted)
            mae_reaction_energy += abs(reaction_energy_predicted - y_true) / int(len(test_smiles))
            mse_reaction_energy += (reaction_energy_predicted - y_true) ** 2 / int(len(test_smiles))

    rmse_reaction_energy = np.sqrt(mse_reaction_energy)
    rmse_activation_energy = np.sqrt(mse_activation_energy)
    test_predicted = pd.DataFrame({"reaction_id": test_rxn_id, "predicted_activation_energy": activation_energies_predicted,
                        "predicted_reaction_energy": reaction_energies_predicted})
    test_predicted.to_csv(os.path.join(args.model_dir, f"test_predicted_{i}.csv"))

    rmse_activation_energy_list.append(rmse_activation_energy)
    mae_activation_energy_list.append(mae_activation_energy)
    rmse_reaction_energy_list.append(rmse_reaction_energy)
    mae_reaction_energy_list.append(mae_reaction_energy)

    # report results for current fold
    print(f"success rate for iter {i} - activation energy: {rmse_activation_energy}, {mae_activation_energy} - reaction energy: {rmse_reaction_energy}, {mae_reaction_energy}")
    logger.info(f"success rate for iter {i} - activation energy: {rmse_activation_energy}, {mae_activation_energy} - reaction energy: {rmse_reaction_energy}, {mae_reaction_energy}")

# report final results at the end of the run
print(f"RMSE for {args.k_fold}-fold cross-validation - activation energy: {np.mean(np.array(rmse_activation_energy_list))} - reaction energy: {np.mean(np.array(rmse_reaction_energy_list))}")
print(f"MAE for {args.k_fold}-fold cross-validation - activation energy: {np.mean(np.array(mae_activation_energy_list))} - reaction_energy: {np.mean(np.array(mae_reaction_energy_list))}")

logger.info(
    f"RMSE for {args.k_fold}-fold cross-validation - activation energy: {np.mean(np.array(rmse_activation_energy_list))} - reaction energy: {np.mean(np.array(rmse_reaction_energy_list))}"
)
logger.info(
    f"MAE for {args.k_fold}-fold cross-validation - activation energy: {np.mean(np.array(mae_activation_energy_list))} - reaction_energy: {np.mean(np.array(mae_reaction_energy_list))}"
)
