import os
import pickle
import numpy as np
import pandas as pd

from GNN.WLN.data_loading import Graph_DataLoader as dataloader
from GNN.WLN.models import WLNRegressor as regressor
from process_descs import predict_atom_descs, predict_reaction_descs
from process_descs import (
    reaction_to_reactants,
    normalize_atom_descs,
    normalize_reaction_descs,
)
from GNN.graph_utils.mol_graph import (
    initialize_qm_descriptors,
    initialize_reaction_descriptors,
)

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from utils import lr_multiply_ratio, parse_args, create_logger
from utils import Dataset, split_data
from utils import evaluate, write_predictions


def initialize_model(args):
    """Initialize a model.

    Args:
        args (Namespace): The namespace containing all the command line arguments

    Returns:
        model (WLNRegressor): An initialized model
    """
    model = regressor(
        args.feature,
        args.depth,
        args.select_atom_descriptors,
        args.select_reaction_descriptors,
        args.w_atom,
        args.w_reaction,
        args.depth_mol_ffn,
        args.hidden_size_multiplier,
    )

    return model


def load_all_models(args, fold_dir, dataloader):
    """Load an ensemble of pre-trained models.

    Args:
        args (Namespace): The namespace containing all the command line arguments
        fold_dir (os.Path): Path of the directory for each fold
        dataloader (Dataloader): A dataloader object so that the models can be build

    Returns:
        all_models (List[WLNRegressor]): List of loaded models
    """
    all_models = []
    for j in range(args.ensemble_size):
        # set up the model
        model = initialize_model(args)
        x_build = dataloader[0][0]
        model.predict_on_batch(x_build)
        # load weights
        save_name = os.path.join(fold_dir, f"best_model_{j}.hdf5")
        model.load_weights(save_name)
        all_models.append(model)

    return all_models


args = parse_args(cross_val=True)

logger = create_logger(name=args.model_dir)

if (
    "none" not in args.select_atom_descriptors
    or "none" not in args.select_bond_descriptors
):
    if args.qm_pred:
        logger.info(f"Predicting atom-level descriptors")
        qmdf = predict_atom_descs(args, normalize=False)
    else:
        qmdf = pd.read_pickle(args.atom_desc_path)
    qmdf.to_csv(os.path.join(args.model_dir, "atom_descriptors.csv"))
    logger.info(
        f"The considered atom-level descriptors are: {args.select_atom_descriptors}"
    )
    logger.info(f"The considered bond descriptors are: {args.select_bond_descriptors}")
if "none" not in args.select_reaction_descriptors:
    if args.qm_pred:
        raise NotImplementedError
    else:
        df_reaction_desc = pd.read_pickle(args.reaction_desc_path)
        df_reaction_desc.to_csv(os.path.join(args.model_dir, "reaction_descriptors"))
        logger.info(
            f"The considered reaction descriptors are: {args.select_reaction_descriptors}"
        )

if args.data_path is not None:
    df = pd.read_csv(args.data_path, index_col=0)
# selective sampling
elif args.train_valid_set_path is not None and args.test_set_path is not None:
    df = pd.read_csv(args.train_valid_set_path, index_col=0)
else:
    raise Exception("Paths are not provided correctly!")
df = df.sample(frac=1, random_state=args.random_state)

# split df into k_fold groups
k_fold_arange = np.linspace(0, len(df), args.k_fold + 1).astype(int)

# create lists to store metrics for each fold
rmse_activation_energy_list, rmse_reaction_energy_list = [], []
mae_activation_energy_list, mae_reaction_energy_list = [], []

# loop over all the folds
for i in range(args.k_fold):
    logger.info(f"Training the {i}th iteration...")
    # make a directory to store model files
    fold_dir = os.path.join(args.model_dir, f"fold_{i}")
    os.makedirs(fold_dir, exist_ok=True)
    all_scalers = []
    # within a fold, loop over the ensemble size (default -> 1)
    for j in range(args.ensemble_size):
        logger.info(f"Training of model {j} started...")
        train, valid, test = split_data(
            df,
            k_fold_arange,
            i,
            j,
            args.rxn_id_column,
            args.data_path,
            args.train_valid_set_path,
            args.sample,
            args.k_fold,
            args.random_state,
            args.test_set_path,
        )

        logger.info(
            f" Size train set: {len(train)} - Size validation set: {len(valid)} - Size test set: {len(test)}"
        )

        # process training and validation data
        train_dataset = Dataset(train, args)
        valid_dataset = Dataset(valid, args, train_dataset.scalers)

        # set up the atom- and reaction-level descriptors
        if (
            "none" not in args.select_atom_descriptors
            or "none" not in args.select_bond_descriptors
        ):
            train_reactants = reaction_to_reactants(train["rxn_smiles"].tolist())
            qmdf_tmp, _ = normalize_atom_descs(
                qmdf.copy(), train_smiles=train_reactants
            )
            initialize_qm_descriptors(df=qmdf_tmp)
        if "none" not in args.select_reaction_descriptors:
            df_reaction_desc_tmp, _ = normalize_reaction_descs(
                df_reaction_desc.copy(), train_smiles=train["rxn_smiles"].tolist()
            )
            initialize_reaction_descriptors(df=df_reaction_desc_tmp)

        # set up dataloaders for training and validation sets
        train_gen = dataloader(
            train_dataset,
            args.selec_batch_size,
            args.select_atom_descriptors,
            args.select_bond_descriptors,
            args.select_reaction_descriptors,
        )
        train_steps = np.ceil(len(train_dataset) / args.selec_batch_size).astype(int)
        valid_gen = dataloader(
            valid_dataset,
            args.selec_batch_size,
            args.select_atom_descriptors,
            args.select_bond_descriptors,
            args.select_reaction_descriptors,
        )
        valid_steps = np.ceil(len(valid_dataset) / args.selec_batch_size).astype(int)

        # set up tensorflow model
        model = initialize_model(args)
        opt = tf.keras.optimizers.Adam(learning_rate=args.ini_lr, clipnorm=5)
        model.compile(
            optimizer=opt,
            loss={
                "activation_energy": "mean_squared_error",
                "reaction_energy": "mean_squared_error",
            },
        )

        save_name = os.path.join(
            os.path.join(args.model_dir, f"fold_{i}"), f"best_model_{j}.hdf5"
        )
        checkpoint = ModelCheckpoint(
            save_name, monitor="val_loss", save_best_only=True, save_weights_only=True
        )
        reduce_lr = LearningRateScheduler(
            lr_multiply_ratio(args.ini_lr, args.lr_ratio), verbose=1
        )

        callbacks = [checkpoint, reduce_lr]

        # run training and save weights
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

        with open(
            os.path.join(
                os.path.join(args.model_dir, f"fold_{i}"), f"history_{i}.pickle"
            ),
            "wb",
        ) as hist_pickle:
            pickle.dump(hist.history, hist_pickle)

        all_scalers.append(train_dataset.scalers)

    # process testing data
    test_dataset = Dataset(test, args, train_dataset.scalers)

    # set up dataloader for testing data
    test_gen = dataloader(
        test_dataset,
        args.selec_batch_size,
        args.select_atom_descriptors,
        args.select_bond_descriptors,
        args.select_reaction_descriptors,
        shuffle=False,
    )
    test_steps = np.ceil(len(test_dataset) / args.selec_batch_size).astype(int)

    # load the models
    all_models = load_all_models(args, fold_dir, train_gen)

    # run model for testing set, save predictions and store metrics
    (
        predicted_activation_energies,
        predicted_reaction_energies,
        rmse_activation_energy,
        rmse_reaction_energy,
        mae_activation_energy,
        mae_reaction_energy,
    ) = evaluate(test_gen, args.selec_batch_size, all_models, all_scalers)

    write_predictions(
        test_gen,
        predicted_activation_energies,
        predicted_reaction_energies,
        args.rxn_id_column,
        os.path.join(args.model_dir, f"test_predicted_{i}.csv"),
    )

    rmse_activation_energy_list.append(rmse_activation_energy)
    mae_activation_energy_list.append(mae_activation_energy)
    rmse_reaction_energy_list.append(rmse_reaction_energy)
    mae_reaction_energy_list.append(mae_reaction_energy)

    logger.info(
        f"success rate for iter {i} - activation energy: {rmse_activation_energy}, {mae_activation_energy}"
        f" - reaction energy: {rmse_reaction_energy}, {mae_reaction_energy}"
    )

# report final results at the end of the run
logger.info(
    f"RMSE for {args.k_fold}-fold cross-validation - activation energy: "
    f"{np.mean(np.array(rmse_activation_energy_list))} - reaction energy: {np.mean(np.array(rmse_reaction_energy_list))}"
    f"\nMAE for {args.k_fold}-fold cross-validation - activation energy: "
    f"{np.mean(np.array(mae_activation_energy_list))} - reaction_energy: {np.mean(np.array(mae_reaction_energy_list))}"
)
