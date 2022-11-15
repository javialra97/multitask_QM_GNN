import os
import pickle
import numpy as np
import pandas as pd

from GNN.WLN import construct_input_pipeline
from GNN.WLN import WLNRegressor as regressor
from process_descs import load_descriptors, setup_and_scale_descriptors
from GNN.graph_utils import initialize_qm_descriptors, initialize_reaction_descriptors

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from utils import lr_multiply_ratio, parse_args, create_logger
from utils import Dataset, split_data_cross_val
from utils import predict_single_model, write_predictions, evaluate

# initialize
args = parse_args(cross_val=True)
logger = create_logger(name=args.model_dir)
splits_dir = os.path.join(args.model_dir, 'splits')
if not os.path.isdir(splits_dir):
    os.mkdir(splits_dir)

# load data set
if args.data_path is not None:
    df = pd.read_csv(args.data_path, index_col=0)
# selective sampling
elif args.train_valid_set_path is not None and args.test_set_path is not None:
    df = pd.read_csv(args.train_valid_set_path, index_col=0)
else:
    raise Exception("Paths are not provided correctly!")

df = df.sample(frac=1, random_state=args.random_state)

# load descriptors
qmdf, df_reaction_desc = load_descriptors(args)
if isinstance(qmdf, pd.DataFrame):
    logger.info(
        f"The considered atom-level descriptors are: {args.select_atom_descriptors}"
    )
if isinstance(df_reaction_desc, pd.DataFrame):
    logger.info(
        f"The considered reaction-level descriptors are: {args.select_reaction_descriptors}"
    )

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

    # initialize list to store predictions for each model in the ensemble
    predicted_activation_energies_ind = []
    predicted_reaction_energies_ind = []

    # within a fold, loop over the ensemble size (default -> 1)
    for j in range(args.ensemble_size):
        if args.ensemble_size > 1:
            logger.info(f"Training of model {j} started...")
        train, valid, test = split_data_cross_val(
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

        # only store splits when a single model is used due to 
        # ballooning storage footprint
        if args.ensemble_size == 1:
            current_split_dir = os.path.join(splits_dir,f"fold_{i}")
            os.makedirs(current_split_dir, exist_ok=True)
            train.to_csv(os.path.join(current_split_dir, 'train.csv'))
            valid.to_csv(os.path.join(current_split_dir, 'valid.csv'))
            test.to_csv(os.path.join(current_split_dir, 'test.csv'))

        logger.info(
            f" Size train set: {len(train)} - size validation set: {len(valid)} - size test set: {len(test)}"
        )

        # process training and validation data
        train_dataset = Dataset(
            train,
            None,
            args.rxn_id_column,
            args.rxn_smiles_column,
            args.target_column1,
            args.target_column2,
        )
        valid_dataset = Dataset(
            valid,
            train_dataset.output_scalers,
            args.rxn_id_column,
            args.rxn_smiles_column,
            args.target_column1,
            args.target_column2,
        )

        # set up the atom- and reaction-level descriptors
        if isinstance(qmdf, pd.DataFrame) or isinstance(df_reaction_desc, pd.DataFrame):
            (
                qmdf_normalized,
                df_reaction_desc_normalized,
                atom_scalers,
                reaction_scalers,
            ) = setup_and_scale_descriptors(
                qmdf, df_reaction_desc, train_dataset.rxn_smiles, i
            )
            if isinstance(qmdf_normalized, pd.DataFrame):
                initialize_qm_descriptors(df=qmdf_normalized)
            if isinstance(df_reaction_desc_normalized, pd.DataFrame):
                initialize_reaction_descriptors(df=df_reaction_desc_normalized)

        # set up input pipeline for training and validation sets
        pipeline_train, _ = construct_input_pipeline(
            train_dataset,
            args.selec_batch_size,
            args.select_atom_descriptors,
            args.select_bond_descriptors,
            args.select_reaction_descriptors,
        )

        pipeline_valid, _ = construct_input_pipeline(
            valid_dataset,
            args.selec_batch_size,
            args.select_atom_descriptors,
            args.select_bond_descriptors,
            args.select_reaction_descriptors,
        )

        # set up tensorflow model
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
        opt = tf.keras.optimizers.Adam(learning_rate=args.ini_lr, clipnorm=5)
        model.compile(
            optimizer=opt,
            loss={
                "activation_energy": "mean_squared_error",
                "reaction_energy": "mean_squared_error",
            },
        )

        save_name = os.path.join(fold_dir, f"best_model_{j}.hdf5")
        checkpoint = ModelCheckpoint(
            save_name, monitor="val_loss", save_best_only=True, save_weights_only=True
        )
        reduce_lr = LearningRateScheduler(
            lr_multiply_ratio(args.ini_lr, args.lr_ratio), verbose=1
        )

        callbacks = [checkpoint, reduce_lr]

        # run training and save weights
        hist = model.fit(
            pipeline_train,
            epochs=args.selec_epochs,
            validation_data=pipeline_valid,
            callbacks=callbacks,
        )

        with open(os.path.join(fold_dir, f"history_{i}.pickle"), "wb") as hist_pickle:
            pickle.dump(hist.history, hist_pickle)

        # set up test dataset and input_pipeline; use output_scalers from train data
        test_dataset = Dataset(
            test,
            train_dataset.output_scalers,
            args.rxn_id_column,
            args.rxn_smiles_column,
            args.target_column1,
            args.target_column2,
        )

        pipeline_test, _ = construct_input_pipeline(
            test_dataset,
            args.selec_batch_size,
            args.select_atom_descriptors,
            args.select_bond_descriptors,
            args.select_reaction_descriptors,
            shuffle=False,
            predict=True,
        )

        # get predictions on test set
        (
            predicted_activation_energies_i,
            predicted_reaction_energies_i,
        ) = predict_single_model(
            pipeline_test, 
            model, 
            test_dataset.output_scalers,
            len(test_dataset),
            args.selec_batch_size,
        )

        predicted_activation_energies_ind.append(predicted_activation_energies_i)
        predicted_reaction_energies_ind.append(predicted_reaction_energies_i)

    # determine ensemble predictions
    predicted_activation_energies = np.sum(
        predicted_activation_energies_ind, axis=0
    ) / len(predicted_activation_energies_ind)

    predicted_reaction_energies = np.sum(predicted_reaction_energies_ind, axis=0) / len(
        predicted_activation_energies_ind
    )

    # write predictions for fold i to csv file
    write_predictions(
        test_dataset.rxn_id,
        predicted_activation_energies,
        predicted_reaction_energies,
        args.rxn_id_column,
        os.path.join(args.model_dir, f"test_predicted_{i}.csv"),
    )

    # compute and store metrics
    (
        rmse_activation_energy,
        rmse_reaction_energy,
        mae_activation_energy,
        mae_reaction_energy,
    ) = evaluate(
        predicted_activation_energies,
        predicted_reaction_energies,
        test_dataset.activation_energy,
        test_dataset.reaction_energy,
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
