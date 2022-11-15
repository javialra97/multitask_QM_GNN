import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
import pandas as pd

from GNN.WLN import construct_input_pipeline
from GNN.WLN.models import WLNRegressor as regressor
from GNN.graph_utils.mol_graph import (
    initialize_qm_descriptors,
    initialize_reaction_descriptors,
)

from process_descs import load_descriptors, setup_and_scale_descriptors
from process_descs import normalize_atom_descs, normalize_reaction_descs

import pickle

from utils import lr_multiply_ratio, parse_args, create_logger
from utils import Dataset, split_data_training
from utils import predict_single_model, write_predictions

# initialize
args = parse_args()
reactivity_data = pd.read_csv(args.data_path, index_col=0)
logger = create_logger(name=args.model_dir)
splits_dir = os.path.join(args.model_dir, 'splits')
if not os.path.isdir(splits_dir):
    os.mkdir(splits_dir)

if args.predict:
    predicted_activation_energies_ind = []
    predicted_reaction_energies_ind = []

for i in range(args.ensemble_size):
    scalers_dir_path = os.path.join(args.model_dir, f"scalers_{i}")
    # training of the model
    if not args.predict:
        logger.info(f"Training of model {i} started...")
        os.makedirs(scalers_dir_path, exist_ok=True)
        train, valid, test = split_data_training(
            reactivity_data, args.rxn_id_column, args.splits, args.random_state, i
        )
        logger.info(
            f" Size train set: {len(train)} - size validation set: {len(valid)} - size test set: {len(test)}"
        )

        # save splits
        current_split_dir = os.path.join(splits_dir,f"model_{i}")
        os.makedirs(current_split_dir, exist_ok=True)
        train.to_csv(os.path.join(current_split_dir, 'train.csv'))
        valid.to_csv(os.path.join(current_split_dir, 'valid.csv'))
        test.to_csv(os.path.join(current_split_dir, 'test.csv'))

        # process the training data
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

        pickle.dump(
            train_dataset.output_scalers[0],
            open(
                os.path.join(scalers_dir_path, f"activation_energy_scaler_{i}.pickle"),
                "wb",
            ),
        )
        pickle.dump(
            train_dataset.output_scalers[1],
            open(
                os.path.join(scalers_dir_path, f"reaction_energy_scaler_{i}.pickle"),
                "wb",
            ),
        )

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

            pickle.dump(
                atom_scalers,
                open(
                    os.path.join(scalers_dir_path, f"atom_desc_scalers_{i}.pickle"),
                    "wb",
                ),
            )

            pickle.dump(
                reaction_scalers,
                open(
                    os.path.join(scalers_dir_path, f"reaction_desc_scalers_{i}.pickle"),
                    "wb",
                ),
            )

        # set up input pipeline for training and validation sets
        pipeline_train, x_build = construct_input_pipeline(
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

    else:
        test = reactivity_data

        # load output scalers
        activation_energy_scaler = pickle.load(
            open(
                os.path.join(scalers_dir_path, f"activation_energy_scaler_{i}.pickle"),
                "rb",
            )
        )
        reaction_energy_scaler = pickle.load(
            open(
                os.path.join(scalers_dir_path, f"reaction_energy_scaler_{i}.pickle"),
                "rb",
            )
        )

        # setup test dataset
        test_dataset = Dataset(
            test,
            [activation_energy_scaler, reaction_energy_scaler],
            args.rxn_id_column,
            args.rxn_smiles_column,
            None,
            None,
        )
        # load descriptors
        qmdf, df_reaction_desc = load_descriptors(args)

        # normalize descriptors
        if isinstance(qmdf, pd.DataFrame):
            logger.info(
                f"The considered atom-level descriptors are: {args.select_atom_descriptors}"
            )
            atom_scalers = pickle.load(
                open(
                    os.path.join(scalers_dir_path, f"atom_desc_scalers_{i}.pickle"),
                    "rb",
                )
            )
            qmdf_normalized, _ = normalize_atom_descs(qmdf, scalers=atom_scalers)
            initialize_qm_descriptors(df=qmdf_normalized)
        if isinstance(df_reaction_desc, pd.DataFrame):
            logger.info(
                f"The considered reaction descriptors are: {args.select_reaction_descriptors}"
            )
            reaction_scalers = pickle.load(
                open(
                    os.path.join(scalers_dir_path, f"reaction_desc_scalers_{i}.pickle"),
                    "rb",
                )
            )
            df_reaction_desc_normalized, _ = normalize_reaction_descs(
                df_reaction_desc, scalers=reaction_scalers
            )
            initialize_reaction_descriptors(df=df_reaction_desc_normalized)

        # set up pipeline for test set
        pipeline_test, x_build = construct_input_pipeline(
            test_dataset,
            args.selec_batch_size,
            args.select_atom_descriptors,
            args.select_bond_descriptors,
            args.select_reaction_descriptors,
            predict=True,
        )

    save_name = os.path.join(args.model_dir, f"best_model_{i}.hdf5")

    # set up the model for evaluation
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
        loss="mean_squared_error",
    )

    # initialize the model by running x_build
    model.predict_on_batch(x_build)
    model.summary()

    if args.restart or args.predict:
        model.load_weights(save_name)

    checkpoint = ModelCheckpoint(
        save_name, monitor="val_loss", save_best_only=True, save_weights_only=True
    )

    reduce_lr = LearningRateScheduler(
        lr_multiply_ratio(args.ini_lr, args.lr_ratio), verbose=1
    )

    callbacks = [checkpoint, reduce_lr]

    if not args.predict:
        # set up the model for training
        hist = model.fit(
            pipeline_train,
            epochs=args.selec_epochs,
            validation_data=pipeline_valid,
            callbacks=callbacks,
        )
    else:
        (
            predicted_activation_energies_i,
            predicted_reaction_energies_i,
        ) = predict_single_model(
            pipeline_test, 
            model, 
            [activation_energy_scaler, reaction_energy_scaler],
            len(test_dataset),
            args.selec_batch_size,
        )
        predicted_activation_energies_ind.append(predicted_activation_energies_i)
        predicted_reaction_energies_ind.append(predicted_reaction_energies_i)

if args.predict:
    # determine ensemble predictions
    predicted_activation_energies = np.sum(
        predicted_activation_energies_ind, axis=0
    ) / len(predicted_activation_energies_ind)
    predicted_reaction_energies = np.sum(predicted_reaction_energies_ind, axis=0) / len(
        predicted_activation_energies_ind
    )

    # write predictions to csv file
    write_predictions(
        test_dataset.rxn_id,
        predicted_activation_energies,
        predicted_reaction_energies,
        args.rxn_id_column,
        os.path.join(args.model_dir, f"{args.output_file}"),
    )
