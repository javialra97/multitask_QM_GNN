import os
import pickle
from sqlite3 import DataError

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
import pandas as pd
import math

from tqdm import tqdm
from utils import lr_multiply_ratio, parse_args, create_logger, scale_targets

args, dataloader, regressor = parse_args(cross_val=True)

logger = create_logger(name=args.model_dir)

if args.model == 'ml_QM_GNN':
    from ml_QM_GNN.graph_utils.mol_graph import initialize_qm_descriptors, initialize_reaction_descriptors
    from predict_desc.predict_desc import predict_desc, predict_reaction_desc, reaction_to_reactants
    from predict_desc.post_process import min_max_normalize, min_max_normalize_reaction
    qmdf = predict_desc(args, normalize=False)
    qmdf.to_csv(os.path.join(args.model_dir, 'atom_descriptors.csv'))
    logger.info(f'The considered atom-condensed descriptors are: {args.select_atom_descriptors}')
    logger.info(f'The considered bond descriptors are: {args.select_bond_descriptors}')
    df_reaction_desc = None
    #df_reaction_desc = predict_reaction_desc(args, normalize=False)
    #df_reaction_desc.to_csv(os.path.join(args.model_dir, 'reaction_descriptors'))
    logger.info(f'The considered reaction descriptors are: {args.select_reaction_descriptors}')
else:
    if args.model == 'QM_GNN':
        from QM_GNN.graph_utils.mol_graph import initialize_qm_descriptors, initialize_reaction_descriptors
        from process_desc.post_process import min_max_normalize,reaction_to_reactants, min_max_normalize_reaction
        qmdf = pd.read_pickle(args.atom_desc_path)
        qmdf.to_csv(os.path.join(args.model_dir, 'atom_descriptors.csv'))
        logger.info(f'The considered atom-condensed descriptors are: {args.select_atom_descriptors}')
        logger.info(f'The considered bond descriptors are: {args.select_bond_descriptors}')
        df_reaction_desc = pd.read_pickle(args.reaction_desc_path)
        df_reaction_desc.to_csv(os.path.join(args.model_dir, 'reaction_descriptors'))
        logger.info(f'The considered reaction descriptors are: {args.select_reaction_descriptors}')

if args.data_path is not None:
    df = pd.read_csv(args.data_path, index_col=0)
    df = df.sample(frac=1, random_state=args.random_state)
# selective sampling
elif args.train_valid_set_path is not None and args.test_set_path is not None:
    df = pd.read_csv(args.train_valid_set_path, index_col=0)
    df = df.sample(frac=1, random_state=args.random_state)
    test = pd.read_csv(args.test_set_path, index_col=0)
else:
    raise Exception('Paths are not provided correctly!')


# split df into k_fold groups
k_fold_arange = np.linspace(0, len(df), args.k_fold+1).astype(int)

# create lists to store metrics for each fold
rmse_list = []
mae_list = []

for i in range(args.k_fold):
    # split data for fold
    if args.data_path is not None:
        test = df[k_fold_arange[i]:k_fold_arange[i+1]]
        valid = df[~df.reaction_id.isin(test.reaction_id)].sample(frac=1/(args.k_fold-1), random_state=args.random_state)
        train = df[~(df.reaction_id.isin(test.reaction_id) | df.reaction_id.isin(valid.reaction_id))]
    elif args.train_valid_set_path is not None:
        valid = df.sample(frac=1/(args.k_fold-1), random_state=args.random_state)
        train = df[~(df.reaction_id.isin(valid.reaction_id))]

    # downsample training and validation sets in case args.sample keyword has been selected
    if args.sample:
        try:
            train = train.sample(n=args.sample, random_state=args.random_state)
            valid = valid.sample(n=math.ceil(int(args.sample)/2))
        except Exception:
            pass

    logger.info(f' \n Size train set: {len(train)} \n Size validation set: {len(valid)} \n Size test set: {len(test)} \n')

    # process training data
    train_rxn_id = train['reaction_id'].values
    train_smiles = train[f'{args.rxn_smiles_column}'].str.split('>', expand=True)[0].values
    train_product = train[f'{args.rxn_smiles_column}'].str.split('>', expand=True)[2].values
    train_target = train[f'{args.target_column}'].values

    # scale target values based on target distribution in the training set
    target_scaler = scale_targets(train_target.copy())
    train_target_scaled = train[f'{args.target_column}'].apply(lambda x: target_scaler.transform([[x]])[0][0]).values

    # process validation data
    valid_rxn_id = valid['reaction_id'].values
    valid_smiles = valid[f'{args.rxn_smiles_column}'].str.split('>', expand=True)[0].values
    valid_product = valid[f'{args.rxn_smiles_column}'].str.split('>', expand=True)[2].values
    valid_target = valid[f'{args.target_column}'].values
    valid_target_scaled = valid[f'{args.target_column}'].apply(lambda x: target_scaler.transform([[x]])[0][0]).values

    # set up the atom- and reaction-level descriptors
    if args.model == 'ml_QM_GNN' or args.model == 'QM_GNN':
        train_reactants = reaction_to_reactants(train['rxn_smiles'].tolist())
        qmdf_temp, _ = min_max_normalize(qmdf.copy(), train_smiles=train_reactants)
        initialize_qm_descriptors(df=qmdf_temp)
        df_reaction_desc_temp, _ = min_max_normalize_reaction(df_reaction_desc.copy(),
                                                              train_smiles=train['rxn_smiles'].tolist())
        initialize_reaction_descriptors(df=df_reaction_desc_temp)

    # set up dataloaders for training and validation sets
    train_gen = dataloader(train_smiles, train_product, train_rxn_id, train_target_scaled, args.selec_batch_size,
                           args.select_atom_descriptors, args.select_bond_descriptors, args.select_reaction_descriptors)
    train_steps = np.ceil(len(train_smiles) / args.selec_batch_size).astype(int)
    valid_gen = dataloader(valid_smiles, valid_product, valid_rxn_id, valid_target_scaled, args.selec_batch_size,
                           args.select_atom_descriptors, args.select_bond_descriptors, args.select_reaction_descriptors)
    valid_steps = np.ceil(len(valid_smiles) / args.selec_batch_size).astype(int)

    # set up tensorflow model
    model = regressor(args.feature, args.depth, args.select_atom_descriptors, args.select_reaction_descriptors)
    opt = tf.keras.optimizers.Adam(lr=args.ini_lr, clipnorm=5)
    model.compile(
        optimizer=opt,
        loss='mean_squared_error',
    )

    save_name = os.path.join(args.model_dir, f'best_model_{i}.hdf5')
    checkpoint = ModelCheckpoint(save_name, monitor='val_loss', save_best_only=True, save_weights_only=True)
    reduce_lr = LearningRateScheduler(lr_multiply_ratio(args.ini_lr, args.lr_ratio), verbose=1)

    callbacks = [checkpoint, reduce_lr]

    # run training and save weights
    print(f'training the {i}th iteration')
    hist = model.fit_generator(
        train_gen, steps_per_epoch=train_steps, epochs=args.selec_epochs,
        validation_data=valid_gen, validation_steps=valid_steps,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=args.workers,
    )

    with open(os.path.join(args.model_dir, f'history_{i}.pickle'), 'wb') as hist_pickle:
        pickle.dump(hist.history, hist_pickle)

    model.load_weights(save_name)

    # process testing data
    test_rxn_id = test['reaction_id'].values
    test_smiles = test[f'{args.rxn_smiles_column}'].str.split('>', expand=True)[0].values
    test_product = test[f'{args.rxn_smiles_column}'].str.split('>', expand=True)[2].values
    test_target = test[f'{args.target_column}'].values

    # set up dataloader for testing data
    test_gen = dataloader(test_smiles, test_product, test_rxn_id, test_target, args.selec_batch_size,
                          args.select_atom_descriptors, args.select_bond_descriptors, args.select_reaction_descriptors, shuffle=False)
    test_steps = np.ceil(len(test_smiles) / args.selec_batch_size).astype(int)

    # run model for testing set, save predictions and store metrics
    predicted = []
    mse = 0
    mae = 0
    for x, y in tqdm(test_gen, total=int(len(test_smiles) / args.selec_batch_size)):
        out = model.predict_on_batch(x)
        out = np.reshape(out, [-1])
        for y_output, y_true in zip(out, y):
            y_predicted = target_scaler.inverse_transform([[y_output]])[0][0]
            predicted.append(y_predicted)
            mae += abs(y_predicted - y_true)/int(len(test_smiles))
            mse += (y_predicted - y_true)**2/int(len(test_smiles))

    rmse = np.sqrt(mse)
    test_predicted = pd.DataFrame({'rxn_id': test_rxn_id, 'predicted': predicted})
    test_predicted.to_csv(os.path.join(args.model_dir, f'test_predicted_{i}.csv'))

    rmse_list.append(rmse)
    mae_list.append(mae)

    # report results for current fold
    print(f'success rate for iter {i}: {rmse}, {mae}')
    logger.info(f'success rate for iter {i}: {rmse}, {mae}')

# report final results at the end of the run
print(f'RMSE for {args.k_fold}-fold cross-validation: {np.mean(np.array(rmse_list))}')
print(f'MAE for {args.k_fold}-fold cross-validation: {np.mean(np.array(mae_list))}')

logger.info(f'RMSE for {args.k_fold}-fold cross-validation: {np.mean(np.array(rmse_list))}')
logger.info(f'MAE for {args.k_fold}-fold cross-validation: {np.mean(np.array(mae_list))}')
