import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
import pandas as pd

from GNN.WLN.data_loading import Graph_DataLoader as dataloader
from GNN.WLN.models import WLNRegressor as regressor
from GNN.graph_utils.mol_graph import initialize_qm_descriptors, initialize_reaction_descriptors
from process_descs import min_max_normalize_atom_descs, reaction_to_reactants, min_max_normalize_reaction_descs
from process_descs import predict_atom_descs, predict_reaction_descs

import pickle
from tqdm import tqdm
from utils import lr_multiply_ratio, parse_args, create_logger, scale_targets

args = parse_args()
reactivity_data = pd.read_csv(args.data_path, index_col=0)

logger = create_logger(name=args.model_dir)

# training of the model
if not args.predict:
    splits = args.splits
    test_ratio = splits[0]/sum(splits)
    valid_ratio = splits[1]/sum(splits[1:])
    test = reactivity_data.sample(frac=test_ratio)
    valid = reactivity_data[~reactivity_data.reaction_id.isin(test.reaction_id)].sample(frac=valid_ratio, random_state=1)
    train = reactivity_data[~(reactivity_data.reaction_id.isin(test.reaction_id) |
                              reactivity_data.reaction_id.isin(valid.reaction_id))]

    logger.info(f' \n Size train set: {len(train)} \n Size validation set: {len(valid)} \n Size test set: {len(test)} \n')

    # initialize the descriptors
    if "none" not in args.select_atom_descriptors or "none" not in args.select_bond_descriptors:
        if args.qm_pred:
            logger.info(f"Predicting atom-level descriptors")
            qmdf = predict_atom_descs(args, normalize=False)
        else:
            qmdf = pd.read_pickle(args.atom_desc_path)
        qmdf.to_csv(os.path.join(args.model_dir, "atom_descriptors.csv"))
        train_reactants = reaction_to_reactants(train['rxn_smiles'].tolist())
        qmdf, atom_scalers = min_max_normalize_atom_descs(qmdf, train_smiles=train_reactants)
        initialize_qm_descriptors(df=qmdf)
        pickle.dump(atom_scalers, open(os.path.join(args.model_dir, 'atom_scalers.pickle'), 'wb'))
        logger.info(f"The considered atom-level descriptors are: {args.select_atom_descriptors}")
        logger.info(f"The considered bond descriptors are: {args.select_bond_descriptors}")
    if "none" not in args.select_reaction_descriptors:
        if args.qm_pred:
            raise NotImplementedError
        else:
            df_reaction_desc = pd.read_pickle(args.reaction_desc_path)
        df_reaction_desc.to_csv(os.path.join(args.model_dir, "reaction_descriptors"))
        logger.info(f"The considered reaction descriptors are: {args.select_reaction_descriptors}")
        df_reaction_desc, reaction_scalers = min_max_normalize_reaction_descs(df_reaction_desc.copy(),
                                                              train_smiles=train['rxn_smiles'].tolist())
        pickle.dump(reaction_scalers, open(os.path.join(args.model_dir, 'reaction_desc_scalers.pickle'), 'wb'))
        initialize_reaction_descriptors(df=df_reaction_desc)

    # process the training data
    train_rxn_id = train['reaction_id'].values
    train_smiles = train.rxn_smiles.str.split('>', expand=True)[0].values
    train_product = train[f'{args.rxn_smiles_column}'].str.split('>', expand=True)[2].values
    train_target = train[f'{args.target_column}'].values

    # scale target values based on target distribution in the training set
    target_scaler = scale_targets(train_target.copy())
    pickle.dump(target_scaler, open(os.path.join(args.model_dir, 'target_scaler.pickle'), 'wb'))

    train_target_scaled = train[f'{args.target_column}'].apply(lambda x: target_scaler.transform([[x]])[0][0]).values

    # process the validation data
    valid_rxn_id = valid['reaction_id'].values
    valid_smiles = valid.rxn_smiles.str.split('>', expand=True)[0].values
    valid_product = valid[f'{args.rxn_smiles_column}'].str.split('>', expand=True)[2].values
    valid_target = valid[f'{args.target_column}'].values

    valid_target_scaled = valid[f'{args.target_column}'].apply(
        lambda x: target_scaler.transform([[x]])[0][0]).values

   # set up dataloaders for training and validation sets
    train_gen = dataloader(train_smiles, train_product, train_rxn_id, train_target_scaled, args.selec_batch_size,
                           args.select_atom_descriptors, args.select_bond_descriptors, args.select_reaction_descriptors)
    train_steps = np.ceil(len(train_smiles) / args.selec_batch_size).astype(int)

    valid_gen = dataloader(valid_smiles, valid_product, valid_rxn_id, valid_target_scaled, args.selec_batch_size,
                           args.select_atom_descriptors, args.select_bond_descriptors, args.select_reaction_descriptors)
    valid_steps = np.ceil(len(valid_smiles) / args.selec_batch_size).astype(int)

    for x, _ in dataloader([train_smiles[0]], [train_product[0]], [train_rxn_id[0]], [train_target_scaled[0]], 1,
                           args.select_atom_descriptors, args.select_bond_descriptors, args.select_reaction_descriptors):
        x_build = x

# only testing
else:
    # process the testing data 
    test = reactivity_data
    test_rxn_id = test['reaction_id'].values
    test_smiles = test.rxn_smiles.str.split('>', expand=True)[0].values
    test_product = test[f'{args.rxn_smiles_column}'].str.split('>', expand=True)[2].values
    test_target = test[f'{args.target_column}'].values

    if "none" not in args.select_atom_descriptors or "none" not in args.select_bond_descriptors:
        if args.qm_pred:
            qmdf = predict_atom_descs(args)
        else:
            qmdf = pd.read_pickle(args.atom_desc_path)
        atom_scalers = pickle.load(open(os.path.join(args.model_dir, 'atom_scalers.pickle'), 'rb'))
        qmdf, _ = min_max_normalize_atom_descs(qmdf, scalers=atom_scalers)
        initialize_qm_descriptors(df=qmdf)
    if "none" not in args.select_reaction_descriptors:
        if args.qm_pred:
            df_reaction_desc = predict_reaction_descs(args)
        else:
            df_reaction_desc = pd.read_pickle(args.reaction_desc_path)
        reaction_scalers = pickle.load(open(os.path.join(args.model_dir, 'reaction_desc_scalers.pickle'), 'rb'))
        df_reaction_desc, _ = min_max_normalize_reaction_descs(df_reaction_desc, scalers=reaction_scalers)
        initialize_reaction_descriptors(df=df_reaction_desc)

    target_scaler = pickle.load(open(os.path.join(args.model_dir, 'target_scaler.pickle'), 'rb'))

    # set up dataloader for test set
    test_gen = dataloader(test_smiles, test_product, test_rxn_id, test_target, args.selec_batch_size,
                          args.select_atom_descriptors, args.select_bond_descriptors, args.select_reaction_descriptors, predict=True)
    test_steps = np.ceil(len(test_smiles) / args.selec_batch_size).astype(int)

    # need an input to initialize the graph network
    for x in dataloader([test_smiles[0]], [test_product[0]], [test_rxn_id[0]], [test_target[0]], 1,
                        args.select_atom_descriptors, args.select_bond_descriptors, args.select_reaction_descriptors, predict=True):
        x_build = x

save_name = os.path.join(args.model_dir, 'best_model.hdf5')

# set up the model for evaluation
model = regressor(args.feature, args.depth, args.select_atom_descriptors, args.select_reaction_descriptors, args.w_atom, args.w_reaction)
opt = tf.keras.optimizers.Adam(learning_rate=args.ini_lr, clipnorm=5)
model.compile(
    optimizer=opt,
    loss='mean_squared_error',
    )

# initialize the model by running x_build
model.predict_on_batch(x_build)
model.summary()

if args.restart or args.predict:
    model.load_weights(save_name)

checkpoint = ModelCheckpoint(save_name, monitor='val_loss', save_best_only=True, save_weights_only=True)

reduce_lr = LearningRateScheduler(lr_multiply_ratio(args.ini_lr, args.lr_ratio), verbose=1)

callbacks = [checkpoint, reduce_lr]

if not args.predict:
    # set up the model for training
    hist = model.fit(
        train_gen, steps_per_epoch=train_steps, epochs=args.selec_epochs,
        validation_data=valid_gen, validation_steps=valid_steps,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=args.workers
    )
else:
    # evaluate predictions
    predicted = []
    for x in tqdm(test_gen, total=int(len(test_smiles) / args.selec_batch_size)):
        out = model.predict_on_batch(x)
        y_predicted = target_scaler.inverse_transform([[out]])[0][0]
        predicted.append(y_predicted)

    predicted = np.concatenate(predicted, axis=0)
    predicted = predicted.reshape(-1)

    test_predicted = pd.DataFrame({'reaction_id': test_rxn_id, 'predicted': predicted})
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    test_predicted.to_csv(os.path.join(args.output_dir, 'predicted.csv'))
