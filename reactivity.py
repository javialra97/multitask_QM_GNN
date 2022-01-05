import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
import pandas as pd

import pickle

from tqdm import tqdm
from utils import lr_multiply_ratio, parse_args, create_logger, scale_targets

args, dataloader, regressor = parse_args()
reactivity_data = pd.read_csv(args.data_path, index_col=0)

logger = create_logger(name=args.model_dir)

if args.model == 'ml_QM_GNN':
    # TO DO: include reaction-level descriptors in the ml_QM_GNN model
    from ml_QM_GNN.graph_utils.mol_graph import initialize_qm_descriptors, initialize_reaction_descriptors
    from predict_desc.predict_desc import predict_desc, reaction_to_reactants
    from predict_desc.post_process import min_max_normalize, min_max_normalize_reaction
    logger.info("The considered atom-condensed descriptors are: {}".format(args.select_atom_descriptors))

else:
    if args.model == 'QM_GNN':
        from QM_GNN.graph_utils.mol_graph import initialize_qm_descriptors, initialize_reaction_descriptors
        from process_desc.post_process import min_max_normalize, reaction_to_reactants, min_max_normalize_reaction
        logger.info("The considered atom-condensed descriptors are: {}".format(args.select_atom_descriptors))
        logger.info("The considered reaction descriptors are: {}".format(args.select_atom_descriptors))

if not args.predict:
    splits = args.splits
    test_ratio = splits[0]/sum(splits)
    valid_ratio = splits[1]/sum(splits[1:])
    test = reactivity_data.sample(frac=test_ratio)
    valid = reactivity_data[~reactivity_data.reaction_id.isin(test.reaction_id)].sample(frac=valid_ratio, random_state=1)
    train = reactivity_data[~(reactivity_data.reaction_id.isin(test.reaction_id) |
                              reactivity_data.reaction_id.isin(valid.reaction_id))]

    logger.info(
        " \n Size train set: {} \n Size validation set: {} \n Size test set: {} \n".format(len(train), len(valid),
                                                                                           len(test)))
    if args.model == 'ml_QM_GNN':
        # Add reaction_descriptors
        qmdf = predict_desc(args, normalize=False)
        qmdf.to_csv(args.model_dir + "/atom_descriptors.csv")
    elif args.model == 'QM_GNN':
        qmdf = pd.read_pickle(args.atom_desc_path)
        qmdf.to_csv(args.model_dir + "/atom_descriptors.csv")
        df_reaction_desc = pd.read_pickle(args.reaction_desc_path)

    if args.model == 'ml_QM_GNN' or args.model == 'QM_GNN':
        train_reactants = reaction_to_reactants(train['rxn_smiles'].tolist())
        qmdf, atom_scalers = min_max_normalize(qmdf, train_smiles=train_reactants)
        initialize_qm_descriptors(df=qmdf)
        pickle.dump(atom_scalers, open(os.path.join(args.model_dir, 'atom_scalers.pickle'), 'wb'))
        df_reaction_desc, reaction_scalers = min_max_normalize_reaction(df_reaction_desc.copy(),
                                                              train_smiles=train['rxn_smiles'].tolist())
        pickle.dump(reaction_scalers, open(os.path.join(args.model_dir, 'reaction_desc_scalers.pickle'), 'wb'))
        initialize_reaction_descriptors(df=df_reaction_desc)

    train_rxn_id = train['reaction_id'].values
    train_smiles = train.rxn_smiles.str.split('>', expand=True)[0].values
    train_product = train['{}'.format(args.rxn_smiles_column)].str.split('>', expand=True)[2].values
    train_target = train['{}'.format(args.target_column)].values

    target_scaler = scale_targets(train_target.copy())
    pickle.dump(target_scaler, open(os.path.join(args.model_dir, 'target_scaler.pickle'), 'wb'))

    train_target_scaled = train['{}'.format(args.target_column)].apply(
        lambda x: target_scaler.transform([[x]])[0][0]).values

    valid_rxn_id = valid['reaction_id'].values
    valid_smiles = valid.rxn_smiles.str.split('>', expand=True)[0].values
    valid_product = valid['{}'.format(args.rxn_smiles_column)].str.split('>', expand=True)[2].values
    valid_target = valid['{}'.format(args.target_column)].values

    valid_target_scaled = valid['{}'.format(args.target_column)].apply(
        lambda x: target_scaler.transform([[x]])[0][0]).values

    train_gen = dataloader(train_smiles, train_product, train_rxn_id, train_target_scaled, args.selec_batch_size,
                           args.select_atom_descriptors, args.select_reaction_descriptors)
    train_steps = np.ceil(len(train_smiles) / args.selec_batch_size).astype(int)

    valid_gen = dataloader(valid_smiles, valid_product, valid_rxn_id, valid_target_scaled, args.selec_batch_size,
                           args.select_atom_descriptors, args.select_reaction_descriptors)
    valid_steps = np.ceil(len(valid_smiles) / args.selec_batch_size).astype(int)

    for x, _ in dataloader([train_smiles[0]], [train_product[0]], [train_rxn_id[0]], [train_target_scaled[0]], 1,
                           args.select_atom_descriptors, args.select_reaction_descriptors):
        x_build = x

else:
    test = reactivity_data
    test_rxn_id = test['reaction_id'].values
    test_smiles = test.rxn_smiles.str.split('>', expand=True)[0].values
    test_product = test['{}'.format(args.rxn_smiles_column)].str.split('>', expand=True)[2].values
    test_target = test['{}'.format(args.target_column)].values

    if args.model == 'ml_QM_GNN':
        # TO DO: Write function to enable prediction of reaction_descs
        qmdf = predict_desc(args)
        initialize_qm_descriptors(df=qmdf)
    if args.model == 'QM_GNN':
        qmdf = pd.read_pickle(args.atom_desc_path)
        atom_scalers = pickle.load(open(os.path.join(args.model_dir, 'atom_scalers.pickle'), 'rb'))
        qmdf, _ = min_max_normalize(qmdf, scalers=atom_scalers)
        initialize_qm_descriptors(df=qmdf)
        df_reaction_desc = pd.read_pickle(args.reaction_desc_path)
        reaction_scalers = pickle.load(open(os.path.join(args.model_dir, 'reaction_desc_scalers.pickle'), 'rb'))
        df_reaction_desc, _ = min_max_normalize_reaction(df_reaction_desc, scalers=reaction_scalers)
        initialize_reaction_descriptors(df=df_reaction_desc)

    target_scaler = pickle.load(open(os.path.join(args.model_dir, 'target_scaler.pickle'), 'rb'))

    test_gen = dataloader(test_smiles, test_product, test_rxn_id, test_target, args.selec_batch_size,
                          args.select_atom_descriptors, args.select_reaction_descriptors, predict=True)
    test_steps = np.ceil(len(test_smiles) / args.selec_batch_size).astype(int)

    # need an input to initialize the graph network
    for x in dataloader([test_smiles[0]], [test_product[0]], [test_rxn_id[0]], [test_target[0]], 1,
                        args.select_atom_descriptors, args.select_reaction_descriptors, predict=True):
        x_build = x

save_name = os.path.join(args.model_dir, 'best_model.hdf5')

model = regressor(args.feature, args.depth, args.select_atom_descriptors, args.select_reaction_descriptors)
opt = tf.keras.optimizers.Adam(lr=args.ini_lr, clipnorm=5)
model.compile(
    optimizer=opt,
    loss='mean_squared_error',
    metrics=[tf.keras.metrics.RootMeanSquaredError(
        name='root_mean_squared_error', dtype=None), tf.keras.metrics.MeanAbsoluteError(
        name='mean_absolute_error', dtype=None), ]
)
model.predict_on_batch(x_build)
model.summary()

if args.restart or args.predict:
    model.load_weights(save_name)

checkpoint = ModelCheckpoint(save_name, monitor='val_loss', save_best_only=True, save_weights_only=True)

reduce_lr = LearningRateScheduler(lr_multiply_ratio(args.ini_lr, args.lr_ratio), verbose=1)

callbacks = [checkpoint, reduce_lr]

if not args.predict:
    hist = model.fit_generator(
        train_gen, steps_per_epoch=train_steps, epochs=args.selec_epochs,
        validation_data=valid_gen, validation_steps=valid_steps,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=args.workers
    )
else:
    predicted = []
    #masks = []
    for x in tqdm(test_gen, total=int(len(test_smiles) / args.selec_batch_size)):
        #masks.append(x[-2])
        out = model.predict_on_batch(x)
        y_predicted = target_scaler.inverse_transform([[out]])[0][0]
        predicted.append(y_predicted)

    predicted = np.concatenate(predicted, axis=0)
    predicted = predicted.reshape(-1)

    test_predicted = pd.DataFrame({'rxn_id': test_rxn_id, 'predicted': predicted})
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    test_predicted.to_csv(os.path.join(args.output_dir, 'predicted.csv'))
