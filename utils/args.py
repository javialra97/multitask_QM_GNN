import os
import argparse

from rdkit import rdBase

rdBase.DisableLog('rdApp.warning')


def parse_args(cross_val=False):
    parser = argparse.ArgumentParser()
    # main/essential command line arguments
    parser.add_argument('--no_qm', action='store_true',
                        help='no QM augmentation')
    parser.add_argument('--qm_pred', action='store_true',
                        help='predict QM descriptors')
    parser.add_argument('--model_dir', default='trained_model',
                        help='path to the checkpoint file of the trained model')
    parser.add_argument('--atom_desc_path', default=None,
                        help='path to the file storing the atom-condensed descriptors (must be provided when using QM_GNN model)')
    parser.add_argument('--reaction_desc_path', default=None,
                        help='path to the file storing the reaction descriptors (must be provided when using QM_GNN model)')
    parser.add_argument('--data_path', default=None, type=str,
                        help='Path to reaction data')
    parser.add_argument('--random_state', type=int, default=0,
                        help='Random state to be selected for sampling/shuffling')
    parser.add_argument('--splits', nargs=3, type=int, default=[10, 10, 80],
                        help='Split of the dataset into testing, validating, and training. The sum should be 100')
    parser.add_argument('--select_atom_descriptors', nargs='+',
                        default=['partial_charge', 'fukui_elec', 'fukui_neu', 'nmr'],
                        help='(Optional) Selection of atom-condensed descriptors to feed to the (ml_QM_)GNN model')
    parser.add_argument('--select_reaction_descriptors', nargs='+',
                        default=['G', 'G_alt1', 'G_alt2'],
                        help='(Optional) Selection of reaction descriptors to feed to the (ml_)QM_GNN model')
    parser.add_argument('--select_bond_descriptors', nargs='+', default=['bond_order', 'bond_length'],
                        help='(Optional) Selection of bond descriptors to feed to the (ml_)QM_GNN model')
    # command line arguments for network architecture finetuning
    parser.add_argument('--ini_lr', default=0.001, type=float,
                        help='Initial learning rate')
    parser.add_argument('--lr_ratio', default=0.97, type=float,
                        help='Learning rate decaying ratio')
    parser.add_argument('--w_atom', type=float, default=1.0,
                        help='initialization of the weights of the qm atom-features relative to the learned atom-features')
    parser.add_argument('--w_reaction', type=float, default=1.0,
                        help='initialization of the weights of the qm reaction-features' 
                        'relative to the sum-pooled atom-feature based representation')
    parser.add_argument('--depth_mol_ffn', type=int, default=2, help='depth of the molecular-level feedforward network')
    parser.add_argument('--hidden_size_multiplier', type=int, default=20, help='multiplication factor for the hidden size in the molecular-level layers')
    parser.add_argument('--selec_epochs', default=100, type=int,
                        help='Number of epochs while training the selectivity model')
    # optional/non-essential command line arguments
    parser.add_argument('-r', '--restart', action='store_true',
                        help='restart the training using the saved the checkpoint file')
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict reactivity for a given .csv file')
    parser.add_argument('-o', '--output_dir', default='output',
                        help='Directory saving output files')
    parser.add_argument('-f', '--feature', default=50, type=int,
                        help='feature size for GNN')
    parser.add_argument('-d', '--depth', default=4, type=int,
                        help='Number of steps for GNN')
    parser.add_argument('-w', '--workers', default=10, type=int,
                        help='Number of workers')
    parser.add_argument('--rxn_id_column', default='rxn_id', type=str,
                        help='the column in which the rxn_id are stored')
    parser.add_argument('--rxn_smiles_column', default='rxn_smiles', type=str,
                        help='the column in which the rxn_smiles are stored')
    parser.add_argument('--target_column1', default='DG_TS',
                        help='the column in which the activation energies are stored')
    parser.add_argument('--target_column2', default='G_r',
                        help='the column in which the reaction energies are stored')
    parser.add_argument('--selec_batch_size', default=10, type=int,
                        help='Batch size while training the selectivity model')
    parser.add_argument('--train_valid_set_path', type=str, default=None,
                        help='In case of selective sampling, indicates path to combined training and validation set csv-file')
    parser.add_argument('--test_set_path', type=str, default=None,
                        help='In case of selective sampling, indicates path to the test set csv-file')

    if cross_val:
        parser.add_argument('--k_fold', default=10, type=int,
                            help='(Optional) # fold for cross-validation')
        parser.add_argument('--sample', type=int,
                            help='(Optional) Randomly sample part of data for training during cross-validation')

    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)

    if args.no_qm:
        args.select_atom_descriptors, args.select_bond_descriptors, args.select_reaction_descriptors = ["none"], ["none"], ["none"]

    return args
