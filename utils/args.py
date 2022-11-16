import os
import argparse

from rdkit import rdBase

rdBase.DisableLog("rdApp.warning")


def parse_args(cross_val=False):
    parser = argparse.ArgumentParser()
    # main/essential command line arguments
    parser.add_argument("--no_qm", action="store_true", help="no QM augmentation")
    parser.add_argument("--qm_pred", action="store_true", help="predict QM descriptors")
    parser.add_argument(
        "--model_dir",
        default="trained_model",
        help="path to the checkpoint file of the trained model",
    )
    parser.add_argument(
        "--atom_desc_path",
        default=None,
        help="path to the .pkl file containing the atom-level descriptors",
    )
    parser.add_argument(
        "--reaction_desc_path",
        default=None,
        help="path to the .pkl file containing the reaction-level descriptors",
    )
    parser.add_argument(
        "--data_path", default=None, type=str, help="path to reaction data .csv file"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="random state to be selected for sampling/shuffling",
    )
    parser.add_argument(
        "--splits",
        nargs=3,
        type=int,
        default=[10, 10, 80],
        help="split of the dataset into testing, validating, and training. The sum should be 100",
    )
    parser.add_argument(
        "--select_atom_descriptors",
        nargs="+",
        default=["partial_charge", "fukui_elec", "fukui_neu", "nmr"],
        help="(optional) selection of atom-condensed descriptors to feed to the GNN model",
    )
    parser.add_argument(
        "--select_reaction_descriptors",
        nargs="+",
        default=["G", "G_alt1", "G_alt2"],
        help="(optional) selection of reaction descriptors to feed to the GNN model",
    )
    parser.add_argument(
        "--select_bond_descriptors",
        nargs="+",
        default=["bond_order", "bond_length"],
        help="(optional) selection of bond descriptors to feed to the (ml_)QM_GNN model",
    )
    # command line arguments for network architecture finetuning
    parser.add_argument(
        "--ini_lr", default=0.001, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--lr_ratio", default=0.97, type=float, help="learning rate decaying ratio"
    )
    parser.add_argument(
        "--w_atom",
        type=float,
        default=0.5,
        help="initialization of the weights of the qm atom-features" 
        "(0.5 means equal weights for structural and QM atom features)",
    )
    parser.add_argument(
        "--w_reaction",
        type=float,
        default=0.5,
        help="initialization of the weights of the qm reaction-features"
        "(0.5 means equal weights for the sum-pooled atom-feature based representation and reaction-level descriptors)",
    )
    parser.add_argument(
        "--depth_mol_ffn",
        type=int,
        default=2,
        help="depth of the molecule-level feedforward network",
    )
    parser.add_argument(
        "--hidden_size_multiplier",
        type=int,
        default=20,
        help="multiplication factor for the hidden size in the molecule-level layers",
    )
    parser.add_argument(
        "--selec_epochs",
        default=100,
        type=int,
        help="number of epochs while training the selectivity model",
    )
    # optional/non-essential command line arguments
    parser.add_argument(
        "-r",
        "--restart",
        action="store_true",
        help="restart the training using the saved the checkpoint file",
    )
    parser.add_argument(
        "-o", "--output_dir", default="output", help="directory saving output files"
    )
    parser.add_argument(
        "-f", "--feature", default=50, type=int, help="feature size for GNN"
    )
    parser.add_argument(
        "-d", "--depth", default=4, type=int, help="number of layers in the WL embedder"
    )
    parser.add_argument(
        "-w", "--workers", default=10, type=int, help="number of workers"
    )
    parser.add_argument(
        "--rxn_id_column",
        default="rxn_id",
        type=str,
        help="the column in which the rxn_id are stored",
    )
    parser.add_argument(
        "--rxn_smiles_column",
        default="rxn_smiles",
        type=str,
        help="the column in which the rxn_smiles are stored",
    )
    parser.add_argument(
        "--target_column1",
        default="DG_TS",
        help="the column in which the activation energies are stored",
    )
    parser.add_argument(
        "--target_column2",
        default="G_r",
        help="the column in which the reaction energies are stored",
    )
    parser.add_argument(
        "--selec_batch_size",
        default=10,
        type=int,
        help="batch size while training the selectivity model",
    )
    parser.add_argument(
        "--train_valid_set_path",
        type=str,
        default=None,
        help="in case of selective sampling, indicates path to combined training and validation set csv-file",
    )
    parser.add_argument(
        "--test_set_path",
        type=str,
        default=None,
        help="in case of selective sampling, indicates path to the test set csv-file",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="the number of models to be ensembled",
    )

    if cross_val:
        parser.add_argument(
            "--k_fold",
            default=10,
            type=int,
            help="(Optional) # fold for cross-validation",
        )
        parser.add_argument(
            "--sample",
            type=int,
            help="(Optional) Randomly sample part of data for training during cross-validation",
        )
    else:
        parser.add_argument(
            "-p",
            "--predict",
            action="store_true",
            help="predict reactivity for a given .csv file",
        )
        parser.add_argument(
            "--output_file",
            default="test_predicted.csv",
            help="name of .csv file to write predictions to"
        )

    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)

    if args.no_qm:
        (
            args.select_atom_descriptors,
            args.select_bond_descriptors,
            args.select_reaction_descriptors,
        ) = (["none"], ["none"], ["none"])

    # temporary, since bond descriptors are not implemented in the current version
    args.select_bond_descriptors = ["none"]

    return args
