# multitask QM-augmented_GNN

This repository contains the code associated with the multitask QM-augmented GNN, used in the project "data-driven discovery of new bio-orthogonal click reactions" project. Code is provided "as-is". Minor edits may be required to tailor the scripts for different computational systems. 


### Conda environment

To set up a conda environment:

```
conda env create -f environment.yml
```

## Data

A model dataset has been included in the main directory. Data points are formatted as follows:

```
,rxn_id,rxn_smiles,solvent,temp,DG_TS,G_r
0,0,[CH3:4][C:5](=[O:6])[C:7]#[N+:8][O-:9].[CH:1]#[C:2][CH3:3]>>[cH:1]1[c:2]([CH3:3])[o:9][n:8][c:7]1[C:5]([CH3:4])=[O:6],water,298.15,18.01382414212364,-66.82234856100307
1,1,[CH3:4][C:5](=[O:6])[C:7]#[N+:8][O-:9].[CH:1]#[C:2][CH3:3]>>[cH:1]1[c:2]([CH3:3])[c:7]([C:5]([CH3:4])=[O:6])[n:8][o:9]1,water,298.15,23.6870788323347,-62.481289358393404
2,2,[CH2:1]=[CH:2][CH3:3].[CH:4]#[N+:5][C-:6]([CH3:7])[C:8]#[N:9]>>[CH2:1]1[CH:2]([CH3:3])[CH:4]=[N:5][C:6]1([CH3:7])[C:8]#[N:9],water,298.15,15.875895621136172,-51.88152615910866
```

When a QM-augmented version of the GNN is employed, atom- and/or reaction-level descriptors need to be provided as a `.pkl` file (bond-level descriptors are currently not implemented). These descriptor files can be generated with the help of the [QM_desc_autodE](https://github.com/tstuyver/QM_desc_autodE) repository.

## Training

To train the model, run:
```
python reactivity.py --data_path <path to reaction data .csv file> [--no_qm] [--atom_desc_path <path to the .pkl file containing the atom-level descriptors>] [--reaction_desc_path <path to the .pkl file containing the reaction-level descriptors>] [--depth <number of layers in the WL embedder>] [--ini_lr <initial learning rate>] [--lr_ratio <learning rate decaying ratio>] [--w_atom <initialization of the weights of the qm atom-features>] [--w_reaction <initialization of the weights of the qm reaction-features>] [--hidden_size_multiplier <multiplication factor for the hidden size in the molecule-level layers>] [--depth_mol_ffn <depth of the molecule-level feedforward network>] [--random_state <random state to be selected for sampling/shuffling>] [--ensemble_size <the number of models to be ensembled>] [--splits <split of the dataset into testing, validating, and training. The sum should be 100>] [--model_dir <path to the checkpoint file of the trained model>]
```
Unless `--no_qm` is specified, the model is augmented with partial charges, electrophilic and nucleophilic Fukui functions and NMR shielding constants at the atom-level, and G, G_alt1 and G_alt2 -- the various VB-based promotion gaps that can be defined -- at the reaction-level. Specific types of descriptors can be removed by including `--select_atom_descriptors none`/`--select_reaction_descriptors none`. Finally, also subsets of atom-/reaction-level descriptors can be selected, e.g., `--select_atom_descriptors partial_charge nmr` and/or `--select_reaction_descriptors G`.

As a pre-processing step, targets and descriptors are normalized; the scalers are saved in directories in the `model_dir` with name `scalers_{n_ensemble}`. Next, the training is performed and a checkpoint for every model in the ensemble is saved as `best_model_{n_ensemble}.hdf5` in the same directory.

For example:
```
python reactivity.py --data_path datasets/iteration0_data.csv --atom_desc_path descriptors/atom_desc_iteration0_wln.pkl --reaction_desc_path descriptors/reaction_desc_iteration0_wln.pkl --depth 2 --ini_lr 0.00165 --lr_ratio 0.93 --w_atom 0.5 --w_reaction 0.3 --hidden_size_multiplier 0 --depth_mol_ffn 1 --random_state 0 --ensemble_size 10 --splits 0 5 95 --model_dir model_iteration0
```

## Predicting
To use the trained model, run:
```
python reactivity.py --data_path <path to the predicting .csv file> --model_dir <directory containing the trained model> -p [--no_qm] [--atom_desc_path <path to the .pkl file containing the atom-level descriptors>] [--reaction_desc_path <path to the .pkl file containing the reaction-level descriptors>] [--depth <number of layers in the WL embedder>] [--ini_lr <initial learning rate>] [--lr_ratio <learning rate decaying ratio>] [--w_atom <initialization of the weights of the qm atom-features>] [--w_reaction <initialization of the weights of the qm reaction-features>] [--hidden_size_multiplier <multiplication factor for the hidden size in the molecule-level layers>] [--depth_mol_ffn <depth of the molecule-level feedforward network>] [--random_state <random state to be selected for sampling/shuffling>] [--ensemble_size <the number of models to be ensembled>] [--splits <split of the dataset into testing, validating, and training. The sum should be 100>]
```
where `data_path` is the path to the data `.csv` file, whose format has been discussed above. `model_dir` is the directory holding the trained model. The `model_dir` include scalers and model checkpoint files as discussed in the [training](#Training) section.

For example:
```
python reactivity.py --data_path datasets/iteration0_data.csv --atom_desc_path descriptors/atom_desc_iteration0_wln.pkl --reaction_desc_path descriptors/reaction_desc_iteration0_wln.pkl --depth 2 --ini_lr 0.00165 --lr_ratio 0.93 --w_atom 0.5 --w_reaction 0.3 --hidden_size_multiplier 0 --depth_mol_ffn 1 --random_state 0 --ensemble_size 10 --splits 0 5 95 --model_dir model_iteration0 -p
```

## Cross-validating
To perform a cross-validation, run (additional parameters to finetune the model architecture have been discussed above):
```
python reactivity.py --data_path <path to reaction data .csv file> [--no_qm] [--atom_desc_path <path to the .pkl file containing the atom-level descriptors>] [--reaction_desc_path <path to the .pkl file containing the reaction-level descriptors>] [--k_fold <number of folds>] [--sample <number of training points>]
```

For example:
```
python cross_val.py --data_path datasets/iteration0_data.csv --atom_desc_path descriptors/atom_desc_iteration0_wln.pkl --reaction_desc_path descriptors/reaction_desc_iteration0_wln.pkl --k_fold 10 --depth 2 --ini_lr 0.00165 --lr_ratio 0.93 --w_atom 0.5 --w_reaction 0.3 --hidden_size_multiplier 0 --depth_mol_ffn 1 --random_state 2 --ensemble_size 10 --model_dir cross_val_ensemble10
```
