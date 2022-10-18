import pandas as pd
import math
from sklearn.preprocessing import StandardScaler


class Dataset:
    """A dataset object for the GNN, initialized from a dataframe."""

    def __init__(
        self,
        df,
        output_scalers=None,
        rxn_id_column="rxn_id",
        rxn_smiles_column="rxn_smiles",
        target_column1="DG_TS",
        target_column2="G_r",
    ):
        """
        Args:
            df (pd.DataFrame): dataframe containing rxn_id, rxn_smiles and targets
            output_scalers (List[sklearn.preprocessing.StandardScaler]):
                activation_energy and reaction_energy scaler
            rxn_id_column (str): name of the rxn-id column
            rxn_smiles_column (str): name of the rxn-smiles column
            target_column1 (str): name of the first target column
            target_column2 (str): name of the second target column

        """
        self.rxn_id = df[f"{rxn_id_column}"].values
        self.rxn_smiles = df[f"{rxn_smiles_column}"].values
        self.reactant_smiles = (
            df[f"{rxn_smiles_column}"].str.split(">", expand=True)[0].values
        )
        self.product_smiles = (
            df[f"{rxn_smiles_column}"].str.split(">", expand=True)[2].values
        )

        if target_column1 != None and target_column2 != None:
            self.activation_energy = df[f"{target_column1}"].values
            self.reaction_energy = df[f"{target_column2}"].values

            if output_scalers != None:
                self.output_scalers = output_scalers
            else:
                self.output_scalers = [
                    scale_targets(self.activation_energy.copy()),
                    scale_targets(self.reaction_energy.copy()),
                ]

            self.activation_energy_scaled = self.output_scalers[0].transform(
                self.activation_energy.reshape(-1, 1)
            )
            self.reaction_energy_scaled = self.output_scalers[1].transform(
                self.reaction_energy.reshape(-1, 1)
            )
        else:
            self.activation_energy, self.reaction_energy = None, None
            self.activation_energy_scaled, self.reaction_energy_scaled = None, None

    def __len__(self):
        return len(self.rxn_smiles)


def split_data_cross_val(
    df,
    k_fold_arange,
    i,
    j,
    rxn_id_column,
    data_path,
    train_valid_set_path,
    sample,
    k_fold,
    random_state,
    test_set_path,
):
    """Splits a dataframe into train, valid, test dataframes for a single fold, single model in cv

    Args:
        df (pd.DataFrame): the entire dataset
        k_fold_arange (np.linspace): limits of the individual folds
        i (int): current fold (cross-validation)
        j (int): current model (ensemble)
        rxn_id_column (str): the name of the rxn-id column in the dataframe
        data_path (str): path to the entire dataset
        train_valid_set_path (str): path the to train/valid dataset (selective sampling)
        sample (int): number of training points to sample
        k_fold (int): number of folds
        random_state (int): the random state to be used for the splitting and sampling
    """
    if data_path is not None:
        test = df[k_fold_arange[i] : k_fold_arange[i + 1]]
        valid = df[~df[f"{rxn_id_column}"].isin(test[f"{rxn_id_column}"])].sample(
            frac=1 / (k_fold - 1), random_state=random_state + j
        )
        train = df[
            ~(
                df[f"{rxn_id_column}"].isin(test[f"{rxn_id_column}"])
                | df[f"{rxn_id_column}"].isin(valid[f"{rxn_id_column}"])
            )
        ]
    elif train_valid_set_path is not None and test_set_path is not None:
        valid = df.sample(frac=1 / (k_fold - 1), random_state=random_state + i + j)
        train = df[~(df[f"{rxn_id_column}"].isin(valid[f"{rxn_id_column}"]))]
        test = pd.read_csv(test_set_path, index_col=0)

    # downsample training and validation sets in case args.sample keyword has been selected
    if sample:
        try:
            train = train.sample(n=sample, random_state=random_state + j)
            valid = valid.sample(
                n=math.ceil(int(sample) / 4), random_state=random_state + j
            )
        except Exception:
            pass

    return train, valid, test


def split_data_training(df, rxn_id_column, splits, random_state, i):
    """Splits a dataframe into train, valid, test dataframes for a single model (regular training)

    Args:
        df (pd.DataFrame): the entire dataset
        rxn_id_column (str): the name of the rxn-id column in the dataframe
        splits (List[int]): relative sizes of train, valid, test sets
        random_state (int): random state used for data splitting
        i (int): current model (ensemble)
    """
    splits = splits
    test_ratio = splits[0] / sum(splits)
    valid_ratio = splits[1] / sum(splits[1:])
    test = df.sample(frac=test_ratio, random_state=random_state)
    valid = df[~df[f"{rxn_id_column}"].isin(test[f"{rxn_id_column}"])].sample(
        frac=valid_ratio, random_state=random_state + i
    )
    train = df[
        ~(
            df[f"{rxn_id_column}"].isin(test[f"{rxn_id_column}"])
            | df[f"{rxn_id_column}"].isin(valid[f"{rxn_id_column}"])
        )
    ]

    return train, valid, test


def scale_targets(df):
    """Fit scaler to target dataframe

    Args:
        df (pd.DataFrame): dataframe to be scaled

    Returns:
        _type_: sklearn.StandardScaler
    """
    target_scaler = StandardScaler()
    data = df.reshape(-1, 1).tolist()
    target_scaler.fit(data)

    return target_scaler
