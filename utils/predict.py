import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf


def predict_single_model(pipeline_test, model, output_scalers, len_test_set, batch_size=10):
    """Predicts output for a single model.

    Args:
        test_gen (GNN.WLN.Dataloader): a dataloader object that generates input
        model (GNN.WLN.WLNRegressor): the GNN model
        output_scalers (List[StandardScalers]): scalers to inverse transform output from GNN
        len_test_set (int): the length of the test set on which a prediction is being made
        batch_size (int): the batch size

    Returns:
        predicted_activation_energy (np.array): array of predicted activation energy values
        predicted_reaction_energy (np.array): array of predicted reaction energy values
    """

    predicted_activation_energies_per_batch = []
    predicted_reaction_energies_per_batch = []

    for x in tqdm(pipeline_test, total=int(len_test_set / batch_size)):
        out = model.predict_on_batch(x)

        predicted_activation_energies_batch = (
            output_scalers[0].inverse_transform(out["activation_energy"]).reshape(-1)
        )
        predicted_reaction_energies_batch = (
            output_scalers[1].inverse_transform(out["reaction_energy"]).reshape(-1)
        )

        predicted_activation_energies_per_batch.append(
            predicted_activation_energies_batch
        )
        predicted_reaction_energies_per_batch.append(predicted_reaction_energies_batch)

    predicted_activation_energies = np.hstack(predicted_activation_energies_per_batch)
    predicted_reaction_energies = np.hstack(predicted_reaction_energies_per_batch)

    return predicted_activation_energies, predicted_reaction_energies


def evaluate(
    predicted_activation_energies,
    predicted_reaction_energies,
    true_activation_energies,
    true_reaction_energies,
):
    """Determine metrics based on predicted and true values.

    Args:
        predicted_activation_energies (np.array): array of predicted activation energies
        predicted_reaction_energies (np.array): array of predicted reaction energies
        true_activation_energies (np.array): array of recorded activation energies
        true_reaction_energies (np.array): array of recorded reaction energies

    Returns:
        rmse_activation_energy (int): RMSE for the activation energy
        rmse_reaction_ (np.array): array of predicted reaction energy values
    """
    mae_activation_energy = mean_absolute_error(
        predicted_activation_energies, true_activation_energies
    )
    mae_reaction_energy = mean_absolute_error(
        predicted_reaction_energies, true_reaction_energies
    )
    rmse_activation_energy = np.sqrt(
        mean_squared_error(predicted_activation_energies, true_activation_energies)
    )
    rmse_reaction_energy = np.sqrt(
        mean_squared_error(predicted_reaction_energies, true_reaction_energies)
    )

    return (
        rmse_activation_energy,
        rmse_reaction_energy,
        mae_activation_energy,
        mae_reaction_energy,
    )


def write_predictions(
    rxn_ids,
    predicted_activation_energies,
    predicted_reaction_energies,
    rxn_id_column,
    file_name,
):
    """Write predictions to a .csv file.

    Args:
        rxn_id (pd.DataFrame): dataframe consisting of the rxn_ids
        activation_energies_predicted (List): list of predicted activation energies
        reaction_energies_predicted (List): list of predicted reaction energies
        rxn_id_column (str): name of the rxn-id column
        file_name (str): name of .csv file to write the predicted values to
    """

    test_predicted = pd.DataFrame(
        {
            f"{rxn_id_column}": rxn_ids,
            "predicted_activation_energy": predicted_activation_energies,
            "predicted_reaction_energy": predicted_reaction_energies,
        }
    )
    test_predicted.to_csv(file_name)
