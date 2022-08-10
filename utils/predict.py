import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate(test_gen, selec_batch_size, models, scalers):
    """Make predictions for the test set, save them and return statistics.

    Args:
        test_gen (Dataloader): dataloader object for the test set
        selec_batch_size (int): the batch size
        models (List[WLNRegressor]): list of models (ensemble)
        scalers (list[sklearn.StandardScaler]): list of scalers for each model (ensemble)
    """
    # initialize
    predicted_activation_energies_per_batch = []
    predicted_reaction_energies_per_batch = []
    true_activation_energies_per_batch, true_reaction_energies_per_batch = [], []

    # make predictions on test set per batch and take ensemble-average
    for x, y in tqdm(test_gen, total=int(len(test_gen.smiles) / selec_batch_size)):
        predicted_activation_energies_batch = np.empty(
            shape=(len(models), selec_batch_size)
        )
        predicted_reaction_energies_batch = np.empty(
            shape=(len(models), selec_batch_size)
        )
        for i, model in enumerate(models):
            # the scaled targets in test_gen correspond to the last scaler
            if i == len(scalers) - 1:
                true_activation_energies_per_batch.append(
                    scalers[i][0].inverse_transform(y["activation_energy"])
                )
                true_reaction_energies_per_batch.append(
                    scalers[i][1].inverse_transform(y["reaction_energy"])
                )
            out = model.predict_on_batch(x)
            predicted_activation_energies_batch[i] = (
                scalers[i][0].inverse_transform(out["activation_energy"]).reshape(10)
            )
            predicted_reaction_energies_batch[i] = (
                scalers[i][1].inverse_transform(out["reaction_energy"]).reshape(10)
            )

        predicted_activation_energies_batch_avg = np.sum(
            predicted_activation_energies_batch, axis=0
        ) / len(models)
        predicted_reaction_energies_batch_avg = np.sum(
            predicted_reaction_energies_batch, axis=0
        ) / len(models)
        predicted_activation_energies_per_batch.append(
            predicted_activation_energies_batch_avg
        )
        predicted_reaction_energies_per_batch.append(
            predicted_reaction_energies_batch_avg
        )

    # combine individual batches
    predicted_activation_energies = np.array(
        predicted_activation_energies_per_batch
    ).reshape(-1)
    predicted_reaction_energies = np.array(
        predicted_reaction_energies_per_batch
    ).reshape(-1)
    true_activation_energies = np.array(true_activation_energies_per_batch).reshape(-1)
    true_reaction_energies = np.array(true_reaction_energies_per_batch).reshape(-1)

    print(true_activation_energies, true_reaction_energies)
    print(test_gen.rxn_id)
    raise KeyError
    # determine accuracy metrics
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
        predicted_activation_energies,
        predicted_reaction_energies,
        rmse_activation_energy,
        rmse_reaction_energy,
        mae_activation_energy,
        mae_reaction_energy,
    )


def write_predictions(
    test_gen,
    predicted_activation_energies,
    predicted_reaction_energies,
    rxn_id_column,
    file_name,
):
    """Write predictions to a .csv file.

    Args:
        test_gen (Dataloader): dataloader object for the test set
        activation_energies_predicted (List): list of predicted activation energies
        reaction_energies_predicted (List): list of predicted reaction energies
        rxn_id_column (str): name of the rxn-id column
        file_name (str): name of .csv file to write the predicted values to
    """

    test_predicted = pd.DataFrame(
        {
            f"{rxn_id_column}": test_gen.rxn_id,
            "predicted_activation_energy": predicted_activation_energies,
            "predicted_reaction_energy": predicted_reaction_energies,
        }
    )
    test_predicted.to_csv(file_name)
