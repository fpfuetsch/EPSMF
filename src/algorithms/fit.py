import warnings
import numpy as np

import pandas as pd

from algorithms.generic import HyperParameters, Model
from algorithms.params.bias import TimeBias
from algorithms.params.bias_shift import FactorBiasShift
from algorithms.params.preference_shift import FactorPreferenceShift
from shared.measures import mae, rmse


def split_and_prepare_data(df, metric, test_set_size=0.3, n_days_grouped=1, test_set_indices=None):
    df = df.copy()

    # add t_ui column
    df["last_progress"] = df["last_progress"].dt.date
    earliest_rating = df["last_progress"].min()
    df["t_ui"] = df["last_progress"].apply(lambda x: (x - earliest_rating).days // n_days_grouped)

    # call metric r_ui
    df = df.rename(columns={metric: "r_ui"})

    # extract t_u and _t_i
    t_u = df.groupby("student")["t_ui"].min()
    t_i = df.groupby("exercise")["t_ui"].min()

    t_i_median = round(df.groupby("exercise")["t_ui"].median())
    t_u_mean = round(df.groupby("student")["t_ui"].mean())
    t_u_max = df.groupby("student")["t_ui"].max()
    n_ratings_u = df.groupby("student")["t_ui"].count()

    # split df into train and test set
    if test_set_indices is None:
        test_set = df.sample(frac=test_set_size)
        train_set = df.drop(test_set.index)
    else:
        test_set = df[df.index.isin(test_set_indices)]
        train_set = df.drop(test_set_indices)

    # drop unnecessary columns
    relevant_columns = ["student", "exercise", "t_ui", "r_ui"]
    test_set = test_set[relevant_columns]
    train_set = train_set[relevant_columns]

    # extract meta_data from train_set
    students = train_set["student"].unique()
    exercises = train_set["exercise"].unique()
    days = train_set["t_ui"].unique()
    taus = (train_set["t_ui"] - train_set["student"].map(t_u)).unique()
    omegas = (train_set["t_ui"] - train_set["exercise"].map(t_i)).unique()

    # prepare meta data
    meta_data = {
        "t_u": t_u,
        "t_u_mean": t_u_mean,
        "t_u_max": t_u_max,
        "t_i": t_i,
        "t_i_median": t_i_median,
        "n_ratings_u": n_ratings_u,
        "students": students,
        "exercises": exercises,
        "days": days,  # TODO rename this to time bins
        "taus": taus,
        "omegas": omegas,
    }

    return train_set, test_set, meta_data


def prune_set(model, df, meta_data):
    components = map(lambda x: x.__class__, model.components)
    df = df[(df["student"].isin(meta_data["students"])) & (df["exercise"].isin(meta_data["exercises"]))].copy()
    if TimeBias in components:
        df = df[(df["t_ui"].isin(meta_data["days"]))].copy()
    if FactorBiasShift in components or FactorPreferenceShift in components:
        df = df[
            ((df["t_ui"] - df["student"].map(meta_data["t_u"])).isin(meta_data["taus"]))
            & ((df["t_ui"] - df["exercise"].map(meta_data["t_i"])).isin(meta_data["omegas"]))
        ].copy()
    return df


def fit_and_evaluate(model: Model, train_set, test_set, meta_data, hyper_params: HyperParameters, verbose=False, clip_min=0, clip_max=1):
    model_instance = model(pd_known_ratings=train_set, meta_data=meta_data, hyper_params=hyper_params, verbose=verbose)
    model_has_temporal_components = model_instance.has_temporal_components()

    # fit model
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            model_instance.fit()
        except Warning as e:
            print(f"Warning: Aborted due to: {e.args[0]}")
            return

    # prune test set
    pruned_test_set = prune_set(model_instance, test_set, meta_data)
    if len(pruned_test_set) != len(test_set):
        diff = len(test_set) - len(pruned_test_set)
        portion = round(diff * 100 / len(test_set), 2)
        if verbose:
            print(f"\nInfo: {diff}({portion}%) entries were pruned from the test set!")

    # set up values to predict
    base_prediction_set = pd.concat([train_set, pruned_test_set])

    # create set of missing predictions with missing student exercise combinations
    train_set_exercises = train_set["exercise"].unique()
    additional_combinations = []
    for student in train_set["student"].unique():
        student_exercises = base_prediction_set[base_prediction_set["student"] == student]["exercise"].unique()
        missing_exercises = train_set_exercises[~np.isin(train_set_exercises, student_exercises)]
        for exercise in missing_exercises:
            additional_combinations.append(
                {"student": student, "exercise": exercise, "t_ui": int(meta_data["t_i_median"][exercise]), "r_ui": np.nan}
            )
    additional_predictions_set = pd.DataFrame(additional_combinations, columns=["student", "exercise", "t_ui", "r_ui"])

    # prune set of additional predictions
    pruned_additional_predictions_set = prune_set(model_instance, additional_predictions_set, meta_data)
    if len(additional_predictions_set) != len(pruned_additional_predictions_set):
        diff = len(additional_predictions_set) - len(pruned_additional_predictions_set)
        portion = round(diff * 100 / len(additional_predictions_set), 2)
        if verbose:
            print(f"\nInfo: {diff}({portion}%) entries were pruned from the additional predictions set!")

    # predict
    prediction_set = pd.concat([base_prediction_set, pruned_additional_predictions_set])
    predictions_list = []
    if model_has_temporal_components:
        prediction_set = prediction_set[["student", "exercise", "t_ui"]]
        for student, exercise, t_ui in prediction_set.itertuples(index=False):
            predictions_list.append(
                {
                    "student": student,
                    "exercise": exercise,
                    "t_ui": t_ui,
                    "r_ui": np.clip(model_instance.predict(student, exercise, t_ui), clip_min, clip_max),
                }
            )
        predictions = pd.DataFrame(predictions_list)
        predictions.set_index(["student", "exercise", "t_ui"], inplace=True)
    else:
        prediction_set = prediction_set[["student", "exercise"]].drop_duplicates()
        for student, exercise in prediction_set.itertuples(index=False):
            predictions_list.append(
                {
                    "student": student,
                    "exercise": exercise,
                    "r_ui": np.clip(model_instance.predict(student, exercise, None), clip_min, clip_max),
                }
            )
        predictions = pd.DataFrame(predictions_list)
        predictions.set_index(["student", "exercise"], inplace=True)
    predictions.sort_index(inplace=True)

    rmse_train = rmse(train_set, predictions, model_has_temporal_components)
    mae_train = mae(train_set, predictions, model_has_temporal_components)

    scores = {
        "rmse": {
            "train": rmse_train,
        },
        "mae": {
            "train": mae_train,
        },
    }

    if verbose:
        print("\nTrain-Set Evaluation:")
        print(f"Root Mean Squared Error: {round(rmse_train, 5)}")
        print(f"Mean Absolute Error: {round(mae_train, 5)}")

    if len(test_set) > 0:
        rmse_test = rmse(pruned_test_set, predictions, model_has_temporal_components)
        mae_test = mae(pruned_test_set, predictions, model_has_temporal_components)
        scores["rmse"]["test"] = rmse_test
        scores["mae"]["test"] = mae_test

        if verbose:
            print("\nTest-Set Evaluation:")
            print(f"Root Mean Squared Error: {round(rmse_test, 5)}")
            print(f"Mean Absolute Error: {round(mae_test, 5)}")

    return (scores, predictions, model_instance.get_params())


def k_fold_split(data_set, k):
    """Split data set into k folds. The last fold might be slightly larger than the others."""
    fold_size = int(len(data_set) // k)
    test_set_indices = []
    for i in range(1, k + 1):
        test_set_start = int((i - 1) * fold_size)
        test_set_end = int(i * fold_size)
        if i < k:
            test_set_indices.append(data_set.iloc[test_set_start:test_set_end].index)
        else:
            test_set_indices.append(data_set.iloc[test_set_start:].index)
    return test_set_indices


def k_fold_cross_validation(model: Model, hyper_params: HyperParameters, data_set, metric, k=5, n_days_grouped=1):
    shuffled_data_set = data_set.sample(frac=1)
    test_set_indices = k_fold_split(shuffled_data_set, k)
    test_rmse_sum = 0
    test_mae_sum = 0
    for i in range(0, k):
        train_set, test_set, meta_data = split_and_prepare_data(
            shuffled_data_set, metric, test_set_indices=test_set_indices[i], n_days_grouped=n_days_grouped
        )
        res = fit_and_evaluate(model, train_set, test_set, meta_data, hyper_params, verbose=False, clip_min=0, clip_max=1)
        if res is None:
            return None
        test_rmse_sum += res[0]["rmse"]["test"]
        test_mae_sum += res[0]["mae"]["test"]
    return {
        "rmse": test_rmse_sum / k,
        "mae": test_mae_sum / k,
    }
