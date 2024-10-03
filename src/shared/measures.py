import numpy as np
import pandas as pd


def rmse(actual, predictions, include_temporal_information=False):
    differences = np.array([], dtype=float)
    for student, exercise, t_ui, actual_rating in actual.itertuples(index=False):
        if include_temporal_information:
            predicted_rating = predictions.loc[student, exercise, t_ui]
        else:
            predicted_rating = predictions.loc[student, exercise]
        differences = np.append(differences, actual_rating - predicted_rating)
    squared_differences = np.square(differences)
    return np.sqrt(squared_differences.mean())


def mae(actual, predictions, include_temporal_information=False):
    differences = np.array([], dtype=float)
    for student, exercise, t_ui, actual_rating in actual.itertuples(index=False):
        if include_temporal_information:
            predicted_rating = predictions.loc[student, exercise, t_ui]
        else:
            predicted_rating = predictions.loc[student, exercise]
        differences = np.append(differences, actual_rating - predicted_rating)
    return np.mean(np.absolute(differences))