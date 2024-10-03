import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning

from shared.constants import (
    reorder_record_data,
    language_element_mapping,
    exercises_el,
    metric_curves,
)

from shared.utils import select_relevant_step_from_attempt_entries


def min_max_scale(x, min, max):
    x_std = (x - x.min()) / (x.max() - x.min())
    x_scaled = x_std * (max - min) + min
    return x_scaled


def prepare_record_data(raw_df, solution_metadata):
    exercises_by_title = {exercise["title"]: exercise for exercise in exercises_el}

    raw_df = raw_df.copy()

    # adapt types and replace unknown values by nan
    raw_df["duration"] = pd.to_timedelta(raw_df["duration"], unit="m")
    raw_df["duration_effective"] = pd.to_timedelta(raw_df["duration_effective"], unit="m")

    raw_df["difficulty"] = raw_df["difficulty"].replace("-1", np.nan)
    raw_df["difficulty"] = raw_df["difficulty"].replace("0", np.nan)

    raw_df["prior_xp"] = raw_df["prior_xp"].replace("-1", np.nan)
    raw_df["prior_xp"] = raw_df["prior_xp"].replace("0", np.nan)

    raw_df["group"] = raw_df["group"].replace("UNKNOWN", np.nan)
    raw_df["n_hints"] = raw_df["n_hints"].replace(-1, 0)
    raw_df["n_chars"] = raw_df["n_chars"].replace(-1, np.nan)
    raw_df["code_structure"] = raw_df["code_structure"].replace(-1, np.nan)
    raw_df["mc_cabe"] = raw_df["mc_cabe"].replace(-1, np.nan)
    raw_df["student"] = raw_df["student"].replace("", np.nan)

    raw_df["metric_correctness"] = raw_df.apply(
        lambda x: x["n_tests_green"] / (x["n_tests_green"] + x["n_tests_red"]) if (x["n_tests_green"] + x["n_tests_red"]) != 0 else 0,
        axis=1,
    )

    raw_df["metric_code_structure"] = raw_df.apply(
        lambda x: x["code_structure"] / 100 if not np.isnan(x["code_structure"]) else np.nan,
        axis=1,
    )

    # Caution: Removes all exercises that do not have a mc_cabe_solution value in solution_metadata
    raw_df = raw_df.merge(solution_metadata[["exercise", "mc_cabe_solution"]], on="exercise", how="inner")

    raw_df["metric_complexity"] = raw_df.apply(
        lambda x: min((x["metric_correctness"] * x["mc_cabe_solution"]) / x["mc_cabe"], 2) if x["mc_cabe"] != 0 else 0,
        axis=1,
    )

    raw_df["metric_test_portion_per_hour"] = raw_df.apply(
        lambda x: x["metric_correctness"]
        / ((x["duration_effective"].total_seconds() if x["duration_effective"].total_seconds() != 0 else 3600) / 3600),
        axis=1,
    )

    def map_to_student_progress_parameter(attempt_entries):
        exercise_name = attempt_entries["exercise"].iloc[0]
        # group attempt entries by last_progress and select relevent entry according to correcteness
        progress_steps = attempt_entries.groupby("last_progress").apply(select_relevant_step_from_attempt_entries)
        # remove progress step after first 100% correctness
        trimmed_progress_steps = []
        for _, step in progress_steps.iterrows():
            trimmed_progress_steps.append(step)
            # stop when correctness is at 100%, even if there are more steps
            if step["metric_correctness"] == 1:
                break

        trimmed_progress_steps = pd.DataFrame(trimmed_progress_steps)

        # estimate actual starting point if first step already has some progress but in nearly no time
        first_step = trimmed_progress_steps.iloc[0]
        no_duration_limit = pd.Timedelta(seconds=10)
        if first_step["n_tests_green"] > 1 and first_step["duration_effective"] < no_duration_limit:
            pre_work_duration = first_step["metric_correctness"] * exercises_by_title[exercise_name]["median_duration"]
        else:
            pre_work_duration = 0
        trimmed_progress_steps["duration_effective_recalculated"] = (
            trimmed_progress_steps["duration_effective"] + pd.Timedelta(minutes=pre_work_duration)
        ).dt.total_seconds() / 60

        # normalize time axis using median duration for exercise
        trimmed_progress_steps["duration_portion"] = (
            trimmed_progress_steps["duration_effective_recalculated"] / exercises_by_title[exercise_name]["median_duration"]
        )

        x_data = trimmed_progress_steps["duration_portion"].to_list()
        y_data = trimmed_progress_steps["metric_correctness"].to_list()

        if x_data[0] != 0:
            x_data = [0, *x_data]
            y_data = [0, *y_data]
        if len(x_data) < 2:
            x_data = [*x_data, x_data[-1] + 1]
            y_data = [*y_data, y_data[-1]]

        parameters = {
            "attempt": attempt_entries["attempt"].iloc[0],
        }
        for name, curve in metric_curves.items():
            # fit curve param to data
            p_opt, p_cov = curve_fit(
                curve,
                x_data,
                y_data,
                bounds=(0, np.inf),
            )
            parameters[f"metric_ps_{name}"] = round(p_opt[0], 3)
        return pd.DataFrame(parameters, index=[0])

    warnings.filterwarnings("ignore", category=OptimizeWarning)
    attempts_with_student_progress_parameters = raw_df.groupby("attempt").apply(map_to_student_progress_parameter)
    attempts_with_student_progress_parameters.set_index("attempt", inplace=True)
    raw_df = raw_df.merge(attempts_with_student_progress_parameters, on="attempt")

    raw_df.reset_index()

    # deduplicate attempts entries
    raw_df = raw_df.groupby("attempt").apply(select_relevant_step_from_attempt_entries)

    # scale and normalize metrics
    for metric in [
        "metric_ps_a_x",
        "metric_ps_a_x_squared",
        "metric_ps_a_log_x",
        "metric_ps_a_p_log_x",
        "metric_ps_a_sqrt_x",
        "metric_test_portion_per_hour",
    ]:
        # log scale
        raw_df[f"{metric}_log"] = raw_df[metric].apply(lambda x: np.log2(x + 0.001))
        # bounded to 99.9 % of data
        lower_bound = raw_df[f"{metric}_log"].quantile(0.005)
        upper_bound = raw_df[f"{metric}_log"].quantile(0.995)
        raw_df[f"{metric}_log_bounded"] = raw_df.apply(lambda x: np.min([np.max([x[f"{metric}_log"], lower_bound]), upper_bound]), axis=1)
        # min-max scale
        raw_df[f"{metric}_log_bounded_min_max"] = min_max_scale(raw_df[f"{metric}_log_bounded"], 0, 1)

        raw_df["metric_complexity_min_max"] = min_max_scale(raw_df["metric_complexity"], 0, 1)

    # reorder
    raw_df = reorder_record_data(raw_df)

    return raw_df, raw_df[raw_df["student"].notna()]


def prepare_solution_metadata(solution_metadata):
    solution_metadata = solution_metadata.copy()
    solution_metadata.insert(
        loc=1,
        column="mc_cabe_solution",
        value=solution_metadata["mc_cabe_main"] + solution_metadata["mc_cabe_without_main"],
    )
    solution_metadata.reset_index()

    solution_metadata_grouped = solution_metadata[["exercise"]].copy()

    for k, v in language_element_mapping.items():
        sum = None
        for group_element in v:
            count = solution_metadata[group_element]
            if group_element == "TYPEDECLARATION":
                count -= 1 # each soluation needs one class
            if group_element == "ARRAYTYPE":
                count -= solution_metadata["MAINDECLARATION"]
            if sum is None:
                sum = count
            else:
                sum = sum + count
        solution_metadata_grouped[k] = sum

    solution_metadata_grouped_scaled = solution_metadata_grouped.copy()
    for exercise in solution_metadata_grouped_scaled.index:
        sum = 0
        for k, v in language_element_mapping.items():
            if k == "primitive":
                solution_metadata_grouped_scaled.loc[exercise, k] = np.log10(solution_metadata_grouped_scaled.loc[exercise, k] + 1)
            else:
                solution_metadata_grouped_scaled.loc[exercise, k] = np.log(solution_metadata_grouped_scaled.loc[exercise, k] + 1)
            sum += solution_metadata_grouped_scaled.loc[exercise, k]
        for k, v in language_element_mapping.items():
            solution_metadata_grouped_scaled.loc[exercise, k] = solution_metadata_grouped_scaled.loc[exercise, k] / sum

    return solution_metadata, solution_metadata_grouped, solution_metadata_grouped_scaled


def prepare_exam_results(known_exam_results, dfs):
    # list all students in dfs that are not in known_exam_results
    students_exam_result_not_known = dfs[~dfs["student"].isin(known_exam_results["student"])]["student"].unique()

    # add student in students_exam_result_not_known to exam_results with grade "-"
    all_exam_results = pd.concat(
        [
            known_exam_results,
            pd.DataFrame(
                {
                    "student": students_exam_result_not_known,
                    "grade": ["-"] * len(students_exam_result_not_known),
                }
            ),
        ]
    )

    # write df to csv
    all_exam_results.to_csv("../data/eval_all.csv.anon", index=False, header=False)
