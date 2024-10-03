import pandas as pd
import io
import numpy as np

from shared.constants import (
    did_not_compile_test_name,
    did_not_compile_replacement_name,
    type_mapping,
    exercises_el,
    ignored_exercises,
)
from shared.utils import calculate_effective_duration_using_break_threshold
from shared.extract import extract_solution_metadata


def replace_did_not_compile_string(record_line):
    record_line_entries = record_line.split(",")
    record_line_entries[15] = record_line_entries[15].replace(did_not_compile_test_name, did_not_compile_replacement_name)
    return ",".join(record_line_entries)


def record_data_to_df(record_entries):
    tests_per_exercise = {item["title"]: item["n_tests"] for item in exercises_el}

    record_lines = []
    for record_line in record_entries:
        if any(record_line.startswith(ignored_exercice) for ignored_exercice in ignored_exercises):
            continue
        record_line_entries = record_line.split(",")
        number_of_tests = tests_per_exercise[record_line_entries[0]]
        if int(record_line_entries[6]) <= number_of_tests:
            record_lines.append(replace_did_not_compile_string(record_line))
        else:
            print(
                f"Warning: ignoring edit log entry with more than max number of tests in exercise {record_line_entries[0]}, attempt {record_line_entries[1]}"
            )

    df = pd.read_csv(
        io.StringIO("\n".join(record_lines)),
        delimiter=",",
        header=None,
        names=type_mapping.keys(),
        dtype={k: v for k, v in type_mapping.items() if v != "date"},
        parse_dates=[k for k, v in type_mapping.items() if v == "date"],
        date_parser=lambda x: pd.to_datetime(int(x), unit="ms"),
    )

    # drop unused column
    del df["unused"]
    del df["exercise_finished"]
    del df["author_confirmed"]

    # calculate number of red tests
    df["n_tests_red"] = df.apply(lambda x: tests_per_exercise[x["exercise"]] - x["n_tests_green"], axis=1)

    effetive_duration = (
        df.groupby("attempt")
        .apply(calculate_effective_duration_using_break_threshold, break_threshold_in_minutes=60, break_surcharge_in_minutes=5)
        .reset_index(drop=False)
        .sort_values(["attempt", "duration_effective"])
    )
    df = df.sort_values(["attempt", "last_submission"])
    df["duration_effective"] = effetive_duration["duration_effective"].values

    return df


def load_student_metadata():
    exam_results_file = "../data/exam.csv.anon"
    testat_results_file = "../data/testat.csv.anon"
    prior_xp_file = "../data/prior_xp.csv.anon"

    exam_results_df = pd.read_csv(exam_results_file, header=None, names=["student", "exam"], delimiter=",")
    testat_results_df = pd.read_csv(testat_results_file, header=None, names=["student", "t1", "t2", "t3"], delimiter=",")
    testat_results_df["t1"] = testat_results_df["t1"].replace("-", np.nan).astype(float)
    testat_results_df["t2"] = testat_results_df["t2"].replace("-", np.nan).astype(float)
    testat_results_df["t3"] = testat_results_df["t3"].replace("-", np.nan).astype(float)
    prior_xp_df = pd.read_csv(prior_xp_file, header=None, names=["student", "prior_xp_in_months"], delimiter=",")
    merged_df = pd.merge(exam_results_df, testat_results_df, on="student", how="outer").merge(prior_xp_df, on="student", how="outer")
    merged_df.set_index("student", inplace=True)

    # calculate exam grade
    def calculate_exam_grade(row):
        exam = row["exam"]
        if np.isnan(exam):
            return exam
        else:
            if exam == 100:
                exam = 104  # exam had 4 bonus points
            if not (np.isnan(row["t1"])):
                exam -= row["t1"] * 0.1
            if not (np.isnan(row["t2"])):
                exam -= row["t2"] * 0.1
            if not (np.isnan(row["t3"])):
                exam -= row["t3"] * 0.1
            return max(0, min(exam, 74))

    merged_df["exam"] = merged_df.apply(calculate_exam_grade, axis=1)
    return merged_df


def load_edit_log_data():
    record_lines = []
    with open("../data/edit.log.anon") as log_file:
        for line in log_file.readlines():
            line_entries = line.split(",")
            num_line_entries = len(line_entries)
            if num_line_entries == 18:  # StudentID field does not exist yet
                line = line[:-1] + ",\n"
            elif num_line_entries != 19:
                print(f"Warning: found edit log entry with {num_line_entries} entries: {line}")
                continue
            record_lines.append(line)
    return record_data_to_df(record_lines)


def load_solution_medatadata():
    extract_solution_metadata()
    return pd.read_csv("../data/solution_metadata.csv", skipinitialspace=True)
