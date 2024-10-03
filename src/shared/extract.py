import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
from shared.constants import langusage_output_order, exercises_el
from shared.utils import select_relevant_step_from_attempt_entries, load_exercise_names


def extract_solution_metadata():
    exercises = load_exercise_names()

    csv_path = "../data/solution_metadata.csv"
    if not Path.exists(Path(csv_path)):
        print("Extracting solution metadata... (this may take a while)")
        with open(csv_path, "w") as file:
            file.write(f'{",".join(langusage_output_order)}\n')
            for exercise in sorted(exercises):
                exercise_name = exercise.parts[-1]
                solution_path = exercise.joinpath(exercise_name + ".java")
                if Path.exists(solution_path):
                    result = subprocess.run(
                        [
                            "java",
                            "-jar",
                            "./langusage.jar",
                            solution_path.absolute(),
                        ],
                        stdout=subprocess.PIPE,
                    )
                    file.write(f"{exercise_name},{result.stdout.decode()}")
                else:
                    print(f"Warning: no solution found for exercise: {exercise_name}")


def extract_exercise_metadata(attempt_data, solution_metadata):
    exercise_metadata = []

    def extract_function(exercise_attempts):
        title = exercise_attempts["exercise"].iloc[0]
        n_tests = (exercise_attempts["n_tests_green"] + exercise_attempts["n_tests_red"]).agg(pd.Series.mode).iloc[0]
        median_chars = round(exercise_attempts["n_chars"].agg(pd.Series.median))
        fulfillment_duration = extract_fulfillment_duration_median(exercise_attempts, 0.5)

        return pd.Series(
            {
                "title": title,
                "n_tests": n_tests,
                "mc_cabe_solution": solution_metadata[solution_metadata["exercise"] == title]["mc_cabe_solution"].iloc[0],
                "median_chars": median_chars,
                "median_duration": fulfillment_duration,
            }
        )

    exercise_metadata = attempt_data.groupby("exercise").apply(extract_function)

    exercises_metadata_dict = exercise_metadata.to_dict(orient="records")

    if (exercises_el != exercises_metadata_dict):
        print("Warning: current exercise metadata doest not match newly extracted data! Replace it in src/shared/constants.py")

    print(exercises_metadata_dict)


def extract_fulfillment_duration_median(exercise_attempt_data, initial_min_correctness):
    pruned_data = exercise_attempt_data.copy()[(exercise_attempt_data["duration"] > 0)]
    pruned_data["metric_correctness"] = pruned_data["n_tests_green"] / (pruned_data["n_tests_green"] + pruned_data["n_tests_red"])

    if len(pruned_data) == 0:
        return np.nan

    result = None
    correctness = initial_min_correctness
    step_size = 0.05
    min_correctness = 0.25
    while result is None:
        last_attempt_entries_matching_test_portion = (
            pruned_data[pruned_data["metric_correctness"] >= correctness]
            .groupby("attempt")
            .apply(select_relevant_step_from_attempt_entries)
        )
        duration_at_full_correctness = last_attempt_entries_matching_test_portion.apply(
            lambda x: x["duration_effective"] / x["metric_correctness"],
            axis=1,
        )
        duration_median = duration_at_full_correctness.median()
        if np.isnan(duration_median):
            print(
                f"Could no calculate duration for exercise {pruned_data['exercise'].iloc[0]} with test portion {correctness}, trying again with {correctness - step_size}"
            )
            correctness -= step_size
        elif len(duration_at_full_correctness) < 5 and correctness >= min_correctness:
            print(
                f"Could not calculate median duratian for '{pruned_data['exercise'].iloc[0]}' using only {len(duration_at_full_correctness)} attempts, trying again with {correctness - step_size}"
            )
            correctness -= step_size
        else:
            if correctness < min_correctness:
                print(f"Calculating median using only {len(duration_at_full_correctness)} attempts")
            result = round(duration_median, 3)
    return result
