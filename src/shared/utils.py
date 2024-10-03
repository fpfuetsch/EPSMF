import pandas as pd
import numpy as np
from pathlib import Path

def select_relevant_step_from_attempt_entries(attempt_entries):
    """deduplicates attempts entries:
    - keeps the latest submission with the highest progress (according to number of passed tests) if the highest progress is not 100% since
      if the students does not pass more tests the additional time he tried to get further should be counted (which tells us something about the skill)
    - keep the first submission with the highest progress if the highest progress is 100% (then any further submissions might be accidental)
    """
    max_correctness = attempt_entries["metric_correctness"].max()
    if max_correctness == 1:
        return attempt_entries.sort_values(["metric_correctness", "last_submission"], ascending=[False, True]).iloc[0]
    else:
        return attempt_entries.sort_values(["metric_correctness", "last_submission"], ascending=[False, False]).iloc[0]


def calculate_effective_duration_using_break_threshold(attempt_entries, break_threshold_in_minutes=120, break_surcharge_in_minutes=5):
    sorted_attempt_entries = attempt_entries.sort_values("last_submission")
    break_threshold = break_threshold_in_minutes * 60
    break_surcharge = break_surcharge_in_minutes * 60
    durations = np.array([calculate_effective_duration_using_decay_function(sorted_attempt_entries.iloc[0]) * 60])
    # iterate over all entries and add difference in last_submission to duration if difference is lower than 2 hours
    for i in range(1, len(sorted_attempt_entries)):
        duration = durations[i - 1]
        difference = (
            sorted_attempt_entries.iloc[i]["last_submission"] - sorted_attempt_entries.iloc[i - 1]["last_submission"]
        ).total_seconds()
        if difference < break_threshold:
            duration += difference
        else:
            duration += break_surcharge
        durations = np.append(durations, duration)
    return pd.Series(durations / 60, index=sorted_attempt_entries.index, name="duration_effective")


def calculate_effective_duration_using_decay_function(record):
    decay_curve = lambda day: (4 / (3 * day + 1)) * 60
    effective_duration = 0
    float_duration = (record["last_progress"] - record["first_submission"]).total_seconds() / 60
    # sometimes the last progress is before the first submission
    if float_duration < 0:
        float_duration *= -1
    number_of_whole_days = int(float_duration // (24 * 60))
    for day in range(number_of_whole_days):
        effective_duration += decay_curve(day)
    remaining_minutes = float_duration - (number_of_whole_days * 24 * 60)
    effective_duration += min(remaining_minutes, decay_curve(number_of_whole_days))
    return min(effective_duration, 10 * 60)

def load_exercise_names():
    return list(filter(lambda entry: Path.is_dir(entry), Path.iterdir(Path("../data/"))))
