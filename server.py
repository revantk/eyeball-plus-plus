from collections import defaultdict
import datetime
from eyeball_pp.classes import TASK_OUTPUT_KEY, FeedbackResult
from eyeball_pp.eval import SUCCESS_CUTOFF
from eyeball_pp.recorders import Checkpoint, EvalRecorder, FileRecorder
from eyeball_pp.system_state import bucketize_checkpoints
from eyeball_pp.utils import time_to_str
import json
import pandas as pd
from statistics import variance
import streamlit as st
import sys
from typing import Optional, Any


@st.cache_data
def get_recorder() -> EvalRecorder:
    return FileRecorder("examples/eyeball_data")


def flatten_dataframe_column(df: pd.DataFrame, column: str) -> str:
    """Flatten a column of a dataframe into its own columns."""
    expanded = pd.json_normalize(df[column])
    position = df.columns.get_loc(column)
    for i, col in enumerate(expanded.columns):
        df.insert(position + i, col, expanded[col])
    return df.drop(columns=[column])


def render_dataframe_with_selections(df: pd.DataFrame) -> pd.DataFrame:
    """Render the dataframe with the selection checkboxes and
    returned the selected items as a dataframe"""
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select":
                       st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )

    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)


def dataframe_from_checkpoints(checkpoints: list[Checkpoint]) -> pd.DataFrame:
    """Convert a list of checkpoints into a dataframe."""
    df = pd.DataFrame([checkpoint.as_dict() for checkpoint in checkpoints])

    columns_displayed = ["input_variables", "eval_params", "output", "score"]

    for col in columns_displayed:
        if col not in df.columns:
            df[col] = ""

    df = df[columns_displayed]

    df = flatten_dataframe_column(df, "input_variables")
    df = flatten_dataframe_column(df, "score")
    df.rename(columns={"task_output.score": "score"}, inplace=True)
    df.rename(columns={"task_output.message": "evaluation"}, inplace=True)

    def extract_evaluations(s):
        data = json.loads(s)
        evaluation = ""
        for criteria in data['evaluations']:
            evaluation += \
            f"{criteria['name']}: {criteria['rating']}. {criteria['reason']}\n"
        return evaluation

    df['evaluation'] = df['evaluation'].apply(extract_evaluations)

    df.fillna("", inplace=True)
    return df


def get_scored_checkpoints(
        recorder: EvalRecorder, task_name: str, num_samples: int = sys.maxsize
) -> list[Checkpoint]:
    """Get the latest scored checkpoints for a given task."""
    output_names_to_score: set[str] = set([TASK_OUTPUT_KEY])
    input_hashes = recorder.get_input_hashes(task_name=task_name)

    all_checkpoints: list[Checkpoint] = []
    input_hash_to_checkpoints: dict[str, list[Checkpoint]] = {}
    for input_hash in input_hashes:
        checkpoints = recorder.get_latest_checkpoints(
            task_name=task_name,
            input_hash=input_hash,
            num_checkpoints=num_samples
        )
        input_hash_to_checkpoints[input_hash] = checkpoints
        all_checkpoints += checkpoints

    if len(all_checkpoints) > 0:
        scored_checkpoints: list[Checkpoint] = []
        rerunid_to_checkpoint_feedback: dict[
            str, dict[str, dict[str, Optional[FeedbackResult]]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))

        for c in all_checkpoints:
            scored_outputs = set(c.scores.keys())
            if len(scored_outputs) > 0:
                scored_checkpoints.append(c)
                output_names_to_score |= scored_outputs

        scored_checkpoints = [c for c in all_checkpoints if len(c.scores) > 0]

        if len(scored_checkpoints) > 0:
            scored_checkpoints.sort(key=lambda x: x.checkpoint_id, reverse=True)
            return scored_checkpoints


def render_checkpoints(checkpoints : list[Checkpoint]) -> None:
    st.markdown("### Selected Checkpoints")
    if len(checkpoints) == 0:
        st.write("No checkpoints selected")
    else:
        df = dataframe_from_checkpoints(checkpoints)
        st.dataframe(df, hide_index=True)


def render_system_health_by_date(
        checkpoints: list[Checkpoint],
        num_samples: int = sys.maxsize,
        plotting_frequency_in_days: int = 1
) -> list[Checkpoint]:
    """Render System Health by Date and return the list of selected checkpoints."""
    st.markdown("### System Health: By Date")

    date_to_use = datetime.datetime.utcnow().date()
    system_health_by_date: list[dict[str, Any]] = []
    checkpoints_by_date: list[list[Checkpoint]] = []
    while checkpoints[-1].created_at.date() <= date_to_use:
        if checkpoints[0].created_at.date() > date_to_use - \
            datetime.timedelta(days=plotting_frequency_in_days):
            num_successes = 0.0
            num_checkpoints_used = 0
            checkpoints_selected: list[Checkpoint] = []
            input_hash_set = set()
            for checkpoint in checkpoints:
                if num_checkpoints_used >= num_samples:
                    break

                if checkpoint.created_at.date() <= date_to_use:
                    if (
                        checkpoint.scores is not None
                        and TASK_OUTPUT_KEY in checkpoint.scores
                    ):
                        if checkpoint.scores[TASK_OUTPUT_KEY].score > SUCCESS_CUTOFF:
                            num_successes += 1
                        num_checkpoints_used += 1
                        checkpoints_selected.append(checkpoint)
                        input_hash_set.add(checkpoint.get_input_hash())

            checkpoints_by_date.append(checkpoints_selected)
            system_health_by_date.append(
                {
                    "Date": time_to_str(date_to_use),
                    "Results": f"{float(num_successes) / float(num_checkpoints_used) * 100.0: .1f}% success ({num_successes}/{num_checkpoints_used})",
                    "Stats": f"{num_checkpoints_used} datapoints, {len(input_hash_set)} unique inputs",
                }
            )
        date_to_use -= datetime.timedelta(days=plotting_frequency_in_days)

    df = pd.DataFrame(system_health_by_date)
    df_selection = render_dataframe_with_selections(df)
    selected_checkpoints: list[Checkpoint] = []
    for index in df_selection.index:
        checkpoints = checkpoints_by_date[index]
        selected_checkpoints = selected_checkpoints + checkpoints
    return selected_checkpoints


def render_system_health_by_run(checkpoints: list[Checkpoint]) -> list[Checkpoint]:
    """Render System Health by Run and return the list of selected checkpoints."""
    st.markdown("### System Health: By Run History")

    buckets_to_checkpoints = bucketize_checkpoints(checkpoints)
    system_health_by_run_history = []
    output_names_to_score: set[str] = set([TASK_OUTPUT_KEY])
    bucket_checkpoint_list = list(buckets_to_checkpoints.items())
    for bucket, checkpoints in bucket_checkpoint_list:
        row = {"Run": bucket}

        for output_name in output_names_to_score:
            num_checkpoints_used = 0
            num_successes = 0
            params_used: set[str] = set()
            input_hash_to_score: dict[str, list[float]] = {}
            for checkpoint in checkpoints:
                if output_name in checkpoint.scores:
                    if checkpoint.scores[output_name].score > SUCCESS_CUTOFF:
                        num_successes += 1
                    num_checkpoints_used += 1
                    input_hash = checkpoint.get_input_hash()
                    if input_hash not in input_hash_to_score:
                        input_hash_to_score[input_hash] = []
                    input_hash_to_score[input_hash].append(
                        checkpoint.scores[output_name].score
                    )
                    if checkpoint.eval_params:
                        keys = sorted(checkpoint.eval_params.keys())
                        params_used.add(
                            "\n".join(
                                f"{k}={checkpoint.eval_params[k]}" for k in keys
                            )
                        )
            if num_checkpoints_used > 0:
                percent = float(num_successes) / float(num_checkpoints_used) * 100.0
                column_name = (
                    "Results"
                    if output_name == TASK_OUTPUT_KEY
                    else f"{output_name}"
                )
                row[
                    column_name
                ] = f"{percent: .1f}% success ({num_successes}/{num_checkpoints_used})"

                if output_name == TASK_OUTPUT_KEY:
                    stats = f"{num_checkpoints_used} datapoints, {len(input_hash_to_score)} unique inputs"
                    row["Stats"] = stats
                    if len(params_used) == 1:
                        row["Params"] = params_used.pop()
        if len(row) > 1:
            system_health_by_run_history.append(row)

    df = pd.DataFrame(system_health_by_run_history)
    df_selection = render_dataframe_with_selections(df)
    selected_checkpoints: list[Checkpoint] = []
    for index in df_selection.index:
        bucket, checkpoints = bucket_checkpoint_list[index]
        selected_checkpoints = selected_checkpoints + checkpoints
    return selected_checkpoints


def sort_checkpoints(checkpoints: list[Checkpoint]) -> list[Checkpoint]:
    """Sort checkpoints by their checkpoint id and remove duplicates."""
    selected_checkpoints = []
    unique_checkpoint_ids = set()
    for checkpoint in checkpoints:
        if checkpoint.checkpoint_id not in unique_checkpoint_ids:
            selected_checkpoints.append(checkpoint)
            unique_checkpoint_ids.add(checkpoint.checkpoint_id)
    selected_checkpoints.sort(key=lambda x: x.checkpoint_id, reverse=True)
    return selected_checkpoints


def render_page() -> None:
    recorder = get_recorder()
    task_name = st.sidebar.selectbox("Task", recorder.get_task_names())

    scored_checkpoints = get_scored_checkpoints(recorder, task_name)
    if not scored_checkpoints or len(scored_checkpoints) == 0:
        "No Data Available"
    else:
        selected_checkpoints = render_system_health_by_date(scored_checkpoints)
        selected_checkpoints += render_system_health_by_run(scored_checkpoints)
        selected_checkpoints = sort_checkpoints(selected_checkpoints)
        render_checkpoints(selected_checkpoints)


if __name__ == "__main__":
    render_page()
