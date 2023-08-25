from typing import Optional
from fire import Fire
from .eval import (
    rerun_recorded_examples,
    set_config,
    record_task,
    record_output,
    record_input,
    record_intermediary_state,
    get_eval_param,
    rate_recorded_examples,
    evaluate_system,
    delete_checkpoints_for_input_vars,
    calculate_system_health,
    start_recording_session,
    default_evaluator,
    get_default_recorder,
    cleanup_old_checkpoints,
    Evaluator,
    get_config_dict,
    time_to_str,
    TASK_OUTPUT_KEY,
    optimize_policy_based_on_feedback,
)

from .classes import (
    SUCCESS_CUTOFF,
    Criteria,
    FeedbackResult,
    OutputFeedback,
    OutputComparator,
    OutputGrader,
    OutputScore,
)

from .recorders import Checkpoint, EvalRecorder

from .system_state import bucketize_checkpoints


def rate_recorded_examples_cmd():
    def _wrapper(
        dir_path: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        task_name: Optional[str] = None,
    ):
        set_config(api_key=api_key, api_url=api_url, dir_path=dir_path)
        print(get_default_recorder().get_task_names())
        rate_recorded_examples(task_name=task_name)

    Fire(_wrapper)


evaluate_system_cmd = lambda: Fire(evaluate_system)

__all__ = [
    "set_config",
    "record_task",
    "record_output",
    "record_input",
    "get_eval_param",
    "rerun_recorded_examples",
    "rate_recorded_examples",
    "evaluate_system",
    "delete_checkpoints_for_input_vars",
    "Criteria",
    "Evaluator",
    "FeedbackResult",
    "OutputFeedback",
    "OutputComparator",
    "OutputScore",
    "OutputGrader",
    "calculate_system_health",
    "record_intermediary_state",
    "start_recording_session",
    "default_evaluator",
    "get_default_recorder",
    "cleanup_old_checkpoints",
    "get_config_dict",
    "Checkpoint",
    "bucketize_checkpoints",
    "time_to_str",
    "TASK_OUTPUT_KEY",
    "SUCCESS_CUTOFF",
    "EvalRecorder",
    "evaluate_system_cmd",
    "rate_recorded_examples_cmd",
    "optimize_policy_based_on_feedback",
]
