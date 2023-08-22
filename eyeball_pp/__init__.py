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
    SUCCESS_CUTOFF
)

from .classes import (
    Criteria,
    FeedbackResult,
    OutputFeedback,
    OutputComparator,
    OutputGrader,
    OutputScore,
)

from .recorders import (
    Checkpoint, 
    EvalRecorder
)

from .system_state import bucketize_checkpoints
evaluate_system_cmd = lambda : Fire(evaluate_system)

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
    "EvalRecorder"
    "evaluate_system_cmd"
]
