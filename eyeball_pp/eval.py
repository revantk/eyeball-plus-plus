from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
import inspect
import json
import os
import types
from .recorders import (
    ApiClientRecorder,
    Checkpoint,
    ComparisonResult,
    DiskRecorder,
    FileRecorder,
    MemoryRecorder,
    EvalRecorder,
    get_input_hash,
)
from .comparators import model_graded_comparator, output_feedback_from_scores
from .classes import (
    FeedbackResult,
    OutputComparator,
    OutputScore,
    OutputScorer,
    OutputFeedback,
)

import random
import threading
from typing import (
    Callable,
    Iterable,
    Iterator,
    Protocol,
    TypeVar,
    Optional,
    Any,
    runtime_checkable,
)
from functools import wraps
from dataclasses import dataclass
import dataclasses
import datetime
from .utils import get_user_input
from .classes import FeedbackResult
from rich.console import Console
from rich.table import Table

GREEN = "\x1b[32m"
ORANGE = "\x1b[33m"
RED = "\x1b[31m"
BOLD = "\x1b[1m"
UNDERLINE = "\x1b[4m"
ITALIC = "\x1b[3m"
HEADING_BG = "\x1b[103m"

END_CLR = "\x1b[0m"


@runtime_checkable
class JsonSerializable(Protocol):
    def to_json(self) -> str:
        ...

    @staticmethod
    def from_json(json_str: str) -> "JsonSerializable":
        ...


T = TypeVar("T", int, float, str, bool, bytes, dict, list, None, JsonSerializable)


@dataclass
class EvaluatorConfig:
    sample_rate: float = 1.0
    dir_path: str = "."
    api_key: Optional[str] = None
    api_url: Optional[str] = None

    @staticmethod
    def _merge(original_config: "EvaluatorConfig", **kwargs) -> "EvaluatorConfig":
        """Kwargs should be a subset of the fields of the original config."""
        if len(kwargs) == 0:
            return original_config

        new_config = EvaluatorConfig()

        for field in dataclasses.fields(original_config):
            if (field_val := kwargs.get(field.name)) is not None:
                setattr(new_config, field.name, field_val)
            else:
                setattr(new_config, field.name, getattr(original_config, field.name))

        return new_config


class EvaluatorMode(Enum):
    RECORD = "record"
    COMPARE_CHECKPOINTS = "compare_checkpoints"
    RERUN_EXAMPLES = "rerun_examples"
    RATE_EXAMPLES = "rate_examples"


class Evaluator:
    def __init__(self, **config_kwargs) -> None:
        self.config = EvaluatorConfig()
        self.mode: EvaluatorMode = EvaluatorMode.RECORD
        self.running_in_notebook = False
        try:
            get_ipython()  # type: ignore
            self.running_in_notebook = True
            self.current_recorder_state = types.SimpleNamespace()
        except NameError:
            self.current_recorder_state = threading.local()  # type: ignore
        self.set_config(**config_kwargs)

    def _get_config(self, **override_config_kwargs) -> EvaluatorConfig:
        return EvaluatorConfig._merge(self.config, **override_config_kwargs)

    def set_config(self, **config_kwargs) -> None:
        self.config = EvaluatorConfig._merge(self.config, **config_kwargs)
        self.data_dir = os.path.join(self.config.dir_path, "eyeball_data")
        if self.config.api_key is not None:
            self.recorder: EvalRecorder = ApiClientRecorder(
                api_key=self.config.api_key, api_url=self.config.api_url
            )
        elif self.running_in_notebook:
            self.recorder = MemoryRecorder()
        else:
            self.recorder = FileRecorder(self.data_dir)

    @contextmanager
    def start_recording_session(
        self,
        task_name: str,
        checkpoint_id: Optional[str] = None,
        checkpoint_id_to_rerun: Optional[str] = None,
    ) -> Iterator[None]:
        if not self.running_in_notebook and hasattr(
            self.current_recorder_state, "recorder_checkpoint_id"
        ):
            # This should not happen but if it does, we should not overwrite the previous checkpoint id
            # Instead we should set it to None so there is no confusion
            # If the user calls get_eval_params this will raise an error
            self.current_recorder_state.task_name = None
            self.current_recorder_state.recorder_checkpoint_id = None
            self.current_recorder_state.checkpoint_id_to_rerun = None
            yield None
            return

        self.current_recorder_state.task_name = task_name
        if checkpoint_id is not None:
            self.current_recorder_state.recorder_checkpoint_id = checkpoint_id
        else:
            self.current_recorder_state.recorder_checkpoint_id = (
                datetime.datetime.utcnow().isoformat()
            )
        self.current_recorder_state.checkpoint_id_to_rerun = checkpoint_id_to_rerun
        yield None

        del self.current_recorder_state.task_name
        del self.current_recorder_state.recorder_checkpoint_id

    def _should_record(self, **config_kwargs) -> bool:
        if self.mode == EvaluatorMode.COMPARE_CHECKPOINTS:
            return False

        if self.mode == EvaluatorMode.RERUN_EXAMPLES:
            return True

        config = self._get_config(**config_kwargs)
        if config.sample_rate == 0:
            return False
        return config.sample_rate == 1.0 or random.random() < config.sample_rate

    def record_input(
        self,
        variable_name: str,
        value: T,
        task_name: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        **config_kwargs,
    ) -> None:
        if self._should_record(**config_kwargs):
            task_name, checkpoint_id = self._get_recorder_state(
                task_name, checkpoint_id
            )

            serialized_value = self._serialize(value)
            self.recorder.record_input_variable(
                task_name=task_name,
                checkpoint_id=checkpoint_id,
                variable_name=variable_name,
                value=serialized_value,
            )

    def record_output(
        self,
        value: Any,
        task_name: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        **config_kwargs,
    ) -> None:
        if self._should_record(**config_kwargs):
            task_name, checkpoint_id = self._get_recorder_state(
                task_name, checkpoint_id
            )
            self.recorder.record_output(
                task_name=task_name,
                checkpoint_id=checkpoint_id,
                output=self._serialize(value),
            )

    def _get_last_checkpoint(
        self,
        task_name: str,
        input_hash: str,
        current_checkpoint_id: str,
    ) -> Optional[Checkpoint]:
        checkpoints = self.recorder.get_latest_checkpoints(
            task_name, input_hash, num_checkpoints=2
        )
        if len(checkpoints) == 0:
            return None

        if checkpoints[0].checkpoint_id == current_checkpoint_id:
            if len(checkpoints) == 1:
                return None
            return checkpoints[1]

        else:
            return checkpoints[0]

    def _get_recorder_state(
        self,
        task_name: Optional[str] = None,
        recorder_checkpoint_id: Optional[str] = None,
    ) -> tuple[str, str]:
        if task_name is None and hasattr(self.current_recorder_state, "task_name"):
            task_name = self.current_recorder_state.task_name
        if task_name is None:
            raise ValueError(
                "Must provide a task name or start a recording session to record a variable"
            )

        if recorder_checkpoint_id is None and hasattr(
            self.current_recorder_state, "recorder_checkpoint_id"
        ):
            recorder_checkpoint_id = self.current_recorder_state.recorder_checkpoint_id
        if recorder_checkpoint_id is None:
            raise ValueError(
                "Must provide a checkpoint id or start a recording session to record a variable"
            )

        return task_name, recorder_checkpoint_id

    def get_eval_params(
        self,
        task_name: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        task_name, checkpoint_id = self._get_recorder_state(task_name, checkpoint_id)

        checkpoint = self.recorder.get_checkpoint(
            task_name=task_name, checkpoint_id=checkpoint_id
        )
        if checkpoint is None:
            return None
        return checkpoint.eval_params

    def get_eval_param(
        self,
        param_name: str,
        task_name: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[Any]:
        params = self.get_eval_params(task_name=task_name, checkpoint_id=checkpoint_id)
        if params is None:
            return None
        return params.get(param_name)

    def _serialize(self, value: T) -> str:
        if isinstance(value, JsonSerializable):
            return value.to_json()
        else:
            return json.dumps(value)

    def record_task(
        self,
        args_to_record: list[str],
        task_name: Optional[str] = None,
        checkpoint_id_arg_name: Optional[str] = None,
        eval_params: Optional[dict[str, Any]] = None,
        request_user_feedback_probability: float = 0.0,
        **config_kwargs,
    ) -> Callable[..., Callable]:
        def _decorator(fn: Callable[..., T]) -> Callable[..., T]:
            @wraps(fn)
            def _wrapper(*args, **kwargs) -> Any:
                if not self._should_record(**config_kwargs):
                    return fn(*args, **kwargs)

                (
                    fn_arg_names,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = inspect.getfullargspec(fn)

                # Set task name
                if self.mode == EvaluatorMode.RERUN_EXAMPLES:
                    local_task_name = self.current_recorder_state.task_name
                else:
                    local_task_name = task_name or fn.__name__

                # Set checkpoint id
                if self.mode == EvaluatorMode.RERUN_EXAMPLES:
                    recorder_checkpoint_id = (
                        self.current_recorder_state.recorder_checkpoint_id
                    )
                elif checkpoint_id_arg_name is not None:
                    if checkpoint_id_arg_name in kwargs:
                        recorder_checkpoint_id = kwargs[checkpoint_id_arg_name]
                    else:
                        try:
                            checkpoint_id_arg_index = fn_arg_names.index(
                                checkpoint_id_arg_name
                            )
                            recorder_checkpoint_id = args[checkpoint_id_arg_index]
                        except ValueError:
                            recorder_checkpoint_id = (
                                datetime.datetime.utcnow().isoformat()
                            )
                else:
                    recorder_checkpoint_id = datetime.datetime.utcnow().isoformat()

                # Set params
                if self.mode == EvaluatorMode.RERUN_EXAMPLES:
                    params = self.get_eval_params(
                        task_name=local_task_name,
                        checkpoint_id=recorder_checkpoint_id,
                    )
                    # TODO: maybe we should not override with default params in this mode. Can lead to some confusion
                    if (params is None or len(params) == 0) and eval_params is not None:
                        self.recorder.record_eval_params(
                            task_name=local_task_name,
                            checkpoint_id=recorder_checkpoint_id,
                            eval_params=eval_params,
                        )
                elif eval_params is not None:
                    self.recorder.record_eval_params(
                        task_name=local_task_name,
                        checkpoint_id=recorder_checkpoint_id,
                        eval_params=eval_params,
                    )

                for arg_name, arg_val in zip(fn_arg_names, args):
                    if arg_name in args_to_record:
                        self.record_input(
                            task_name=local_task_name,
                            checkpoint_id=recorder_checkpoint_id,
                            variable_name=arg_name,
                            value=arg_val,
                        )

                for kwarg_name, kwarg_val in kwargs.items():
                    if kwarg_name in args_to_record:
                        self.record_input(
                            task_name=local_task_name,
                            checkpoint_id=recorder_checkpoint_id,
                            variable_name=kwarg_name,
                            value=kwarg_val,
                        )

                if self.mode == EvaluatorMode.RERUN_EXAMPLES:
                    result = fn(*args, **kwargs)  # type: ignore
                else:
                    with self.start_recording_session(
                        task_name=local_task_name,
                        checkpoint_id=recorder_checkpoint_id,
                    ):
                        result = fn(*args, **kwargs)  # type: ignore

                self.record_output(
                    task_name=local_task_name,
                    checkpoint_id=recorder_checkpoint_id,
                    value=result,
                )
                if request_user_feedback_probability > 0:
                    if random.random() < request_user_feedback_probability:
                        print("Requesting user feedback")
                        self.request_user_feedback(
                            output=str(result),
                            task_name=local_task_name,
                            checkpoint_id=recorder_checkpoint_id,
                        )
                return result

            return _wrapper

        return _decorator

    def request_user_feedback(
        self,
        output: str,
        task_name: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        print_inputs: bool = False,
    ) -> Optional[OutputFeedback]:
        task_name, checkpoint_id = self._get_recorder_state(task_name, checkpoint_id)
        checkpoint = self.recorder.get_checkpoint(
            task_name=task_name, checkpoint_id=checkpoint_id
        )
        if checkpoint is None:
            return None

        if print_inputs:
            print(checkpoint.get_input_var_str())

        feedback_result = FeedbackResult[
            get_user_input(
                f"What do you think of the result?\n{output}\n",
                choices=[f.name for f in FeedbackResult],
            )
        ]
        details = get_user_input(
            "Any details about your feedback (optional)\n", choices=None
        )

        output_feedback = OutputFeedback(result=feedback_result, message=details)

        self.recorder.record_output_feedback(
            task_name=task_name, checkpoint_id=checkpoint_id, feedback=output_feedback
        )
        return output_feedback

    def rerun_recorded_examples(
        self,
        *eval_params_list: dict[str, Any],
        task_name: Optional[str] = None,
        input_hashes: Optional[list[str]] = None,
    ) -> Iterable[dict[str, Any]]:
        if task_name is None:
            task_names = self.recorder.get_task_names()
            if len(task_names) == 0:
                raise ValueError("No task names found")
            elif len(task_names) == 1:
                task_name = task_names[0]
            else:
                raise ValueError(
                    f"Must provide a task_name as you have multiple tasks recorded -- Found: {task_names}"
                )

        if input_hashes is None or len(input_hashes) == 0:
            input_hashes = self.recorder.get_input_hashes(task_name=task_name)

        try:
            self.mode = EvaluatorMode.RERUN_EXAMPLES
            rerun_metadata = {
                "id": datetime.datetime.utcnow().isoformat(),
            }

            print(f"Will rerun {len(input_hashes)} inputs for task:`{task_name}`")

            for idx, input_hash in enumerate(input_hashes):
                recorder_checkpoint_id = datetime.datetime.utcnow().isoformat()
                last_checkpoint = self._get_last_checkpoint(
                    task_name=task_name,
                    input_hash=input_hash,
                    current_checkpoint_id=recorder_checkpoint_id,
                )
                if last_checkpoint is None:
                    continue

                self.checkpoint_id_to_rerun = last_checkpoint.checkpoint_id
                checkpoint_to_rerun = last_checkpoint
                if checkpoint_to_rerun is None:
                    continue

                input_vars = {
                    k: json.loads(v)
                    for k, v in checkpoint_to_rerun.get_input_variables().items()
                }
                print(
                    f"\nRerunning input #{idx}:\n{checkpoint_to_rerun.get_input_var_str()}"
                )

                if len(eval_params_list) > 0:
                    for eval_params in eval_params_list:
                        with self.start_recording_session(
                            task_name=task_name,
                            checkpoint_id=recorder_checkpoint_id,
                            checkpoint_id_to_rerun=checkpoint_to_rerun.checkpoint_id,
                        ):
                            self.recorder.record_eval_params(
                                task_name=task_name,
                                checkpoint_id=recorder_checkpoint_id,
                                eval_params=eval_params,
                                rerun_metadata=rerun_metadata,
                            )
                            print(f"Using eval params: {eval_params}")
                            yield input_vars
                else:
                    with self.start_recording_session(
                        task_name=task_name,
                        checkpoint_id=recorder_checkpoint_id,
                    ):
                        self.recorder.record_eval_params(
                            task_name=task_name,
                            checkpoint_id=recorder_checkpoint_id,
                            eval_params={},
                            rerun_metadata=rerun_metadata,
                        )
                        yield input_vars
        finally:
            self.mode = EvaluatorMode.RECORD

    def rate_recorded_examples(
        self,
        *input_hashes: str,
        task_name: Optional[str] = None,
    ) -> None:
        if task_name is None:
            task_names = self.recorder.get_task_names()
            if len(task_names) == 0:
                raise ValueError("No task names found")
            elif len(task_names) == 1:
                task_name = task_names[0]
            else:
                raise ValueError(
                    f"Must provide a task_name as you have multiple tasks recorded -- Found: {task_names}"
                )

        input_hashes_lst: list[str] = list(input_hashes)

        if len(input_hashes_lst) == 0:
            input_hashes_lst = self.recorder.get_input_hashes(task_name=task_name)

        try:
            self.mode = EvaluatorMode.RATE_EXAMPLES
            for input_hash in input_hashes_lst:
                already_seen_outputs_to_feedback: dict[str:OutputFeedback] = {}
                checkpoints = self.recorder.get_latest_checkpoints(
                    task_name, input_hash, num_checkpoints=4
                )

                if len(checkpoints) == 0:
                    continue

                print(f"For the inputs:\n{checkpoints[0].get_input_var_str()}")
                for checkpoint in checkpoints:
                    if checkpoint.output_feedback is None:
                        if feedback := already_seen_outputs_to_feedback.get(
                            checkpoint.output
                        ):
                            self.recorder.record_output_feedback(
                                task_name=task_name,
                                checkpoint_id=checkpoint.checkpoint_id,
                                feedback=feedback,
                            )
                        else:
                            feedback = self.request_user_feedback(
                                checkpoint.output or "",
                                task_name=task_name,
                                checkpoint_id=checkpoint.checkpoint_id,
                            )
                            if feedback is not None:
                                already_seen_outputs_to_feedback[
                                    checkpoint.output
                                ] = feedback
                    elif new_feedback := already_seen_outputs_to_feedback.get(
                        checkpoint.output
                    ):
                        if new_feedback.result != checkpoint.output_feedback.result:
                            print(
                                f"For output: {checkpoint.output}\nOld feedback {checkpoint.output_feedback} is different from new feedback {new_feedback}"
                            )
                            print(f"Updating feedback to {new_feedback}")
                            self.recorder.record_output_feedback(
                                task_name=task_name,
                                checkpoint_id=checkpoint.checkpoint_id,
                                feedback=new_feedback,
                            )
                    else:
                        print(f"Already has feedback: {checkpoint.output_feedback}")
        finally:
            self.mode = EvaluatorMode.RECORD

    def record_human_feedback(
        self,
        feedback: OutputFeedback,
        details: Optional[str] = None,
        task_name: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
    ):
        task_name, checkpoint_id = self._get_recorder_state(task_name, checkpoint_id)
        pass

    def delete_checkpoints_for_input_vars(
        self,
        input_vars: dict[str, str],
        task_name: Optional[str] = None,
    ) -> None:
        if task_name is None:
            task_names = self.recorder.get_task_names()
            if len(task_names) == 0:
                raise ValueError("No task names found")
            elif len(task_names) == 1:
                task_name = task_names[0]
            else:
                raise ValueError(
                    f"Must provide a task_name as you have multiple tasks recorded -- Found: {task_names}"
                )
        input_hash = get_input_hash(
            {k: self._serialize(v) for k, v in input_vars.items()}
        )
        self.recorder.delete_checkpoints_for_input_hash(task_name, input_hash)

    def compute_latest_comparison_results(self, task_name: str) -> None:
        input_names: set[str] = set()
        rows: list[dict[str, str]] = []
        comparison_column_names = [
            "latest_checkpoint",
            "previous_checkpoint",
            "the_checkpoint_before",
        ]

        for input_hash in self.recorder.get_input_hashes(task_name):
            # TODO: this is a slow operation
            checkpoints = self.recorder.get_latest_checkpoints(
                task_name, input_hash, num_checkpoints=len(comparison_column_names)
            )
            if len(checkpoints) == 0:
                continue

            checkpoint = checkpoints[0]
            if checkpoint is None:
                continue

            row_data = {}
            input_names.update(checkpoint.input_variables.keys())
            for input_name, input_value in checkpoint.input_variables.items():
                row_data[input_name] = input_value

            # TODO: this is a slow operation, we should probably load all checkpoints and comparison results for an input hash in one iteration
            comparison_results = self.recorder.get_comparison_results_for_input_hash(
                task_name=task_name,
                input_hash=input_hash,
                num_results=len(comparison_column_names),
            )
            sorted_comparison_results = sorted(
                comparison_results,
                key=lambda x: x.newer_checkpoint_id,
                reverse=True,
            )
            for idx, comparison_result in enumerate(sorted_comparison_results):
                msg = str(comparison_result.output_feedback.result)
                new_checkpoint = self.recorder.get_checkpoint(
                    task_name=task_name,
                    checkpoint_id=comparison_result.newer_checkpoint_id,
                )
                if new_checkpoint is not None:
                    if new_checkpoint.output_score is not None:
                        msg += f" (score: {new_checkpoint.output_score})"
                    eval_params_str = ", ".join(
                        f"{k}={v}" for k, v in new_checkpoint.eval_params.items()
                    )
                    if eval_params_str:
                        msg += f" ({eval_params_str})"
                row_data[comparison_column_names[idx]] = msg

            for idx in range(len(comparison_column_names)):
                if (
                    comparison_column_names[idx] not in row_data
                    and idx < len(checkpoints)
                    and (checkpoint := checkpoints[idx]).output_score is not None
                ):
                    msg = f"score: **{checkpoint.output_score.score: .2f}**, {checkpoint.output_score.message}"
                    eval_params_str = ", ".join(
                        f"{k}={v}" for k, v in checkpoint.eval_params.items()
                    )
                    if eval_params_str:
                        msg += f" ({eval_params_str})"
                    row_data[comparison_column_names[idx]] = msg

            rows.append(row_data)

        md_data = []
        table = Table(title="Comparison Results")

        if len(rows) > 0:
            column_names = sorted(list(input_names)) + comparison_column_names
            for column_name in column_names:
                table.add_column(column_name, justify="left")

            md_data.append("| " + " | ".join(column_names) + " |")
            md_data.append("| " + " | ".join(["---"] * len(column_names)) + " |")

            for row in rows:
                row_tuple = tuple(
                    row.get(column_name, "") for column_name in column_names
                )
                table.add_row(*row_tuple)
                md_data.append("| " + " | ".join(row_tuple) + " |")

            task_data_dir = os.path.join(self.data_dir, task_name)
            if not os.path.exists(task_data_dir):
                os.makedirs(task_data_dir)

            with open(
                os.path.join(task_data_dir, "benchmark.md"),
                "w+",
            ) as f:
                f.write("\n".join(md_data))

            console = Console()
            console.print(table)

    def compare_recorded_checkpoints(
        self,
        task_name: Optional[str] = None,
        num_input_hashes: int = 5,
        num_checkpoints_per_input_hash: int = 3,
        task_objective: Optional[str] = None,
        output_comparator: Optional[OutputComparator] = None,
        output_scorer: Optional[OutputScorer] = None,
        use_cached_scores: bool = True,
        use_cached_comparisons: bool = True,
    ) -> None:
        if task_name is None:
            task_names = self.recorder.get_task_names()
            if len(task_names) == 0:
                raise ValueError("No task names found")
            elif len(task_names) == 1:
                task_name = task_names[0]
            else:
                raise ValueError(
                    f"Must provide a task_name as you have multiple tasks recorded -- Found: {task_names}"
                )

        if output_comparator is None and output_scorer is None:
            output_comparator = model_graded_comparator
            assert (
                task_objective is not None
            ), "Must provide an objective to compare or an output comparator"

        input_hashes_list = self.recorder.get_input_hashes(task_name)
        random.shuffle(input_hashes_list)
        input_hashes_list = input_hashes_list[:num_input_hashes]

        print(f"Comparing {len(input_hashes_list)} inputs for task:`{task_name}`")
        try:
            self.mode = EvaluatorMode.COMPARE_CHECKPOINTS

            params_to_succesful_examples: dict[str, int] = defaultdict(int)
            rerun_to_succesful_examples: dict[str, int] = defaultdict(int)

            num_comparisons = 0
            for idx, input_hash in enumerate(input_hashes_list):
                checkpoints_list = self.recorder.get_latest_checkpoints(
                    task_name,
                    input_hash,
                    num_checkpoints=num_checkpoints_per_input_hash,
                )

                checkpoints: dict[str, Checkpoint] = {
                    c.checkpoint_id: c
                    for c in checkpoints_list
                    if c.output is not None
                    and c.output != "null"
                    and c.output != "None"
                }
                checkpoint_ids = sorted(checkpoints.keys(), reverse=True)

                if len(checkpoint_ids) == 0:
                    continue

                scores: dict[str, OutputScore] = {}
                if output_scorer is not None:
                    print(f"\nInput #{idx} - Scoring {len(checkpoint_ids)} checkpoints")
                    for checkpoint_id in checkpoint_ids:
                        checkpoint_to_score = checkpoints[checkpoint_id]
                        if (
                            use_cached_scores
                            and checkpoint_to_score.output_score is not None
                        ):
                            print(
                                f"Using cached score for {checkpoint_id}: {checkpoint_to_score.output_score}"
                            )
                            scores[checkpoint_id] = checkpoint_to_score.output_score
                            continue

                        assert checkpoint_to_score.output is not None
                        score = output_scorer(
                            task_objective or "",
                            checkpoint_to_score.get_input_variables(),
                            checkpoint_to_score.output,
                        )
                        self.recorder.record_output_score(
                            task_name=task_name,
                            checkpoint_id=checkpoint_id,
                            score=score,
                        )
                        scores[checkpoint_id] = score
                        print(f"Scored {checkpoint_id}: {score}")

                if len(checkpoint_ids) < 2:
                    continue

                # Newer checkpoints are at the start of the list so for every comparison
                # we compare the newer_checkpoint_id with the older_checkpoint_id
                to_compare = [
                    (checkpoint_ids[i], checkpoint_ids[i + 1])
                    for i in range(len(checkpoint_ids) - 1)
                ]

                print(f"\nInput #{idx} - Running {len(to_compare)} comparison(s)")

                for newer_checkpoint_id, older_checkpoint_id in to_compare:
                    newer_checkpoint = self.recorder.get_checkpoint(
                        task_name=task_name,
                        checkpoint_id=newer_checkpoint_id,
                    )
                    older_checkpoint = self.recorder.get_checkpoint(
                        task_name=task_name,
                        checkpoint_id=older_checkpoint_id,
                    )

                    if newer_checkpoint is None or older_checkpoint is None:
                        print(
                            f"Could not find checkpoint {newer_checkpoint_id} or {older_checkpoint_id}"
                        )
                        continue

                    newer_checkpoint_output = newer_checkpoint.output
                    older_checkpoint_output = older_checkpoint.output

                    assert newer_checkpoint_output is not None
                    assert older_checkpoint_output is not None

                    should_record_comparison = True
                    if use_cached_comparisons and (
                        comparator_result := self.recorder.get_comparison_result(
                            task_name=task_name,
                            older_checkpoint_id=older_checkpoint_id,
                            newer_checkpoint_id=newer_checkpoint_id,
                        )
                    ):
                        print(f"Using cached comparison result for {input_hash}")
                        should_record_comparison = False
                        output_comparison_feedback = comparator_result.output_feedback
                    elif output_comparator is not None:
                        output_comparison_feedback = output_comparator(
                            objective=task_objective or "",
                            input_variables=newer_checkpoint.get_input_variables(),
                            older_checkpoint_output=older_checkpoint_output,
                            newer_checkpoint_output=newer_checkpoint_output,
                        )
                    elif output_scorer is not None:
                        output_comparison_feedback = output_feedback_from_scores(
                            older_score=scores[older_checkpoint_id],
                            newer_score=scores[newer_checkpoint_id],
                        )
                    else:
                        raise ValueError(
                            "Should not happen. We need an output comparator or output scorer"
                        )

                    new_unique_params = []
                    old_unqiue_params = []
                    for key in (
                        newer_checkpoint.eval_params.keys()
                        | older_checkpoint.eval_params.keys()
                    ):
                        param_val_a = newer_checkpoint.eval_params.get(key)
                        param_val_b = older_checkpoint.eval_params.get(key)
                        if param_val_a != param_val_b:
                            if param_val_a is not None:
                                new_unique_params.append(f"{key}={param_val_a}")
                            elif param_val_b is not None:
                                old_unqiue_params.append(f"{key}={param_val_b}")

                    new_unique_params_str = (
                        ("(" + ", ".join(new_unique_params) + ")")
                        if len(new_unique_params) > 0
                        else ""
                    )
                    old_unique_params_str = (
                        ("(" + ", ".join(old_unqiue_params) + ")")
                        if len(old_unqiue_params) > 0
                        else ""
                    )

                    new_params_str = ",".join(
                        f"{k}={v}" for k, v in newer_checkpoint.eval_params.items()
                    )
                    old_params_str = ",".join(
                        f"{k}={v}" for k, v in older_checkpoint.eval_params.items()
                    )

                    rerun_id_new = (
                        newer_checkpoint.rerun_metadata.get("id")
                        if newer_checkpoint.rerun_metadata is not None
                        else None
                    )
                    rerun_id_old = (
                        older_checkpoint.rerun_metadata.get("id")
                        if older_checkpoint.rerun_metadata is not None
                        else None
                    )

                    num_comparisons += 1

                    if output_comparison_feedback.result == FeedbackResult.NEUTRAL:
                        print(
                            f"{ORANGE}[neutral] task output is the same between checkpoints {older_checkpoint_id} {old_unique_params_str} & {newer_checkpoint_id} {new_unique_params_str} {END_CLR}"
                        )
                        params_to_succesful_examples[new_params_str] += 1
                        params_to_succesful_examples[old_params_str] += 1
                        if rerun_id_new is not None:
                            rerun_to_succesful_examples[rerun_id_new] += 1
                        if rerun_id_old is not None:
                            rerun_to_succesful_examples[rerun_id_old] += 1
                    elif output_comparison_feedback.result == FeedbackResult.NEGATIVE:
                        print(
                            f"{RED}[regression] task output was better in the older checkpoint {older_checkpoint_id} {old_unique_params_str} than {newer_checkpoint_id} {new_unique_params_str} {END_CLR}"
                        )
                        params_to_succesful_examples[old_params_str] += 1
                        if rerun_id_old is not None:
                            rerun_to_succesful_examples[rerun_id_old] += 1
                    else:
                        print(
                            f"{GREEN}[improvement] task output improved from checkpoint {older_checkpoint_id} {old_unique_params_str} to {newer_checkpoint_id} {new_unique_params_str}{END_CLR}"
                        )
                        params_to_succesful_examples[new_params_str] += 1
                        if rerun_id_new is not None:
                            rerun_to_succesful_examples[rerun_id_new] += 1
                    if should_record_comparison:
                        self.recorder.record_comparison_result(
                            task_name=task_name,
                            input_hash=input_hash,
                            result=ComparisonResult(
                                older_checkpoint_id=older_checkpoint_id,
                                newer_checkpoint_id=newer_checkpoint_id,
                                output_feedback=output_comparison_feedback,
                            ),
                        )

            print("\nSummary:")
            print("---------")
            if len(rerun_to_succesful_examples) > 0:
                print("Your most sucessful re-runs:")
                for rerun_id, num_successes in sorted(
                    rerun_to_succesful_examples.items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    print(
                        f"{rerun_id}: {num_successes}/{len(input_hashes_list)} successes"
                    )

            print("\nYour most sucessful params:")
            for params, num_successes in sorted(
                params_to_succesful_examples.items(), key=lambda x: x[1], reverse=True
            ):
                if params == "":
                    params = "default"
                print(f"{params}: {num_successes}/{num_comparisons} successes")

            self.compute_latest_comparison_results(task_name)

        finally:
            self.mode = EvaluatorMode.RECORD


_default_evaluator = Evaluator()


set_config = _default_evaluator.set_config
record_task = _default_evaluator.record_task
record_output = _default_evaluator.record_output
record_input = _default_evaluator.record_input
get_eval_param = _default_evaluator.get_eval_param
get_eval_params = _default_evaluator.get_eval_params
rerun_recorded_examples = _default_evaluator.rerun_recorded_examples
rate_recorded_examples = _default_evaluator.rate_recorded_examples
compare_recorded_checkpoints = _default_evaluator.compare_recorded_checkpoints
delete_checkpoints_for_input_vars = _default_evaluator.delete_checkpoints_for_input_vars
