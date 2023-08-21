from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from enum import Enum
import inspect
import json
import os
import types
import logging
from rich import print
import subprocess

from eyeball_pp.graders import model_based_grader
from .recorders import (
    ApiClientRecorder,
    Checkpoint,
    ComparisonResult,
    FileRecorder,
    MemoryRecorder,
    EvalRecorder,
    get_input_hash,
)
from .comparators import model_graded_comparator, output_feedback_from_scores
from .classes import (
    Criteria,
    FeedbackResult,
    MultiOutputFeedback,
    MultiOutputScores,
    OutputComparator,
    OutputGrader,
    OutputScore,
    OutputFeedback,
    EvaluatorConfig,
    TASK_OUTPUT_KEY,
)

import random
from statistics import variance
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
import datetime
from .utils import get_score_map, get_user_input, output_table, time_to_str
from .system_state import bucketize_checkpoints, get_system_tags
from tqdm import tqdm

GREEN = "\x1b[32m"
ORANGE = "\x1b[33m"
RED = "\x1b[31m"
BOLD = "\x1b[1m"
UNDERLINE = "\x1b[4m"
ITALIC = "\x1b[3m"
HEADING_BG = "\x1b[103m"
SUCCESS_CUTOFF = 0.5
END_CLR = "\x1b[0m"

logger = logging.getLogger(__name__)


@runtime_checkable
class JsonSerializable(Protocol):
    def to_json(self) -> str:
        ...


T = TypeVar("T", int, float, str, bool, bytes, dict, list, None, JsonSerializable)


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

    def get_recorder(self) -> EvalRecorder:
        return self.recorder
    
    def get_config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def set_config(self, **config_kwargs) -> None:
        self.config = EvaluatorConfig._merge(self.config, **config_kwargs)
        self.data_dir = os.path.join(self.config.dir_path, "eyeball_data")

        if self.config.record_in_memory:
            self.recorder: EvalRecorder = MemoryRecorder()
        elif self.config.api_key is not None:
            self.recorder = ApiClientRecorder(
                api_key=self.config.api_key, api_url=self.config.api_url
            )
        elif api_key := os.environ.get("EYEBALLPP_API_KEY"):
            self.recorder = ApiClientRecorder(api_key=api_key, api_url=self.config.api_url)
        elif self.config.sample_rate == 0:
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
            if self.mode == EvaluatorMode.RERUN_EXAMPLES:
                yield None
                return

            # This should not happen but if it does, we should not overwrite the previous checkpoint id
            # Instead we should set it to None so there is no confusion
            # If the user calls get_eval_params this will raise an error
            self.current_recorder_state.task_name = None
            self.current_recorder_state.recorder_checkpoint_id = None
            self.current_recorder_state.checkpoint_id_to_rerun = None
            yield None
            return

        self.current_recorder_state.task_name = task_name
        if checkpoint_id is None:
            checkpoint_id = datetime.datetime.utcnow().isoformat()

        self.current_recorder_state.recorder_checkpoint_id = checkpoint_id
        self.current_recorder_state.checkpoint_id_to_rerun = checkpoint_id_to_rerun
        yield None

        for tag in get_system_tags():
            self.recorder.add_checkpoint_tag(
                task_name=task_name,
                checkpoint_id=checkpoint_id,
                tag=tag,
            )

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

    def record_intermediary_state(
        self,
        state_name: str,
        state_value: str,
        task_name: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        **config_kwargs,
    ) -> None:
        if self._should_record(**config_kwargs):
            task_name, checkpoint_id = self._get_recorder_state(
                task_name, checkpoint_id
            )

            self.recorder.record_intermediary_state(
                task_name=task_name,
                checkpoint_id=checkpoint_id,
                state_name=state_name,
                state_value=state_value,
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

    def _get_recorded_task_name(self, task_name: Optional[str]) -> str:
        if task_name is not None:
            return task_name
        if hasattr(self.current_recorder_state, "task_name"):
            return self.current_recorder_state.task_name

        task_names = self.recorder.get_task_names()
        if len(task_names) == 0:
            raise ValueError("No task names found")
        elif len(task_names) == 1:
            return task_names[0]
        else:
            raise ValueError(
                f"Must provide a task_name as you have multiple tasks recorded -- Found: {task_names}"
            )

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
            task_name=task_name,
            checkpoint_id=checkpoint_id,
            feedback=MultiOutputFeedback({TASK_OUTPUT_KEY: output_feedback}),
        )
        return output_feedback

    def rerun_recorded_examples(
        self,
        *eval_params_list: dict[str, Any],
        task_name: Optional[str] = None,
        input_hashes: Optional[list[str]] = None,
        limit: Optional[int] = None,
        randomize: Optional[bool] = False,
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

        if randomize and input_hashes:
            random.shuffle(input_hashes)

        if limit is not None:
            input_hashes = input_hashes[:limit]

        try:
            self.mode = EvaluatorMode.RERUN_EXAMPLES
            rerun_metadata = {
                "id": datetime.datetime.utcnow().isoformat(),
            }

            print(f"Will rerun {len(input_hashes)} inputs for task:`{task_name}`")

            for idx, input_hash in tqdm(enumerate(input_hashes)):
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
                logger.debug(
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
                            logger.debug(f"Using eval params: {eval_params}")
                            yield input_vars
                else:
                    with self.start_recording_session(
                        task_name=task_name,
                        checkpoint_id=recorder_checkpoint_id,
                        checkpoint_id_to_rerun=checkpoint_to_rerun.checkpoint_id,
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
                already_seen_outputs_to_feedback: dict[str, MultiOutputFeedback] = {}
                checkpoints = self.recorder.get_latest_checkpoints(
                    task_name, input_hash, num_checkpoints=4
                )

                if len(checkpoints) == 0:
                    continue

                print(f"\nFor inputs:\n{checkpoints[0].get_input_var_str()}")
                for checkpoint in checkpoints:
                    if checkpoint.output is None:
                        continue

                    if not checkpoint.feedback:
                        if feedback := already_seen_outputs_to_feedback.get(
                            checkpoint.output
                        ):
                            self.recorder.record_output_feedback(
                                task_name=task_name,
                                checkpoint_id=checkpoint.checkpoint_id,
                                feedback=feedback,
                            )
                        else:
                            output_feedback = self.request_user_feedback(
                                checkpoint.output or "",
                                task_name=task_name,
                                checkpoint_id=checkpoint.checkpoint_id,
                            )
                            if output_feedback is not None:
                                already_seen_outputs_to_feedback[
                                    checkpoint.output
                                ] = MultiOutputFeedback(
                                    {TASK_OUTPUT_KEY: output_feedback}
                                )
                    elif new_feedback := already_seen_outputs_to_feedback.get(
                        checkpoint.output
                    ):
                        if (
                            new_feedback[TASK_OUTPUT_KEY].result
                            != checkpoint.feedback[TASK_OUTPUT_KEY].result
                        ):
                            print(
                                f"For output: {checkpoint.output}\nOld feedback {checkpoint.feedback[TASK_OUTPUT_KEY]} is different from new feedback {new_feedback[TASK_OUTPUT_KEY]}"
                            )
                            print(f"Updating feedback to {new_feedback}")
                            self.recorder.record_output_feedback(
                                task_name=task_name,
                                checkpoint_id=checkpoint.checkpoint_id,
                                feedback=new_feedback,
                            )
                    else:
                        print(
                            f"Output: {checkpoint.output[:140]} already has feedback: {checkpoint.feedback.get(TASK_OUTPUT_KEY)}"
                        )
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

    def evaluate_system(
        self,
        task_name: Optional[str] = None,
        num_input_hashes: int = 5,
        num_checkpoints_per_input_hash: int = 3,
        task_objective: Optional[str] = None,
        grading_criteria: Optional[list[Criteria]] = None,
        grading_criteria_custom: Optional[dict[str, str]] = None,
        output_comparator: Optional[OutputComparator] = None,
        output_grader: Optional[OutputGrader] = None,
        intermediate_objectives: Optional[dict[str, str]] = None,
        use_cached_scores: bool = True,
        use_cached_comparisons: bool = True,
    ) -> None:
        task_name = self._get_recorded_task_name(task_name)

        if output_grader is None:
            output_grader = model_based_grader

        input_hashes_list = self.recorder.get_input_hashes(task_name)
        random.shuffle(input_hashes_list)
        input_hashes_list = input_hashes_list[:num_input_hashes]

        print(f"Evaluating {len(input_hashes_list)} inputs for task:`{task_name}`")
        try:
            self.mode = EvaluatorMode.COMPARE_CHECKPOINTS

            num_comparisons = 0
            for idx, input_hash in enumerate(
                tqdm(
                    input_hashes_list,
                    bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt}",
                )
            ):
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

                if output_grader is not None:
                    # TODO: change output scorer to score multiple output types
                    logger.debug(
                        f"\nInput #{idx} - Grading {len(checkpoint_ids)} checkpoints"
                    )
                    for checkpoint_id in tqdm(
                        checkpoint_ids,
                        desc=f"Checkpoints for Input #{idx}",
                        bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt}",
                        leave=False,
                    ):
                        checkpoint_to_score = checkpoints[checkpoint_id]
                        if (
                            use_cached_scores
                            and checkpoint_to_score.scores
                            and TASK_OUTPUT_KEY in checkpoint_to_score.scores
                        ):
                            logger.debug(
                                f"Using cached score for {checkpoint_id}: {checkpoint_to_score.scores[TASK_OUTPUT_KEY].score}"
                            )
                            continue

                        assert checkpoint_to_score.output is not None
                        output_score = output_grader(
                            input_variables=checkpoint_to_score.get_input_variables(),
                            output=checkpoint_to_score.output,
                            intermediary_state=checkpoint_to_score.intermediary_state,
                            objective=task_objective,
                            criteria=grading_criteria,
                            custom_criteria=grading_criteria_custom,
                        )
                        multi_output_scores = MultiOutputScores(
                            {TASK_OUTPUT_KEY: output_score}
                        )
                        self.recorder.record_output_scores(
                            task_name=task_name,
                            checkpoint_id=checkpoint_id,
                            scores=multi_output_scores,
                        )
                        checkpoint_to_score.scores = multi_output_scores
                        logger.debug(f"Scored {checkpoint_id}: {output_score.score}")

                if len(checkpoint_ids) < 2:
                    continue

                # Newer checkpoints are at the start of the list so for every comparison
                # we compare the newer_checkpoint_id with the older_checkpoint_id
                to_compare = [
                    (checkpoint_ids[i], checkpoint_ids[i + 1])
                    for i in range(len(checkpoint_ids) - 1)
                ]

                logger.debug(
                    f"\nInput #{idx} - Running {len(to_compare)} comparison(s)"
                )

                for newer_checkpoint_id, older_checkpoint_id in tqdm(
                    to_compare, desc="Comparisons", disable=True
                ):
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
                        logger.debug(f"Using cached comparison result for {input_hash}")
                        should_record_comparison = False
                        comparison_feedback = comparator_result.feedback
                    elif output_comparator is not None:
                        comparison_feedback = output_comparator(
                            input_variables=newer_checkpoint.get_input_variables(),
                            older_checkpoint_output=older_checkpoint_output,
                            newer_checkpoint_output=newer_checkpoint_output,
                            task_objective=task_objective or "",
                            objectives_intermediary_state=intermediate_objectives,
                            older_checkpoint_intermediary_state=older_checkpoint.intermediary_state,
                            newer_checkpoint_intermediary_state=newer_checkpoint.intermediary_state,
                        )
                    elif output_grader is not None:
                        comparison_feedback = output_feedback_from_scores(
                            older_scores=older_checkpoint.scores,
                            newer_scores=newer_checkpoint.scores,
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

                    num_comparisons += 1
                    output_comparison_feedback = comparison_feedback[TASK_OUTPUT_KEY]
                    if output_comparison_feedback.result == FeedbackResult.NEUTRAL:
                        logger.debug(
                            f"{ORANGE}[neutral] task output is the same between checkpoints {older_checkpoint_id} {old_unique_params_str} & {newer_checkpoint_id} {new_unique_params_str} {END_CLR}"
                        )
                    elif output_comparison_feedback.result == FeedbackResult.NEGATIVE:
                        logger.debug(
                            f"{RED}[regression] task output was better in the older checkpoint {older_checkpoint_id} {old_unique_params_str} than {newer_checkpoint_id} {new_unique_params_str} {END_CLR}"
                        )
                    else:
                        logger.debug(
                            f"{GREEN}[improvement] task output improved from checkpoint {older_checkpoint_id} {old_unique_params_str} to {newer_checkpoint_id} {new_unique_params_str}{END_CLR}"
                        )
                    if should_record_comparison:
                        self.recorder.record_comparison_result(
                            task_name=task_name,
                            input_hash=input_hash,
                            result=ComparisonResult(
                                older_checkpoint_id=older_checkpoint_id,
                                newer_checkpoint_id=newer_checkpoint_id,
                                feedback=comparison_feedback,
                            ),
                        )

            self.calculate_system_health(task_name=task_name)
            self.show_data_via_streamlit(task_name=task_name)
        finally:
            self.mode = EvaluatorMode.RECORD

    def show_data_via_streamlit(self, task_name: str) -> None:
        "Run streamlit server in a subprocess"
        server_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
        print(server_file)
        config_dict = json.dumps(self.get_config_dict())
        subprocess.Popen(
            f"python -m streamlit run {server_file} -- --task_name={task_name} --eyeball_config='{config_dict}'",
            shell=True,
        )


    def calculate_system_health(
        self,
        num_samples: int = 10,
        plotting_frequency_in_days: int = 1,
        task_name: Optional[str] = None,
    ) -> None:
        task_name = self._get_recorded_task_name(task_name=task_name)
        output_names_to_score: set[str] = set([TASK_OUTPUT_KEY])

        # calculate rolling average if we have output scores for the last num_samples checkpoints
        input_hashes = self.recorder.get_input_hashes(task_name=task_name)
        if len(input_hashes) == 0:
            print("No input hashes exist for this task")

        all_checkpoints: list[Checkpoint] = []
        input_hash_to_checkpoints: dict[str, list[Checkpoint]] = {}
        for input_hash in input_hashes:
            checkpoints = self.recorder.get_latest_checkpoints(
                task_name=task_name,
                input_hash=input_hash,
                num_checkpoints=num_samples,
            )
            input_hash_to_checkpoints[input_hash] = checkpoints
            all_checkpoints += checkpoints

        all_checkpoints_dict: dict[str, Checkpoint] = {
            c.checkpoint_id: c for c in all_checkpoints
        }

        if len(all_checkpoints) == 0:
            return

        # If the outputs have scores, we can calculate a rolling average
        scored_checkpoints: list[Checkpoint] = []
        rerunid_to_checkpoint_feedback: dict[
            str, dict[str, dict[str, Optional[FeedbackResult]]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))

        for c in all_checkpoints:
            scored_outputs = set(c.scores.keys())
            if len(scored_outputs) > 0:
                scored_checkpoints.append(c)
                output_names_to_score |= scored_outputs

        if len(scored_checkpoints) == 0:
            # If the outputs don't have scores, we can give the better output a score of 1 and the worse output a score of 0.5 and propagate that score to the older checkpoints
            comparison_results: list[ComparisonResult] = []
            for input_hash in input_hashes:
                comparison_results += (
                    self.recorder.get_comparison_results_for_input_hash(
                        task_name=task_name,
                        input_hash=input_hash,
                        num_results=num_samples,
                    )
                )
            for cr in comparison_results:
                print(f"Feedback: {cr.feedback.keys()}")
                output_names_to_score |= set(cr.feedback.keys())
            for output_name in output_names_to_score:
                edges: dict[str, dict[str, float]] = defaultdict(lambda: dict())
                for comparison_result in comparison_results:
                    feedback = comparison_result.feedback.get(output_name)
                    if feedback is None:
                        continue

                    if (
                        new_checkpoint := all_checkpoints_dict.get(
                            comparison_result.newer_checkpoint_id
                        )
                    ) is not None:
                        if new_checkpoint.rerun_metadata:
                            rerunid_to_checkpoint_feedback[
                                new_checkpoint.rerun_metadata["id"]
                            ][comparison_result.newer_checkpoint_id][
                                output_name
                            ] = feedback.result

                    if feedback.result == FeedbackResult.POSITIVE:
                        edges[comparison_result.older_checkpoint_id][
                            comparison_result.newer_checkpoint_id
                        ] = 1.0
                    elif feedback.result == FeedbackResult.NEGATIVE:
                        edges[comparison_result.newer_checkpoint_id][
                            comparison_result.older_checkpoint_id
                        ] = 1.0
                    else:
                        edges[comparison_result.newer_checkpoint_id][
                            comparison_result.older_checkpoint_id
                        ] = 0.0
                        edges[comparison_result.older_checkpoint_id][
                            comparison_result.newer_checkpoint_id
                        ] = 0.0
                scores = get_score_map(edges)
                denom = max(scores.values()) if len(scores) > 0 else 0.0
                for checkpoint_id, score in scores.items():
                    # TODO: fetch checkpoint if it doesn't exist
                    # TODO: change input hash to checkpoints dict too
                    if checkpoint := all_checkpoints_dict.get(checkpoint_id):
                        checkpoint.scores[output_name] = OutputScore(
                            score=(score / denom) if denom > 0 else 0.0,
                            message="",
                        )

        scored_checkpoints = [c for c in all_checkpoints if len(c.scores) > 0]

        if len(scored_checkpoints) == 0:
            print(f"Not enough checkpoints with comparisons to calculate system health")
            return
        scored_checkpoints.sort(key=lambda x: x.checkpoint_id, reverse=True)
        system_health_by_date: list[dict[str, Any]] = []
        date_to_use = datetime.datetime.utcnow().date()

        system_health_by_date = []
        while scored_checkpoints[-1].created_at.date() <= date_to_use:
            num_successes = 0.0
            num_checkpoints_used = 0
            input_hash_set = set()
            for checkpoint in scored_checkpoints:
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
                        input_hash_set.add(checkpoint.get_input_hash())

            system_health_by_date.append(
                {
                    "Date": time_to_str(date_to_use),
                    "Results": f"{float(num_successes) / float(num_checkpoints_used) * 100.0: .1f}% success ({num_successes}/{num_checkpoints_used})",
                    "Stats": f"{num_checkpoints_used} datapoints, {len(input_hash_set)} unique inputs",
                }
            )
            date_to_use -= datetime.timedelta(days=plotting_frequency_in_days)

        output_table(
            system_health_by_date,
            title=f"System health for task: '{task_name}' (by Date)",
            markdown_file=os.path.join(
                self.data_dir, task_name, "system_health_by_date.md"
            ),
        )
        buckets_to_checkpoints = bucketize_checkpoints(scored_checkpoints)
        system_health_by_run_history = []
        for bucket, checkpoints in buckets_to_checkpoints.items():
            row = {"Run": str(bucket)}

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
                        total_variance = 0.0
                        num_inputs_with_variance = 0
                        for input_hash, score_list in input_hash_to_score.items():
                            if len(score_list) > 1:
                                total_variance += variance(score_list)
                                num_inputs_with_variance += 1
                        if num_inputs_with_variance > 0:
                            row[
                                "Output Variance (higher value ⇒ unpredictable system)"
                            ] = str(total_variance / float(num_inputs_with_variance))
                        if len(params_used) == 1:
                            row["Params"] = params_used.pop()
            if len(row) > 1:
                system_health_by_run_history.append(row)
        output_table(
            system_health_by_run_history,
            title=f"System health for task: '{task_name}' (by Run History)",
            markdown_file=os.path.join(
                self.data_dir, task_name, "system_health_by_run_history.md"
            ),
        )

        # Group by reruns if they exist
        rerun_ids_to_score: dict[str, list[Checkpoint]] = defaultdict(list)
        for checkpoint in scored_checkpoints:
            if rerun_id := checkpoint.rerun_metadata.get("id"):
                rerun_ids_to_score[rerun_id].append(checkpoint)

        # rerun_rows = []
        # for rerun_id, checkpoints in sorted(
        #     rerun_ids_to_score.items(), key=lambda x: x[0], reverse=True
        # ):
        #     # We want to show how many inputs performed better in this re-run ideally
        #     # For now let's show score and then change it up
        #     row: dict[str, Any] = {"rerun_id": rerun_id}

        #     # num_better, num_worse, num_same
        #     output_feedback_status: dict[str, list[float]] = defaultdict(
        #         lambda: [0.0, 0.0, 0.0, 0.0]
        #     )
        #     all_output_names: set[str] = set()
        #     rerun_input_hashes: set[str] = set()
        #     for checkpoint in checkpoints:
        #         rerun_input_hashes.add(checkpoint.get_input_hash())
        #         for output_name, output_score in checkpoint.scores.items():
        #             all_output_names.add(output_name)
        #             if (
        #                 feedback_result := rerunid_to_checkpoint_feedback[rerun_id][
        #                     checkpoint.checkpoint_id
        #                 ]
        #             ) is not None:
        #                 if feedback_result == FeedbackResult.POSITIVE:
        #                     output_feedback_status[output_name][0] += 1.0
        #                 elif feedback_result == FeedbackResult.NEGATIVE:
        #                     output_feedback_status[output_name][1] += 1.0
        #                 else:
        #                     output_feedback_status[output_name][2] += 1.0

        #             output_feedback_status[output_name][3] += output_score.score

        #     for output_name in all_output_names:
        #         num_better, num_worse, num_same, score = output_feedback_status[
        #             output_name
        #         ]
        #         total = int(num_better + num_worse + num_same)
        #         trend_msg = ""
        #         if num_better > 0:
        #             trend_msg += f"{int(num_better)}/{total} got better\n"
        #         if num_worse > 0:
        #             trend_msg += f"{int(num_worse)}/{total} got worse\n"
        #         if num_same > 0:
        #             trend_msg += f"{int(num_same)}/{total} stayed the same\n"

        #         if trend_msg:
        #             row[f"{output_name} trend"] = trend_msg
        #         else:
        #             row[f"{output_name} relative score"] = score

        #     row["num_checkpoints_used"] = len(checkpoints)
        #     row["input_diversity"] = len(input_hashes)

        #     rerun_rows.append(row)
        # output_table(rerun_rows, title="Rerun stats")

        for output_name in output_names_to_score:
            input_specific_rows: list[dict[str, str]] = []
            for input_hash, checkpoints in input_hash_to_checkpoints.items():
                best_checkpoint: Optional[Checkpoint] = None
                most_recent_checkpoint: Optional[Checkpoint] = None
                worst_checkpoint: Optional[Checkpoint] = None

                for checkpoint in checkpoints:
                    if output_name not in checkpoint.scores:
                        continue

                    if (
                        best_checkpoint is None
                        or checkpoint.scores[output_name].score
                        > best_checkpoint.scores[output_name].score
                    ):
                        best_checkpoint = checkpoint
                    if (
                        most_recent_checkpoint is None
                        or checkpoint.created_at > most_recent_checkpoint.created_at
                    ):
                        most_recent_checkpoint = checkpoint
                    if (
                        worst_checkpoint is None
                        or checkpoint.scores[output_name].score
                        < worst_checkpoint.scores[output_name].score
                    ):
                        worst_checkpoint = checkpoint

                row_data = dict(checkpoint.input_variables)
                if best_checkpoint is not None:
                    row_data["best_checkpoint"] = best_checkpoint.output_score_repr(
                        output_name=output_name
                    )
                if most_recent_checkpoint is not None:
                    row_data[
                        "most_recent_checkpoint"
                    ] = most_recent_checkpoint.output_score_repr(output_name)
                if worst_checkpoint is not None:
                    row_data["worst_checkpoint"] = worst_checkpoint.output_score_repr(
                        output_name=output_name
                    )
                input_specific_rows.append(row_data)
            if output_name == TASK_OUTPUT_KEY:
                per_input_filename = os.path.join(
                    self.data_dir, task_name, "per_input_breakdown.md"
                )
                output_table(
                    input_specific_rows,
                    title=f"Task success breakdown by input",
                    markdown_file=per_input_filename,
                    print_table=False,
                )
                abs_path = os.path.abspath(per_input_filename)
                file_link = f"[link=file://{abs_path}]per_input_breakdown.md[/link]"
                print(f"\nA per input breakdown can be found here: {file_link}")
            else:
                output_table(
                    input_specific_rows,
                    title=f"Intermediary state: {output_name} breakdown by input",
                )

    def cleanup_old_checkpoints(
        self,
        task_name: Optional[str] = None,
        ask_before_delete: bool = True,
        days_to_look_back: int = 4,
    ) -> None:
        task_name = self._get_recorded_task_name(task_name=task_name)

        tags = get_system_tags()
        if len(tags) == 0:
            return

        checkpoints = self.recorder.get_checkpoints_by_tags(task_name, tags)
        if len(checkpoints) == 0:
            print(f"No checkpoints found for tag combination: {tags}")
            return

        input_hash_to_checkpoints: dict[str, list[Checkpoint]] = defaultdict(list)
        time_since = datetime.datetime.utcnow() - datetime.timedelta(
            days=days_to_look_back
        )
        for checkpoint in checkpoints:
            if checkpoint.created_at > time_since:
                input_hash_to_checkpoints[checkpoint.get_input_hash()].append(
                    checkpoint
                )

        for input_hash in input_hash_to_checkpoints.keys():
            checkpoints = input_hash_to_checkpoints[input_hash]
            checkpoints.sort(key=lambda x: x.created_at, reverse=True)
            if len(checkpoints) == 1:
                continue

            num_deleted = 0
            if ask_before_delete:
                print(
                    f"\nFound {len(checkpoints)} checkpoints for input hash: {input_hash} for tags: {tags}"
                )
                val = input(
                    f"Enter 'Y' to delete {len(checkpoints) - 1} checkpoints and keep most recent checkpoint -- {checkpoints[0].checkpoint_id}\n"
                )
                if val != "Y":
                    print("Skipping deletion")
                    continue

            for checkpoint in checkpoints[1:]:
                self.recorder.delete_checkpoint(task_name, checkpoint.checkpoint_id)
                num_deleted += 1
            print(
                f"Deleted {num_deleted} checkpoints for input hash: {input_hash} for tags: {tags}"
            )


_default_evaluator = Evaluator()


set_config = _default_evaluator.set_config
record_task = _default_evaluator.record_task
record_output = _default_evaluator.record_output
record_input = _default_evaluator.record_input
get_eval_param = _default_evaluator.get_eval_param
get_eval_params = _default_evaluator.get_eval_params
rerun_recorded_examples = _default_evaluator.rerun_recorded_examples
rate_recorded_examples = _default_evaluator.rate_recorded_examples
evaluate_system = _default_evaluator.evaluate_system
delete_checkpoints_for_input_vars = _default_evaluator.delete_checkpoints_for_input_vars
calculate_system_health = _default_evaluator.calculate_system_health
record_intermediary_state = _default_evaluator.record_intermediary_state
start_recording_session = _default_evaluator.start_recording_session
default_evaluator = _default_evaluator
get_default_recorder = _default_evaluator.get_recorder
cleanup_old_checkpoints = _default_evaluator.cleanup_old_checkpoints
get_config_dict = _default_evaluator.get_config_dict