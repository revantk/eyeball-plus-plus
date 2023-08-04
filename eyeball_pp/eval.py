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
    FileRecorder,
    MemoryRecorder,
    EvalRecorder,
    get_input_hash,
)
from .comparators import model_graded_comparator, output_feedback_from_scores
from .classes import (
    FeedbackResult,
    MultiOutputFeedback,
    OutputComparator,
    OutputScore,
    OutputScorer,
    OutputFeedback,
    TASK_OUTPUT_KEY,
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
from .utils import get_score_map, get_user_input, output_table

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


T = TypeVar("T", int, float, str, bool, bytes, dict, list, None, JsonSerializable)


@dataclass
class EvaluatorConfig:
    sample_rate: float = 1.0
    dir_path: str = "."
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    record_in_memory: bool = False

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

        if self.config.record_in_memory:
            self.recorder: EvalRecorder = MemoryRecorder()
        elif self.config.api_key is not None:
            self.recorder = ApiClientRecorder(
                api_key=self.config.api_key, api_url=self.config.api_url
            )
        elif self.running_in_notebook or self.config.sample_rate == 0:
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
            print(f"Input hashes {input_hashes}")

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

    def compare_recorded_checkpoints(
        self,
        task_name: Optional[str] = None,
        num_input_hashes: int = 5,
        num_checkpoints_per_input_hash: int = 3,
        task_objective: Optional[str] = None,
        output_comparator: Optional[OutputComparator] = None,
        output_scorer: Optional[OutputScorer] = None,
        intermediate_objectives: Optional[dict[str, str]] = None,
        use_cached_scores: bool = True,
        use_cached_comparisons: bool = True,
    ) -> None:
        task_name = self._get_recorded_task_name(task_name)

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

                if output_scorer is not None:
                    # TODO: change output scorer to score multiple output types
                    print(f"\nInput #{idx} - Scoring {len(checkpoint_ids)} checkpoints")
                    for checkpoint_id in checkpoint_ids:
                        checkpoint_to_score = checkpoints[checkpoint_id]
                        if use_cached_scores and checkpoint_to_score.scores:
                            print(
                                f"Using cached score for {checkpoint_id}: {checkpoint_to_score.scores}"
                            )

                        assert checkpoint_to_score.output is not None
                        multi_output_scores = output_scorer(
                            task_objective or "",
                            checkpoint_to_score.get_input_variables(),
                            checkpoint_to_score.output,
                            checkpoint_to_score.intermediary_state,
                        )
                        self.recorder.record_output_scores(
                            task_name=task_name,
                            checkpoint_id=checkpoint_id,
                            scores=multi_output_scores,
                        )
                        checkpoint_to_score.scores = multi_output_scores
                        print(f"Scored {checkpoint_id}: {multi_output_scores}")

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
                    elif output_scorer is not None:
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

                    output_comparison_feedback = comparison_feedback[TASK_OUTPUT_KEY]
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
                                feedback=comparison_feedback,
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

            self.calculate_system_health(task_name=task_name)

        finally:
            self.mode = EvaluatorMode.RECORD

    def calculate_system_health(
        self,
        num_samples: int = 10,
        plotting_frequency_in_days: int = 1,
        task_name: Optional[str] = None,
    ) -> None:
        task_name = self._get_recorded_task_name(task_name=task_name)
        output_names_to_score: set[str] = set()

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

        rolling_averages: list[dict[str, Any]] = []

        date_to_use = datetime.datetime.utcnow().date()
        scored_checkpoints.sort(key=lambda x: x.checkpoint_id, reverse=True)

        # Group by reruns if they exist
        rerun_ids_to_score: dict[str, list[Checkpoint]] = defaultdict(list)
        for checkpoint in scored_checkpoints:
            if rerun_id := checkpoint.rerun_metadata.get("id"):
                rerun_ids_to_score[rerun_id].append(checkpoint)

        rerun_rows = []
        for rerun_id, checkpoints in sorted(
            rerun_ids_to_score.items(), key=lambda x: x[0], reverse=True
        ):
            # We want to show how many inputs performed better in this re-run ideally
            # For now let's show score and then change it up
            row: dict[str, Any] = {"rerun_id": rerun_id}

            # num_better, num_worse, num_same
            output_feedback_status: dict[str, list[float]] = defaultdict(
                lambda: [0.0, 0.0, 0.0, 0.0]
            )
            all_output_names: set[str] = set()
            rerun_input_hashes: set[str] = set()
            for checkpoint in checkpoints:
                rerun_input_hashes.add(checkpoint.get_input_hash())
                for output_name, output_score in checkpoint.scores.items():
                    all_output_names.add(output_name)
                    if (
                        feedback_result := rerunid_to_checkpoint_feedback[rerun_id][
                            checkpoint.checkpoint_id
                        ]
                    ) is not None:
                        if feedback_result == FeedbackResult.POSITIVE:
                            output_feedback_status[output_name][0] += 1.0
                        elif feedback_result == FeedbackResult.NEGATIVE:
                            output_feedback_status[output_name][1] += 1.0
                        else:
                            output_feedback_status[output_name][2] += 1.0

                    output_feedback_status[output_name][3] += output_score.score

            for output_name in all_output_names:
                num_better, num_worse, num_same, score = output_feedback_status[
                    output_name
                ]
                total = int(num_better + num_worse + num_same)
                trend_msg = ""
                if num_better > 0:
                    trend_msg += f"{int(num_better)}/{total} got better\n"
                if num_worse > 0:
                    trend_msg += f"{int(num_worse)}/{total} got worse\n"
                if num_same > 0:
                    trend_msg += f"{int(num_same)}/{total} stayed the same\n"

                if trend_msg:
                    row[f"{output_name} trend"] = trend_msg
                else:
                    row[f"{output_name} relative score"] = score

            row["num_checkpoints_used"] = len(checkpoints)
            row["input_diversity"] = len(input_hashes)

            rerun_rows.append(row)
        output_table(rerun_rows, title="Rerun stats")

        for output_name in sorted(output_names_to_score):
            while scored_checkpoints[-1].created_at.date() <= date_to_use:
                total_score = 0.0
                num_checkpoints_used = 0
                input_hash_set = set()
                for checkpoint in scored_checkpoints:
                    if num_checkpoints_used >= num_samples:
                        break

                    if checkpoint.created_at.date() <= date_to_use:
                        if checkpoint.scores is not None:
                            total_score += checkpoint.scores[output_name].score
                            num_checkpoints_used += 1
                            input_hash_set.add(checkpoint.get_input_hash())

                rolling_averages.append(
                    {
                        "date": date_to_use.isoformat(),
                        "rolling_average": total_score / float(num_checkpoints_used),
                        "num_checkpoints_used": num_checkpoints_used,
                        "input_diversity": len(input_hash_set),
                    }
                )
                date_to_use -= datetime.timedelta(days=plotting_frequency_in_days)

            output_table(
                rolling_averages,
                title=f"{output_name} system health (Rolling average of scores for the last {num_samples} checkpoints)",
                column_names=[
                    "date",
                    "rolling_average",
                    "num_checkpoints_used",
                    "input_diversity",
                ],
                markdown_file=os.path.join(
                    self.data_dir, task_name, "system_health.md"
                ),
            )

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
            output_table(input_specific_rows, title=f"{output_name} per input stats")


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
calculate_system_health = _default_evaluator.calculate_system_health
record_intermediary_state = _default_evaluator.record_intermediary_state
