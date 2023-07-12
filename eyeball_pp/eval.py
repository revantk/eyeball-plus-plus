from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
import inspect
import json
from .recorders import (
    Checkpoint,
    ComparisonResult,
    DiskRecorder,
    FileRecorder,
    MemoryRecorder,
    EvalRecorder,
    get_input_hash,
)
from .comparators import model_graded_comparator, comparator_from_scores
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
    dir_path: str = "./eyeball_data"

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
        self.config = EvaluatorConfig._merge(EvaluatorConfig(), **config_kwargs)
        self.mode: EvaluatorMode = EvaluatorMode.RECORD
        self.recorder: EvalRecorder = FileRecorder(self.config.dir_path)
        # self.recorder: EvalRecorder = MemoryRecorder()
        self.current_recorder_state = threading.local()

    def _get_config(self, **override_config_kwargs) -> EvaluatorConfig:
        return EvaluatorConfig._merge(self.config, **override_config_kwargs)

    def set_config(self, **config_kwargs) -> None:
        self.config = EvaluatorConfig._merge(self.config, **config_kwargs)

    @contextmanager
    def start_recording_session(
        self,
        task_name: str,
        checkpoint_id: Optional[str] = None,
        checkpoint_id_to_rerun: Optional[str] = None,
    ) -> Iterator[None]:
        if hasattr(self.current_recorder_state, "recorder_checkpoint_id"):
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

    def _get_last_checkpoint_id(
        self,
        task_name: str,
        input_hash: str,
        current_checkpoint_id: str,
    ) -> Optional[str]:
        checkpoints = self.recorder.get_latest_checkpoints(
            task_name, input_hash, num_checkpoints=2
        )

        for checkpoint_id in reversed(checkpoints):
            if checkpoint_id == current_checkpoint_id:
                continue
            return checkpoint_id
        return None

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
                last_checkpoind_id = self._get_last_checkpoint_id(
                    task_name=task_name,
                    input_hash=input_hash,
                    current_checkpoint_id=recorder_checkpoint_id,
                )
                if last_checkpoind_id is None:
                    continue

                self.checkpoint_id_to_rerun = last_checkpoind_id
                checkpoint_to_rerun = self.recorder.get_checkpoint(
                    task_name=task_name,
                    checkpoint_id=last_checkpoind_id,
                )
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
                            checkpoint_id_to_rerun=last_checkpoind_id,
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
                checkpoint_ids = self.recorder.get_latest_checkpoints(
                    task_name, input_hash, num_checkpoints=4
                )
                raw_checkpoints = [
                    self.recorder.get_checkpoint(task_name, c) for c in checkpoint_ids
                ]
                checkpoints: list[Checkpoint] = [
                    c for c in raw_checkpoints if c is not None
                ]
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
                checkpoint_ids = self.recorder.get_latest_checkpoints(
                    task_name,
                    input_hash,
                    num_checkpoints=num_checkpoints_per_input_hash,
                )

                checkpoints: dict[str, Checkpoint] = {}
                filtered_checkpoint_ids = []
                for checkpoind_id in checkpoint_ids:
                    checkpoint = self.recorder.get_checkpoint(
                        task_name=task_name,
                        checkpoint_id=checkpoind_id,
                    )
                    if (
                        checkpoint is not None
                        and checkpoint.output is not None
                        and checkpoint.output != "null"
                        and checkpoint.output != "None"
                    ):
                        filtered_checkpoint_ids.append(checkpoind_id)
                        checkpoints[checkpoind_id] = checkpoint
                checkpoint_ids = filtered_checkpoint_ids

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

                to_compare = [
                    (checkpoint_ids[i], checkpoint_ids[i + 1])
                    for i in range(len(checkpoint_ids) - 1)
                ]

                print(f"\nInput #{idx} - Running {len(to_compare)} comparison(s)")

                for checkpoint_id_a, checkpoint_id_b in to_compare:
                    checkpoint_a = self.recorder.get_checkpoint(
                        task_name=task_name,
                        checkpoint_id=checkpoint_id_a,
                    )
                    checkpoint_b = self.recorder.get_checkpoint(
                        task_name=task_name,
                        checkpoint_id=checkpoint_id_b,
                    )

                    if checkpoint_a is None or checkpoint_b is None:
                        print(
                            f"Could not find checkpoint {checkpoint_id_a} or {checkpoint_id_b}"
                        )
                        continue

                    output_a = checkpoint_a.output
                    output_b = checkpoint_b.output

                    assert output_a is not None
                    assert output_b is not None

                    should_record_comparison = True
                    if use_cached_comparisons and (
                        comparator_result := self.recorder.get_comparison_result(
                            task_name=task_name,
                            checkpoint_id_a=checkpoint_id_a,
                            checkpoint_id_b=checkpoint_id_b,
                        )
                    ):
                        print(f"Using cached comparison result for {input_hash}")
                        should_record_comparison = False
                        output_comparison_feedback = comparator_result.output_feedback
                    elif output_comparator is not None:
                        output_comparison_feedback = output_comparator(
                            task_objective or "",
                            checkpoint_a.get_input_variables(),
                            output_a,
                            output_b,
                        )
                    elif output_scorer is not None:
                        output_comparison_feedback = comparator_from_scores(
                            scores[checkpoint_id_a], scores[checkpoint_id_b]
                        )
                    else:
                        raise ValueError(
                            "Should not happen. We need an output comparator or output scorer"
                        )

                    a_unique_params = []
                    b_unqiue_params = []
                    for key in (
                        checkpoint_a.eval_params.keys()
                        | checkpoint_b.eval_params.keys()
                    ):
                        param_val_a = checkpoint_a.eval_params.get(key)
                        param_val_b = checkpoint_b.eval_params.get(key)
                        if param_val_a != param_val_b:
                            if param_val_a is not None:
                                a_unique_params.append(f"{key}={param_val_a}")
                            elif param_val_b is not None:
                                b_unqiue_params.append(f"{key}={param_val_b}")

                    a_unique_params_str = (
                        ("(" + ", ".join(a_unique_params) + ")")
                        if len(a_unique_params) > 0
                        else ""
                    )
                    b_unique_params_str = (
                        ("(" + ", ".join(b_unqiue_params) + ")")
                        if len(b_unqiue_params) > 0
                        else ""
                    )

                    a_params_str = ",".join(
                        f"{k}={v}" for k, v in checkpoint_a.eval_params.items()
                    )
                    b_params_str = ",".join(
                        f"{k}={v}" for k, v in checkpoint_b.eval_params.items()
                    )

                    rerun_id_a = (
                        checkpoint_a.rerun_metadata.get("id")
                        if checkpoint_a.rerun_metadata is not None
                        else None
                    )
                    rerun_id_b = (
                        checkpoint_b.rerun_metadata.get("id")
                        if checkpoint_b.rerun_metadata is not None
                        else None
                    )

                    num_comparisons += 1

                    if output_comparison_feedback.result == FeedbackResult.NEUTRAL:
                        print(
                            f"{ORANGE}[neutral] task output is the same between checkpoints {checkpoint_id_a} {a_unique_params_str} & {checkpoint_id_b} {b_unique_params_str} {END_CLR}"
                        )
                        params_to_succesful_examples[a_params_str] += 1
                        params_to_succesful_examples[b_params_str] += 1
                        if rerun_id_a is not None:
                            rerun_to_succesful_examples[rerun_id_a] += 1
                        if rerun_id_b is not None:
                            rerun_to_succesful_examples[rerun_id_b] += 1
                    elif output_comparison_feedback.result == FeedbackResult.NEGATIVE:
                        print(
                            f"{RED}[regression] task output was better in checkpoint {checkpoint_id_a} {a_unique_params_str} than {checkpoint_id_b} {b_unique_params_str} {END_CLR}"
                        )
                        params_to_succesful_examples[a_params_str] += 1
                        if rerun_id_a is not None:
                            rerun_to_succesful_examples[rerun_id_a] += 1
                    else:
                        print(
                            f"{GREEN}[improvement] task output improved from checkpoint {checkpoint_id_a} {a_unique_params_str} to {checkpoint_id_b} {b_unique_params_str}{END_CLR}"
                        )
                        params_to_succesful_examples[b_params_str] += 1
                        if rerun_id_b is not None:
                            rerun_to_succesful_examples[rerun_id_b] += 1
                    if should_record_comparison:
                        self.recorder.record_comparison_result(
                            task_name=task_name,
                            input_hash=input_hash,
                            result=ComparisonResult(
                                checkpoint_id_a=checkpoint_id_a,
                                checkpoint_id_b=checkpoint_id_b,
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
