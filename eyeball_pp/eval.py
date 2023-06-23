from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
import inspect
import json
from .recorders import MemoryRecorder, EvalRecorder
from .comparators import OutputComparator, model_graded_comparator

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
from functools import _make_key, wraps
from dataclasses import dataclass
import dataclasses
import uuid
import datetime

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
        # self.recorder = DiskRecorder(self.config.dir_path)
        self.recorder: EvalRecorder = MemoryRecorder()
        self.current_recorder_state = threading.local()

    def _get_config(self, **override_config_kwargs) -> EvaluatorConfig:
        return EvaluatorConfig._merge(self.config, **override_config_kwargs)

    def set_config(self, **config_kwargs) -> None:
        self.config = EvaluatorConfig._merge(self.config, **config_kwargs)

    @contextmanager
    def start_recording_session(
        self,
        task_name: str,
        example_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        checkpoint_id_to_rerun: Optional[str] = None,
    ) -> Iterator[None]:
        if hasattr(self.current_recorder_state, "example_id"):
            # This should not happen but if it does, we should not overwrite the previous example id
            # Instead we should set it to None so there is no confusion
            # If the user calls get_eval_params this will raise an error
            self.current_recorder_state.example_id = None
            self.current_recorder_state.task_name = None
            self.current_recorder_state.recorder_checkpoint_id = None
            self.current_recorder_state.checkpoint_id_to_rerun = None
            yield None
            return

        self.current_recorder_state.task_name = task_name
        self.current_recorder_state.example_id = example_id or str(uuid.uuid4())
        if checkpoint_id is not None:
            self.current_recorder_state.recorder_checkpoint_id = checkpoint_id
        else:
            self.current_recorder_state.recorder_checkpoint_id = (
                datetime.datetime.utcnow().isoformat()
            )
        self.current_recorder_state.checkpoint_id_to_rerun = checkpoint_id_to_rerun
        yield None

        del self.current_recorder_state.example_id
        del self.current_recorder_state.task_name
        del self.current_recorder_state.recorder_checkpoint_id

    def _should_record(self, **config_kwargs) -> bool:
        if self.mode in (
            EvaluatorMode.COMPARE_CHECKPOINTS,
            EvaluatorMode.RERUN_EXAMPLES,
        ):
            return True

        config = self._get_config(**config_kwargs)
        if config.sample_rate == 0:
            return False
        return config.sample_rate == 1.0 or random.random() < config.sample_rate

    def record_input(
        self,
        variable_name: str,
        value: T,
        example_id: Optional[str] = None,
        task_name: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        **config_kwargs,
    ) -> T:
        if self._should_record(**config_kwargs):
            return self._record_variable(
                task_name=task_name,
                example_id=example_id,
                recorder_checkpoint_id=checkpoint_id,
                variable_name=variable_name,
                value=value,
                is_output=False,
            )
        else:
            return value

    def record_output(
        self,
        variable_name: str,
        value: Any,
        task_name: Optional[str] = None,
        example_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        **config_kwargs,
    ) -> None:
        if self._should_record(**config_kwargs):
            self._record_variable(
                task_name=task_name,
                example_id=example_id,
                recorder_checkpoint_id=checkpoint_id,
                variable_name=variable_name,
                value=value,
                is_output=True,
            )

    def _get_last_checkpoint_id(
        self,
        task_name: str,
        example_id: str,
        current_checkpoint_id: str,
    ) -> Optional[str]:
        checkpoints = self.recorder.get_latest_checkpoints(
            task_name, example_id, num_checkpoints=2
        )

        for checkpoint_id in reversed(checkpoints):
            if checkpoint_id == current_checkpoint_id:
                continue
            return checkpoint_id
        return None

    def _fetch_variable_value(
        self,
        task_name: str,
        variable_name: str,
        passed_in_value: T,
        example_id: str,
        checkpoint_id_to_fetch: str,
    ) -> T:
        example = self.recorder.get_example(
            task_name, example_id, checkpoint_id=checkpoint_id_to_fetch
        )
        if example is None:
            return passed_in_value

        if variable_name not in example.variables:
            return passed_in_value

        value_str = example.variables[variable_name]

        if isinstance(passed_in_value, JsonSerializable):
            return passed_in_value.from_json(value_str)  # type: ignore
        else:
            return json.loads(value_str)

    def new_function_which_does_nothing(self, input: str) -> str:
        return f"Hello {input}"

    def _get_recorder_state(
        self,
        task_name: Optional[str] = None,
        example_id: Optional[str] = None,
        recorder_checkpoint_id: Optional[str] = None,
    ) -> tuple[str, str, str]:
        if task_name is None and hasattr(self.current_recorder_state, "task_name"):
            task_name = self.current_recorder_state.task_name
        if task_name is None:
            raise ValueError(
                "Must provide a task name or start a recording session to record a variable"
            )

        if example_id is None and hasattr(self.current_recorder_state, "example_id"):
            example_id = self.current_recorder_state.example_id
        if example_id is None:
            raise ValueError(
                "Must provide an example id or start a recording session to record a variable"
            )

        if recorder_checkpoint_id is None and hasattr(
            self.current_recorder_state, "recorder_checkpoint_id"
        ):
            recorder_checkpoint_id = self.current_recorder_state.recorder_checkpoint_id
        if recorder_checkpoint_id is None:
            raise ValueError(
                "Must provide a checkpoint id or start a recording session to record a variable"
            )

        return task_name, example_id, recorder_checkpoint_id

    def get_eval_params(
        self,
        task_name: Optional[str] = None,
        example_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        task_name, example_id, checkpoint_id = self._get_recorder_state(
            task_name, example_id, checkpoint_id
        )

        example = self.recorder.get_example(
            task_name=task_name, example_id=example_id, checkpoint_id=checkpoint_id
        )
        if example is None:
            return None
        return example.params

    def get_eval_param(
        self,
        param_name: str,
        task_name: Optional[str] = None,
        example_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[Any]:
        params = self.get_eval_params(task_name, example_id, checkpoint_id)
        if params is None:
            return None
        return params.get(param_name)

    def _record_variable(
        self,
        variable_name: str,
        value: T,
        is_output: bool,
        task_name: Optional[str] = None,
        example_id: Optional[str] = None,
        recorder_checkpoint_id: Optional[str] = None,
        checkpoint_id_to_rerun: Optional[str] = None,
    ) -> T:
        task_name, example_id, recorder_checkpoint_id = self._get_recorder_state(
            task_name, example_id, recorder_checkpoint_id
        )

        if self.mode == EvaluatorMode.COMPARE_CHECKPOINTS:
            return value

        # if (
        #     self.mode == EvaluatorMode.RERUN_EXAMPLES
        #     and not is_output
        #     and checkpoint_id_to_rerun is not None
        # ):
        #     if checkpoint_id_to_rerun is None:
        #         checkpoint_id_to_rerun = (
        #             self.checkpoint_id_to_rerun
        #             or self._get_last_checkpoint_id(
        #                 task_name=task_name,
        #                 example_id=example_id,
        #                 current_checkpoint_id=recorder_checkpoint_id,
        #             )
        #         )

        #     value = self._fetch_variable_value(
        #         task_name=task_name,
        #         passed_in_value=value,
        #         example_id=example_id,
        #         checkpoint_id_to_fetch=checkpoint_id_to_rerun,
        #         variable_name=variable_name,
        #     )

        if isinstance(value, JsonSerializable):
            value_str = value.to_json()
        else:
            value_str = json.dumps(value)

        self.recorder.record(
            task_name=task_name,
            example_id=example_id,
            checkpoint_id=recorder_checkpoint_id,
            variable_name=variable_name,
            value=value_str,
            is_output=is_output,
        )

        return value

    def record_task(
        self,
        task_name: Optional[str] = None,
        args_to_skip: list[str] = ["self"],
        example_id_arg_name: Optional[str] = None,
        reset_checkpoint_every_call: bool = True,
        eval_params: Optional[dict[str, Any]] = None,
        **config_kwargs,
    ) -> Callable[..., object]:
        def _decorator(fn: Callable[..., T]) -> Callable[..., T]:
            @wraps(fn)
            def _wrapper(*args, **kwargs):
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

                # Set example id
                if self.mode == EvaluatorMode.RERUN_EXAMPLES:
                    example_id = self.current_recorder_state.example_id
                elif example_id_arg_name in kwargs:
                    example_id = str(kwargs[example_id_arg_name])
                elif example_id_arg_name in fn_arg_names:
                    example_id = str(args[fn_arg_names.index(example_id_arg_name)])
                else:
                    example_id = str(_make_key(args, kwargs, False).__hash__())

                # Set checkpoint id
                if self.mode == EvaluatorMode.RERUN_EXAMPLES:
                    recorder_checkpoint_id = (
                        self.current_recorder_state.recorder_checkpoint_id
                    )
                elif reset_checkpoint_every_call:
                    recorder_checkpoint_id = datetime.datetime.utcnow().isoformat()
                else:
                    recorder_checkpoint_id = self.recorder_checkpoint_id

                # Set params
                if self.mode == EvaluatorMode.RERUN_EXAMPLES:
                    params = self.get_eval_params(
                        task_name=local_task_name,
                        example_id=example_id,
                        checkpoint_id=recorder_checkpoint_id,
                    )
                    # TODO: maybe we should not override with default params in this mode. Can lead to some confusion
                    if (params is None or len(params) == 0) and eval_params is not None:
                        self.recorder.set_eval_params(
                            example_id, recorder_checkpoint_id, eval_params
                        )
                elif eval_params is not None:
                    self.recorder.set_eval_params(
                        example_id, recorder_checkpoint_id, eval_params
                    )

                for arg_name, arg_val in zip(fn_arg_names, args):
                    if arg_name not in args_to_skip:
                        arg_val = self._record_variable(
                            task_name=local_task_name,
                            example_id=example_id,
                            recorder_checkpoint_id=recorder_checkpoint_id,
                            variable_name=arg_name,
                            value=arg_val,
                            is_output=False,
                        )

                for kwarg_name, kwarg_val in kwargs.items():
                    if kwarg_name not in args_to_skip:
                        kwarg_val = self._record_variable(
                            task_name=local_task_name,
                            example_id=example_id,
                            recorder_checkpoint_id=recorder_checkpoint_id,
                            variable_name=kwarg_name,
                            value=kwarg_val,
                            is_output=False,
                        )

                if self.mode == EvaluatorMode.RERUN_EXAMPLES:
                    result = fn(*args, **kwargs)
                else:
                    with self.start_recording_session(
                        task_name=local_task_name,
                        example_id=example_id,
                        checkpoint_id=recorder_checkpoint_id,
                    ):
                        result = fn(*args, **kwargs)

                self._record_variable(
                    task_name=local_task_name,
                    example_id=example_id,
                    recorder_checkpoint_id=recorder_checkpoint_id,
                    variable_name="fn_return_val",
                    value=result,
                    is_output=True,
                )
                return result

            return _wrapper

        return _decorator

    def rerun_recorded_examples(
        self,
        *eval_params_list: dict[str, Any],
        task_name: Optional[str] = None,
        example_ids: Optional[list[str]] = None,
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

        if example_ids is None or len(example_ids) == 0:
            example_ids = self.recorder.get_example_ids(task_name=task_name)

        try:
            self.mode = EvaluatorMode.RERUN_EXAMPLES
            recorder_checkpoint_id = datetime.datetime.utcnow().isoformat()

            for example_id in example_ids:
                last_checkpoind_id = self._get_last_checkpoint_id(
                    task_name, example_id, recorder_checkpoint_id
                )
                if last_checkpoind_id is None:
                    continue

                self.checkpoint_id_to_rerun = last_checkpoind_id
                example_to_rerun = self.recorder.get_example(
                    task_name=task_name,
                    example_id=example_id,
                    checkpoint_id=last_checkpoind_id,
                )
                if example_to_rerun is None:
                    continue

                print(f"\n\nRerunning {example_to_rerun}")

                if len(eval_params_list) > 0:
                    for eval_params in eval_params_list:
                        with self.start_recording_session(
                            task_name=task_name,
                            example_id=example_id,
                            checkpoint_id=recorder_checkpoint_id,
                            checkpoint_id_to_rerun=last_checkpoind_id,
                        ):
                            self.recorder.set_eval_params(
                                task_name,
                                example_id,
                                recorder_checkpoint_id,
                                eval_params,
                            )
                            print(f"Using eval params: {eval_params}")
                            yield example_to_rerun.get_input_variables()
                else:
                    print("Using default eval params")
                    with self.start_recording_session(
                        task_name=task_name,
                        example_id=example_id,
                        checkpoint_id=recorder_checkpoint_id,
                    ):
                        yield example_to_rerun.get_input_variables()
        finally:
            self.mode = EvaluatorMode.RECORD

    def rate_recorded_examples(
        self,
        *example_ids: str,
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

        example_ids_lst: list[str] = list(example_ids)

        if len(example_ids_lst) == 0:
            example_ids_lst = self.recorder.get_example_ids(task_name=task_name)

        try:
            self.mode = EvaluatorMode.RATE_EXAMPLES
            for example_id in example_ids_lst:
                # TODO: implement this
                pass
        finally:
            self.mode = EvaluatorMode.RECORD

    def compare_recorded_checkpoints(
        self,
        task_name: Optional[str] = None,
        num_examples_to_compare: int = 5,
        num_checkpoints_to_eval: int = 2,
        task_objective: Optional[str] = None,
        output_comparator: Optional[OutputComparator] = None,
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

        if output_comparator is None:
            output_comparator = model_graded_comparator
            assert (
                task_objective is not None
            ), "Must provide an objective to compare or an output comparator"

        example_ids_lst = self.recorder.get_example_ids(task_name)
        random.shuffle(example_ids_lst)
        example_ids_lst = example_ids_lst[:num_examples_to_compare]

        try:
            self.mode = EvaluatorMode.COMPARE_CHECKPOINTS

            params_to_succesful_examples: dict[str, int] = defaultdict(int)
            checkpoint_to_succesful_examples: dict[str, int] = defaultdict(int)

            for example_id in example_ids_lst:
                example = self.recorder.get_example(
                    task_name=task_name, example_id=example_id
                )
                if example is None:
                    continue

                print(
                    f"\n\nComparing {num_checkpoints_to_eval} checkpoints of {example}"
                )
                checkpoint_ids = self.recorder.get_latest_checkpoints(
                    task_name, example_id, num_checkpoints=num_checkpoints_to_eval
                )

                to_compare = [
                    (checkpoint_ids[i], checkpoint_ids[i + 1])
                    for i in range(len(checkpoint_ids) - 1)
                ]

                for checkpoint_a, checkpoint_b in to_compare:
                    print()
                    example_a = self.recorder.get_example(
                        task_name=task_name,
                        example_id=example_id,
                        checkpoint_id=checkpoint_a,
                    )
                    example_b = self.recorder.get_example(
                        task_name=task_name,
                        example_id=example_id,
                        checkpoint_id=checkpoint_b,
                    )
                    if example_a is None or example_b is None:
                        print(
                            f"Could not find example {example_id} for checkpoint {checkpoint_a} or {checkpoint_b}"
                        )
                        continue
                    for var_name in example_a.output_variable_names:
                        if var_name not in example_b.output_variable_names:
                            print(
                                f"Variable {var_name} not found in checkpoint {checkpoint_b}"
                            )
                            continue
                        val_a = example_a.variables[var_name]
                        val_b = example_b.variables[var_name]

                        comparator_result = output_comparator(
                            task_objective or "",
                            example_a.get_input_variables(),
                            val_a,
                            val_b,
                        )

                        a_unique_params = []
                        b_unqiue_params = []
                        for key in example_a.params.keys() | example_b.params.keys():
                            param_val_a = example_a.params.get(key)
                            param_val_b = example_b.params.get(key)
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
                            f"{k}={v}" for k, v in example_a.params.items()
                        )
                        b_params_str = ",".join(
                            f"{k}={v}" for k, v in example_b.params.items()
                        )

                        if comparator_result == 0:
                            print(
                                f"{ORANGE}[neutral] `{var_name}` is the same between checkpoints {checkpoint_a} {a_unique_params_str} & {checkpoint_b} {b_unique_params_str} {END_CLR}"
                            )
                            params_to_succesful_examples[a_params_str] += 1
                            params_to_succesful_examples[b_params_str] += 1
                            checkpoint_to_succesful_examples[checkpoint_a] += 1
                            checkpoint_to_succesful_examples[checkpoint_b] += 1
                        elif comparator_result < 0:
                            print(
                                f"{RED}[regression] `{var_name}` was better in checkpoint {checkpoint_a} {a_unique_params_str} than {checkpoint_b} {b_unique_params_str} {END_CLR}"
                            )
                            params_to_succesful_examples[a_params_str] += 1
                            checkpoint_to_succesful_examples[checkpoint_a] += 1
                        else:
                            print(
                                f"{GREEN}[improvement] `{var_name}` improved from checkpoint {checkpoint_a} {a_unique_params_str} to {checkpoint_b} {b_unique_params_str}{END_CLR}"
                            )
                            params_to_succesful_examples[b_params_str] += 1
                            checkpoint_to_succesful_examples[checkpoint_b] += 1
            print("\nSummary:")
            print("---------")
            print("Your most sucessful checkpoints:")
            for checkpoint, num_successes in sorted(
                checkpoint_to_succesful_examples.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                print(f"{checkpoint}: {num_successes}/{len(example_ids_lst)} successes")
            print("\nYour most sucessful params:")
            for params, num_successes in sorted(
                params_to_succesful_examples.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"{params}: {num_successes}/{len(example_ids_lst)} successes")

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
