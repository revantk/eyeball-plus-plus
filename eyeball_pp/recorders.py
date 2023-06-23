import os
import pickle
from enum import Enum
from typing import Any, Optional, Protocol
from dataclasses import dataclass
import json
import dataclasses


class ResponseFeedback(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class Example:
    id: str
    checkpoint_id: str
    variables: dict[str, str]
    output_variable_names: set[str]
    params: dict[str, Any]
    feedback: Optional[ResponseFeedback] = None
    feedback_details: Optional[str] = None

    def get_input_variables(self) -> dict[str, str]:
        return {
            var_name: json.loads(var_val)
            for var_name, var_val in self.variables.items()
            if var_name not in self.output_variable_names
        }

    def __str__(self) -> str:
        msg = f"Example: {self.id} @ {self.checkpoint_id}\n"
        msg += "Input Variables:\n"
        for var_name, var_val in self.get_input_variables().items():
            msg += f"{var_name}={var_val}\n"
        return msg


class EvalRecorder(Protocol):
    def record(
        self,
        task_name: str,
        example_id: str,
        checkpoint_id: str,
        variable_name: str,
        value: str,
        is_output: bool,
    ) -> None:
        ...

    def set_eval_params(
        self,
        task_name: str,
        example_id: str,
        checkpoint_id: str,
        params: dict[str, Any],
    ) -> None:
        ...

    def set_response_feedback(
        self,
        task_name: str,
        example_id: str,
        checkpoint_id: str,
        feedback: ResponseFeedback,
        details: Optional[str] = None,
    ) -> None:
        ...

    def get_example(
        self, task_name: str, example_id: str, checkpoint_id: Optional[str] = None
    ) -> Optional[Example]:
        ...

    def get_latest_checkpoints(
        self, task_name: str, example_id: str, num_checkpoints: int = 2
    ) -> list[str]:
        ...

    def get_example_ids(self, task_name: str) -> list[str]:
        ...

    def get_task_names(self) -> list[str]:
        ...


@dataclass
class Task:
    name: str
    records: dict[str, Example] = dataclasses.field(default_factory=dict)
    checkpoints: dict[str, set[str]] = dataclasses.field(default_factory=dict)


class MemoryRecorder(EvalRecorder):
    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}

    def _example_key(self, example_id: str, checkpoint_id: str) -> str:
        return f"{example_id}:{checkpoint_id}"

    def _fetch_or_create_example(
        self, task_name: str, example_id: str, checkpoint_id: str
    ) -> Example:
        if task_name not in self.tasks:
            task = Task(name=task_name)
            self.tasks[task_name] = task
        else:
            task = self.tasks[task_name]

        key = self._example_key(example_id, checkpoint_id)
        if example_id not in task.checkpoints:
            task.checkpoints[example_id] = set()

        task.checkpoints[example_id].add(checkpoint_id)

        if key not in task.records:
            example = Example(
                id=example_id,
                checkpoint_id=checkpoint_id,
                variables={},
                output_variable_names=set(),
                params={},
            )
            task.records[key] = example
            return example
        else:
            return task.records[key]

    def record(
        self,
        task_name: str,
        example_id: str,
        checkpoint_id: str,
        variable_name: str,
        value: str,
        is_output: bool,
    ) -> None:
        example = self._fetch_or_create_example(task_name, example_id, checkpoint_id)
        example.variables[variable_name] = value
        if is_output:
            example.output_variable_names.add(variable_name)

    def get_latest_checkpoints(
        self, task_name: str, example_id: str, num_checkpoints: int = 2
    ) -> list[str]:
        if task_name not in self.tasks:
            return []
        task = self.tasks[task_name]
        if example_id not in task.checkpoints:
            return []

        return sorted(task.checkpoints[example_id])[-num_checkpoints:]

    def get_example(
        self, task_name: str, example_id: str, checkpoint_id: Optional[str] = None
    ) -> Optional[Example]:
        if task_name not in self.tasks:
            return None

        task = self.tasks[task_name]
        if checkpoint_id is None:
            checkpoints = self.get_latest_checkpoints(
                task_name, example_id, num_checkpoints=1
            )
            if len(checkpoints) == 0:
                return None
            checkpoint_id = checkpoints[0]

        key = self._example_key(example_id, checkpoint_id)

        if key not in task.records:
            return None

        return task.records[key]

    def get_example_ids(self, task_name: str) -> list[str]:
        task = self.tasks.get(task_name)
        if task is None:
            return []

        return list(task.checkpoints.keys())

    def set_eval_params(
        self,
        task_name: str,
        example_id: str,
        checkpoint_id: str,
        params: dict[str, Any],
    ) -> None:
        example = self._fetch_or_create_example(task_name, example_id, checkpoint_id)
        example.params.update(params)

    def set_response_feedback(
        self,
        task_name: str,
        example_id: str,
        checkpoint_id: str,
        feedback: ResponseFeedback,
        details: str | None = None,
    ) -> None:
        example = self._fetch_or_create_example(task_name, example_id, checkpoint_id)
        example.feedback = feedback
        if details is not None:
            example.feedback_details = details

    def get_task_names(self) -> list[str]:
        return list(self.tasks.keys())


class DiskRecorder(EvalRecorder):
    # TODO: improve this by using rocksdb etc. vs a stupid pickle file
    def __init__(self, dir_path: str) -> None:
        self.file_name = os.path.join(dir_path, "evaluator.pkl")
        if os.path.exists(self.file_name):
            self.memory_recorder = pickle.load(open(self.file_name, "rb"))
        else:
            self.memory_recorder = MemoryRecorder()

    def record(
        self,
        task_name: str,
        example_id: str,
        checkpoint_id: str,
        name: str,
        value: str,
        is_output: bool,
    ) -> None:
        self.memory_recorder.record(example_id, checkpoint_id, name, value, is_output)
        pickle.dump(self.memory_recorder, open(self.file_name, "wb"))

    def set_eval_params(
        self,
        task_name: str,
        example_id: str,
        checkpoint_id: str,
        params: dict[str, Any],
    ) -> None:
        self.memory_recorder.set_eval_params(example_id, checkpoint_id, params)
        pickle.dump(self.memory_recorder, open(self.file_name, "wb"))

    def set_response_feedback(
        self,
        task_name: str,
        example_id: str,
        checkpoint_id: str,
        feedback: ResponseFeedback,
        details: str | None = None,
    ) -> None:
        self.memory_recorder.set_response_feedback(
            task_name, example_id, checkpoint_id, feedback, details
        )
        pickle.dump(self.memory_recorder, open(self.file_name, "wb"))

    def get_example(
        self, task_name: str, example_id: str, checkpoint_id: Optional[str] = None
    ) -> Optional[Example]:
        return self.memory_recorder.get_example(task_name, example_id, checkpoint_id)

    def get_latest_checkpoints(
        self, task_name: str, example_id: str, num_checkpoints: int = 2
    ) -> list[str]:
        return self.memory_recorder.get_latest_checkpoints(
            task_name, example_id, num_checkpoints
        )

    def get_example_ids(self, task_name: str) -> list[str]:
        return self.memory_recorder.get_example_ids(task_name)

    def get_task_names(self) -> list[str]:
        return self.memory_recorder.get_task_names()
