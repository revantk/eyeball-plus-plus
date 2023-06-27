import os
import pickle
from enum import Enum
from typing import Any, Optional, Protocol
from dataclasses import dataclass
import json
import dataclasses
from functools import _make_key


class ResponseFeedback(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class Checkpoint:
    checkpoint_id: str
    variables: dict[str, str]
    output_variable_names: set[str]
    eval_params: dict[str, Any]
    feedback: Optional[ResponseFeedback] = None
    feedback_details: Optional[str] = None
    rerun_metadata: Optional[dict[str, Any]] = None

    def get_input_variables(self) -> dict[str, str]:
        return {
            var_name: json.loads(var_val)
            for var_name, var_val in self.variables.items()
            if var_name not in self.output_variable_names
        }

    def get_input_hash(self) -> str:
        sorted_input_vars = sorted(
            [
                (var_name, var_val)
                for var_name, var_val in self.variables.items()
                if var_name not in self.output_variable_names
            ],
            key=lambda x: x[0],
        )
        return str(hash(tuple(sorted_input_vars)))

    def __str__(self) -> str:
        msg = f"Example: {self.get_input_hash()} @ {self.checkpoint_id}\n"
        msg += "Input Variables:\n"
        for var_name, var_val in self.get_input_variables().items():
            msg += f"{var_name}={var_val}\n"
        return msg


class EvalRecorder(Protocol):
    def record(
        self,
        task_name: str,
        checkpoint_id: str,
        variable_name: str,
        value: str,
        is_output: bool,
    ) -> None:
        ...

    def set_eval_params(
        self,
        task_name: str,
        checkpoint_id: str,
        eval_params: dict[str, Any],
        rerun_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        ...

    def set_response_feedback(
        self,
        task_name: str,
        checkpoint_id: str,
        feedback: ResponseFeedback,
        details: Optional[str] = None,
    ) -> None:
        ...

    def get_checkpoint(
        self, task_name: str, checkpoint_id: str
    ) -> Optional[Checkpoint]:
        ...

    def get_latest_checkpoints(
        self, task_name: str, input_hash: str, num_checkpoints: int = 2
    ) -> list[str]:
        ...

    def get_input_hashes(self, task_name: str) -> list[str]:
        ...

    def get_task_names(self) -> list[str]:
        ...


@dataclass
class Task:
    name: str
    checkpoints: dict[str, Checkpoint] = dataclasses.field(default_factory=dict)
    input_hashes: dict[str, set[str]] = dataclasses.field(default_factory=dict)


class MemoryRecorder(EvalRecorder):
    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}

    def _example_key(self, example_id: str, checkpoint_id: str) -> str:
        return f"{example_id}:{checkpoint_id}"

    def _fetch_or_create_checkpoint(
        self, task_name: str, checkpoint_id: str
    ) -> Checkpoint:
        if task_name not in self.tasks:
            task = Task(name=task_name)
            self.tasks[task_name] = task
        else:
            task = self.tasks[task_name]

        if checkpoint_id not in task.checkpoints:
            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                variables={},
                output_variable_names=set(),
                eval_params={},
            )
            task.checkpoints[checkpoint_id] = checkpoint
            return checkpoint
        else:
            return task.checkpoints[checkpoint_id]

    def _save_input_hash(self, task_name: str, checkpoint: Checkpoint) -> None:
        # Remove the checkpoint from any previous input hashes
        task = self.tasks[task_name]
        checkpoint_id = checkpoint.checkpoint_id

        for input_hash in task.input_hashes:
            if checkpoint_id in task.input_hashes[input_hash]:
                task.input_hashes[input_hash].remove(checkpoint_id)

        # Add the checkpoint to the new input hash
        input_hash = checkpoint.get_input_hash()
        if input_hash not in task.input_hashes:
            task.input_hashes[input_hash] = set()
        task.input_hashes[input_hash].add(checkpoint_id)

    def record(
        self,
        task_name: str,
        checkpoint_id: str,
        variable_name: str,
        value: str,
        is_output: bool,
    ) -> None:
        checkpoint = self._fetch_or_create_checkpoint(
            task_name=task_name, checkpoint_id=checkpoint_id
        )
        checkpoint.variables[variable_name] = value
        if is_output:
            checkpoint.output_variable_names.add(variable_name)
        else:
            self._save_input_hash(task_name, checkpoint)

    def get_latest_checkpoints(
        self, task_name: str, input_hash: str, num_checkpoints: int = 2
    ) -> list[str]:
        if task_name not in self.tasks:
            return []
        task = self.tasks[task_name]
        if input_hash not in task.input_hashes:
            return []

        return sorted(task.input_hashes[input_hash])[-num_checkpoints:]

    def get_checkpoint(
        self, task_name: str, checkpoint_id: str
    ) -> Optional[Checkpoint]:
        if task_name not in self.tasks:
            return None
        task = self.tasks[task_name]

        if checkpoint_id not in task.checkpoints:
            return None

        return task.checkpoints[checkpoint_id]

    def get_input_hashes(self, task_name: str) -> list[str]:
        task = self.tasks.get(task_name)
        if task is None:
            return []

        hashes_to_remove = []
        for input_hash in task.input_hashes:
            if len(task.input_hashes[input_hash]) == 0:
                hashes_to_remove.append(input_hash)

        for input_hash in hashes_to_remove:
            del task.input_hashes[input_hash]

        return list(task.input_hashes.keys())

    def set_eval_params(
        self,
        task_name: str,
        checkpoint_id: str,
        eval_params: dict[str, Any],
        rerun_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        checkpoint = self._fetch_or_create_checkpoint(
            task_name=task_name, checkpoint_id=checkpoint_id
        )
        checkpoint.eval_params.update(eval_params)
        if rerun_metadata is not None:
            checkpoint.rerun_metadata = rerun_metadata

    def set_response_feedback(
        self,
        task_name: str,
        checkpoint_id: str,
        feedback: ResponseFeedback,
        details: str | None = None,
    ) -> None:
        example = self._fetch_or_create_checkpoint(
            task_name=task_name, checkpoint_id=checkpoint_id
        )
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
        checkpoint_id: str,
        variable_name: str,
        value: str,
        is_output: bool,
    ) -> None:
        self.memory_recorder.record(
            task_name=task_name,
            checkpoint_id=checkpoint_id,
            variable_name=variable_name,
            value=value,
            is_output=is_output,
        )
        pickle.dump(self.memory_recorder, open(self.file_name, "wb"))

    def set_eval_params(
        self,
        task_name: str,
        checkpoint_id: str,
        params: dict[str, Any],
    ) -> None:
        self.memory_recorder.set_eval_params(
            task_name=task_name, checkpoint_id=checkpoint_id, params=params
        )
        pickle.dump(self.memory_recorder, open(self.file_name, "wb"))

    def set_response_feedback(
        self,
        task_name: str,
        checkpoint_id: str,
        feedback: ResponseFeedback,
        details: str | None = None,
    ) -> None:
        self.memory_recorder.set_response_feedback(
            task_name=task_name,
            checkpoint_id=checkpoint_id,
            feedback=feedback,
            details=details,
        )
        pickle.dump(self.memory_recorder, open(self.file_name, "wb"))

    def get_checkpoint(
        self, task_name: str, checkpoint_id: str
    ) -> Optional[Checkpoint]:
        return self.memory_recorder.get_checkpoint(
            task_name=task_name, checkpoint_id=checkpoint_id
        )

    def get_latest_checkpoints(
        self, task_name: str, input_hash: str, num_checkpoints: int = 2
    ) -> list[str]:
        return self.memory_recorder.get_latest_checkpoints(
            task_name=task_name, input_hash=input_hash, num_checkpoints=num_checkpoints
        )

    def get_input_hashes(self, task_name: str) -> list[str]:
        return self.memory_recorder.get_input_hashes(task_name)

    def get_task_names(self) -> list[str]:
        return self.memory_recorder.get_task_names()
