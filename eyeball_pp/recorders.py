import hashlib
import json
import os
import pickle
from typing import Any, Optional, Protocol
from dataclasses import dataclass
import dataclasses

from .classes import OutputFeedback


@dataclass
class ComparisonResult:
    checkpoint_id_a: str
    checkpoint_id_b: str
    output_feedback: OutputFeedback


@dataclass
class Checkpoint:
    checkpoint_id: str
    input_variables: dict[str, str]
    output: str
    eval_params: dict[str, Any]
    output_feedback: Optional[OutputFeedback] = None
    output_score: Optional[float] = None
    rerun_metadata: dict[str, str] = dataclasses.field(default_factory=dict)

    def get_input_variables(self) -> dict[str, str]:
        return dict(self.input_variables)

    def get_input_hash(self) -> str:
        sorted_input_vars = sorted(
            [
                (str(var_name), str(var_val))
                for var_name, var_val in self.input_variables.items()
            ],
            key=lambda x: x[0],
        )
        input_var_str = json.dumps(sorted_input_vars)
        return hashlib.sha256(input_var_str.encode("utf-8")).hexdigest()

    def __str__(self) -> str:
        msg = f"Example: {self.get_input_hash()} @ {self.checkpoint_id}\n"
        msg += "Input Variables:\n"
        for var_name, var_val in self.input_variables.items():
            msg += f"{var_name}={var_val}\n"
        return msg


class EvalRecorder(Protocol):
    def record_input_variable(
        self,
        task_name: str,
        checkpoint_id: str,
        variable_name: str,
        value: str,
    ) -> None:
        ...

    def record_output(
        self,
        task_name: str,
        checkpoint_id: str,
        output: str,
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

    def record_output_feedback(
        self,
        task_name: str,
        checkpoint_id: str,
        feedback: OutputFeedback,
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

    def record_comparison_result(
        self,
        task_name: str,
        input_hash: str,
        result: ComparisonResult,
    ) -> None:
        ...

    def record_output_score(
        self,
        task_name: str,
        checkpoint_id: str,
        score: float,
    ) -> None:
        ...

    def get_comparison_result(
        self,
        task_name: str,
        checkpoint_id_a: str,
        checkpoint_id_b: str,
    ) -> Optional[ComparisonResult]:
        ...

    def get_comparison_results_for_input_hash(
        self,
        task_name: str,
        input_hash: str,
    ) -> list[ComparisonResult]:
        ...


@dataclass
class Task:
    name: str
    checkpoints: dict[str, Checkpoint] = dataclasses.field(default_factory=dict)
    input_hashes: dict[str, set[str]] = dataclasses.field(default_factory=dict)
    comparison_results: dict[str, ComparisonResult] = dataclasses.field(
        default_factory=dict
    )
    input_hash_to_comparison_results: dict[str, set[str]] = dataclasses.field(
        default_factory=dict
    )


class MemoryRecorder(EvalRecorder):
    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}

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
                input_variables={},
                output="",
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

        input_hashes_to_remove = []
        for input_hash in task.input_hashes:
            if checkpoint_id in task.input_hashes[input_hash]:
                task.input_hashes[input_hash].remove(checkpoint_id)
            if len(task.input_hashes[input_hash]) == 0:
                input_hashes_to_remove.append(input_hash)

        for input_hash in input_hashes_to_remove:
            del task.input_hashes[input_hash]

        # Add the checkpoint to the new input hash
        input_hash = checkpoint.get_input_hash()
        if input_hash not in task.input_hashes:
            task.input_hashes[input_hash] = set()
        task.input_hashes[input_hash].add(checkpoint_id)

    def record_input_variable(
        self,
        task_name: str,
        checkpoint_id: str,
        variable_name: str,
        value: str,
    ) -> None:
        checkpoint = self._fetch_or_create_checkpoint(
            task_name=task_name, checkpoint_id=checkpoint_id
        )
        checkpoint.input_variables[variable_name] = value
        self._save_input_hash(task_name, checkpoint)

    def record_output(self, task_name: str, checkpoint_id: str, output: str) -> None:
        checkpoint = self._fetch_or_create_checkpoint(
            task_name=task_name, checkpoint_id=checkpoint_id
        )
        checkpoint.output = output

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

    def record_output_feedback(
        self,
        task_name: str,
        checkpoint_id: str,
        feedback: OutputFeedback,
    ) -> None:
        checkpoint = self._fetch_or_create_checkpoint(
            task_name=task_name, checkpoint_id=checkpoint_id
        )
        checkpoint.output_feedback = feedback

    def get_task_names(self) -> list[str]:
        return list(self.tasks.keys())

    def record_comparison_result(
        self,
        task_name: str,
        input_hash: str,
        result: ComparisonResult,
    ) -> None:
        task = self.tasks[task_name]
        key = f"{result.checkpoint_id_a},{result.checkpoint_id_b}"
        task.comparison_results[key] = result

        if input_hash not in task.input_hash_to_comparison_results:
            task.input_hash_to_comparison_results[input_hash] = set()
        task.input_hash_to_comparison_results[input_hash].add(key)

    def record_output_score(
        self,
        task_name: str,
        checkpoint_id: str,
        score: float,
    ) -> None:
        checkpoint = self._fetch_or_create_checkpoint(
            task_name=task_name, checkpoint_id=checkpoint_id
        )
        checkpoint.output_score = score

    def get_comparison_result(
        self,
        task_name: str,
        checkpoint_id_a: str,
        checkpoint_id_b: str,
    ) -> Optional[ComparisonResult]:
        task = self.tasks[task_name]
        key = f"{checkpoint_id_a},{checkpoint_id_b}"
        if key not in task.comparison_results:
            return None
        return task.comparison_results[key]

    def get_comparison_results_for_input_hash(
        self,
        task_name: str,
        input_hash: str,
    ) -> list[ComparisonResult]:
        task = self.tasks[task_name]
        if input_hash not in task.input_hash_to_comparison_results:
            return []
        comparison_results = task.input_hash_to_comparison_results[input_hash]
        return [task.comparison_results[key] for key in comparison_results]


class DiskRecorder(EvalRecorder):
    # TODO: improve this by using rocksdb etc. vs a stupid pickle file
    def __init__(self, dir_path: str) -> None:
        self.file_name = os.path.join(dir_path, "eyeball_checkpoints.pkl")
        if os.path.exists(self.file_name):
            self.memory_recorder = pickle.load(open(self.file_name, "rb"))
        else:
            self.memory_recorder = MemoryRecorder()

    def record_input_variable(
        self,
        task_name: str,
        checkpoint_id: str,
        variable_name: str,
        value: str,
    ) -> None:
        self.memory_recorder.record_input_variable(
            task_name=task_name,
            checkpoint_id=checkpoint_id,
            variable_name=variable_name,
            value=value,
        )
        pickle.dump(self.memory_recorder, open(self.file_name, "wb"))

    def set_eval_params(
        self,
        task_name: str,
        checkpoint_id: str,
        eval_params: dict[str, Any],
        rerun_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.memory_recorder.set_eval_params(
            task_name=task_name,
            checkpoint_id=checkpoint_id,
            eval_params=eval_params,
            rerun_metadata=rerun_metadata,
        )
        pickle.dump(self.memory_recorder, open(self.file_name, "wb"))

    def record_output_feedback(
        self,
        task_name: str,
        checkpoint_id: str,
        feedback: OutputFeedback,
    ) -> None:
        self.memory_recorder.record_output_feedback(
            task_name=task_name,
            checkpoint_id=checkpoint_id,
            feedback=feedback,
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

    def record_comparison_result(
        self,
        task_name: str,
        input_hash: str,
        result: ComparisonResult,
    ) -> None:
        self.memory_recorder.record_comparison_result(
            task_name=task_name, input_hash=input_hash, result=result
        )
        pickle.dump(self.memory_recorder, open(self.file_name, "wb"))

    def record_output_score(
        self,
        task_name: str,
        checkpoint_id: str,
        score: float,
    ) -> None:
        self.memory_recorder.record_output_score(
            task_name=task_name, checkpoint_id=checkpoint_id, score=score
        )
        pickle.dump(self.memory_recorder, open(self.file_name, "wb"))

    def get_comparison_result(
        self,
        task_name: str,
        checkpoint_id_a: str,
        checkpoint_id_b: str,
    ) -> Optional[ComparisonResult]:
        return self.memory_recorder.get_comparison_result(
            task_name=task_name,
            checkpoint_id_a=checkpoint_id_a,
            checkpoint_id_b=checkpoint_id_b,
        )

    def get_comparison_results_for_input_hash(
        self,
        task_name: str,
        input_hash: str,
    ) -> list[ComparisonResult]:
        return self.memory_recorder.get_comparison_results_for_input_hash(
            task_name=task_name, input_hash=input_hash
        )

    def record_output(
        self,
        task_name: str,
        checkpoint_id: str,
        output: str,
    ) -> None:
        self.memory_recorder.record_output(
            task_name=task_name,
            checkpoint_id=checkpoint_id,
            output=output,
        )
        pickle.dump(self.memory_recorder, open(self.file_name, "wb"))
