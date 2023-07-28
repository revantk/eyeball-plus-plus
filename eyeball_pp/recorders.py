from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import pkg_resources
import yaml
import os
import pickle
from typing import Any, Optional, Protocol
from dataclasses import dataclass
import dataclasses
import requests

from .utils import LruCache
from .classes import OutputFeedback, OutputScore
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    older_checkpoint_id: str
    newer_checkpoint_id: str
    output_feedback: OutputFeedback

    def as_dict(self):
        return {
            "older_checkpoint_id": self.older_checkpoint_id,
            "newer_checkpoint_id": self.newer_checkpoint_id,
            "output_feedback": self.output_feedback.as_dict(),
        }


def get_input_hash(input_variables: dict[str, str]) -> str:
    sorted_input_vars = sorted(
        [
            (str(var_name), str(var_val))
            for var_name, var_val in input_variables.items()
        ],
        key=lambda x: x[0],
    )
    input_var_str = json.dumps(sorted_input_vars)
    return hashlib.sha256(input_var_str.encode("utf-8")).hexdigest()


@dataclass
class Checkpoint:
    checkpoint_id: str
    input_variables: dict[str, str] = dataclasses.field(default_factory=dict)
    eval_params: dict[str, Any] = dataclasses.field(default_factory=dict)
    output: Optional[str] = None
    output_feedback: Optional[OutputFeedback] = None
    output_score: Optional[OutputScore] = None
    rerun_metadata: dict[str, str] = dataclasses.field(default_factory=dict)

    def get_input_variables(self) -> dict[str, str]:
        return dict(self.input_variables)

    def get_input_hash(self) -> str:
        return get_input_hash(self.input_variables)

    def get_input_var_str(self) -> str:
        sorted_input_vars = sorted(
            [
                (str(var_name), str(var_val))
                for var_name, var_val in self.input_variables.items()
            ],
            key=lambda x: x[0],
        )
        msg = ""
        for var_name, var_val in sorted_input_vars:
            msg += f"{var_name}={var_val}\n"
        return msg

    def __str__(self) -> str:
        msg = f"Checkpoint: {self.checkpoint_id}\n"
        msg += self.get_input_var_str()
        return msg

    def as_dict(self) -> dict[str, Any]:
        checkpoint_dict = {
            "checkpoint_id": self.checkpoint_id,
            "input_variables": self.input_variables,
            "output": self.output,
        }
        if self.eval_params:
            checkpoint_dict["eval_params"] = self.eval_params
        if self.output_feedback is not None:
            checkpoint_dict["output_feedback"] = self.output_feedback.as_dict()
        if self.output_score is not None:
            checkpoint_dict["output_score"] = self.output_score.as_dict()
        if self.rerun_metadata:
            checkpoint_dict["rerun_metadata"] = self.rerun_metadata

        return checkpoint_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        return cls(
            checkpoint_id=data["checkpoint_id"],
            input_variables=data["input_variables"],
            eval_params=data.get("eval_params") or {},
            output=data["output"],
            output_feedback=OutputFeedback.from_dict(data["output_feedback"])
            if data.get("output_feedback") is not None
            else None,
            output_score=OutputScore.from_dict(data["output_score"])
            if data.get("output_score") is not None
            else None,
            rerun_metadata=data.get("rerun_metadata") or {},
        )


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

    def record_eval_params(
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

    def record_output_score(
        self,
        task_name: str,
        checkpoint_id: str,
        score: OutputScore,
    ) -> None:
        ...

    def get_checkpoint(
        self, task_name: str, checkpoint_id: str
    ) -> Optional[Checkpoint]:
        ...

    def get_latest_checkpoints(
        self, task_name: str, input_hash: str, num_checkpoints: int = 2
    ) -> list[Checkpoint]:
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

    def get_comparison_result(
        self,
        task_name: str,
        older_checkpoint_id: str,
        newer_checkpoint_id: str,
    ) -> Optional[ComparisonResult]:
        ...

    def get_comparison_results_for_input_hash(
        self, task_name: str, input_hash: str, num_results: int = 3
    ) -> list[ComparisonResult]:
        ...

    def delete_checkpoints_for_input_hash(
        self,
        task_name: str,
        input_hash: str,
    ) -> None:
        ...

    def edit_checkpoint(
        self, task_name: str, checkpoint_id: str, edit_dict: dict[str, Any]
    ) -> None:
        ...


class ApiClientRecorder(EvalRecorder):
    def __init__(self, api_key: str, api_url: Optional[str] = None) -> None:
        self.api_key = api_key
        if api_url is None:
            self.url = "https://eyeball.tark.ai/"
        else:
            self.url = api_url
        self.checkpoint_dicts: LruCache = LruCache(max_size=100)
        self.pool = ThreadPoolExecutor(max_workers=1)

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    def _record_checkpoint(
        self,
        task_name: str,
        checkpoint_id: str,
        prefixes: list[str],
        name: str,
        value: Any,
        flush: bool = False,
    ) -> dict[str, Any]:
        dict_key = f"{task_name},{checkpoint_id}"
        if dict_key not in self.checkpoint_dicts:
            checkpoint_dict: dict[str, Any] = {"checkpoint_id": checkpoint_id}
        else:
            checkpoint_dict = self.checkpoint_dicts[dict_key]

        current_dict = checkpoint_dict
        for prefix in prefixes:
            if prefix not in current_dict:
                current_dict[prefix] = {}
            current_dict = current_dict[prefix]
        current_dict[name] = value
        self.checkpoint_dicts[dict_key] = dict(checkpoint_dict)

        if flush:

            def _record():
                response = requests.post(
                    f"{self.url}/record_checkpoint",
                    json={
                        "task_name": task_name,
                        "checkpoint_id": checkpoint_id,
                        **checkpoint_dict,
                    },
                    headers=self._get_headers(),
                )
                if response.status_code != 200:
                    logger.error(
                        f"Failed to record checkpoint {checkpoint_id} -- {response.status_code}, {response.text}"
                    )
                else:
                    logger.debug(f"Recorded checkpoint {checkpoint_id}")

            self.pool.submit(_record)
        return checkpoint_dict

    def record_input_variable(
        self, task_name: str, checkpoint_id: str, variable_name: str, value: str
    ) -> None:
        self._record_checkpoint(
            task_name=task_name,
            checkpoint_id=checkpoint_id,
            prefixes=["input_variables"],
            name=variable_name,
            value=value,
        )

    def record_output(self, task_name: str, checkpoint_id: str, output: str) -> None:
        self._record_checkpoint(
            task_name=task_name,
            checkpoint_id=checkpoint_id,
            prefixes=[],
            name="output",
            value=output,
            flush=True,
        )

    def record_eval_params(
        self,
        task_name: str,
        checkpoint_id: str,
        eval_params: dict[str, Any],
        rerun_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._record_checkpoint(
            task_name=task_name,
            checkpoint_id=checkpoint_id,
            prefixes=[],
            name="eval_params",
            value=eval_params,
        )
        if rerun_metadata is not None:
            self._record_checkpoint(
                task_name=task_name,
                checkpoint_id=checkpoint_id,
                prefixes=[],
                name="rerun_metadata",
                value=rerun_metadata,
            )

    def record_comparison_result(
        self, task_name: str, input_hash: str, result: ComparisonResult
    ) -> None:
        requests.post(
            f"{self.url}/record_comparison_result",
            json={
                "task_name": task_name,
                "input_hash": input_hash,
                "result": result.as_dict(),
            },
            headers=self._get_headers(),
        )

    def record_output_score(
        self, task_name: str, checkpoint_id: str, score: OutputScore
    ) -> None:
        requests.post(
            f"{self.url}/record_output_score",
            json={
                "task_name": task_name,
                "checkpoint_id": checkpoint_id,
                "score": score.as_dict(),
            },
            headers=self._get_headers(),
        )

    def record_output_feedback(
        self, task_name: str, checkpoint_id: str, feedback: OutputFeedback
    ) -> None:
        requests.post(
            f"{self.url}/record_output_feedback",
            json={
                "task_name": task_name,
                "checkpoint_id": checkpoint_id,
                "feedback": feedback.as_dict(),
            },
            headers=self._get_headers(),
        )

    def delete_checkpoints_for_input_hash(
        self, task_name: str, input_hash: str
    ) -> None:
        ...

    def get_checkpoint(self, task_name: str, checkpoint_id: str) -> Checkpoint | None:
        dict_key = f"{task_name},{checkpoint_id}"
        if dict_key in self.checkpoint_dicts:
            return Checkpoint(**(self.checkpoint_dicts[dict_key]))

        response = requests.get(
            f"{self.url}/get_checkpoint",
            json={
                "task_name": task_name,
                "checkpoint_id": checkpoint_id,
            },
            headers=self._get_headers(),
        )
        if response.status_code != 200:
            logger.debug(
                f"Failed to get checkpoint - {checkpoint_id}: {response.status_code}"
            )
            return None
        return Checkpoint(**response.json())

    def get_comparison_result(
        self, task_name: str, older_checkpoint_id: str, newer_checkpoint_id: str
    ) -> ComparisonResult | None:
        ...

    def get_comparison_results_for_input_hash(
        self, task_name: str, input_hash: str, num_results: int = 3
    ) -> list[ComparisonResult]:
        return []

    def get_input_hashes(self, task_name: str) -> list[str]:
        response = requests.get(
            f"{self.url}/get_input_hashes",
            json={
                "task_name": task_name,
            },
            headers=self._get_headers(),
        )
        if response.status_code != 200:
            logger.debug(f"Failed to get input hashes: {response.status_code}")
            return []
        return response.json()["input_hashes"]

    def get_latest_checkpoints(
        self, task_name: str, input_hash: str, num_checkpoints: int = 2
    ) -> list[Checkpoint]:
        response = requests.get(
            f"{self.url}/get_latest_checkpoints",
            json={
                "task_name": task_name,
                "input_hash": input_hash,
                "num_checkpoints": num_checkpoints,
            },
            headers=self._get_headers(),
        )
        if response.status_code != 200:
            logger.debug(f"Failed to get latest checkpoints: {response.status_code}")
            return []
        return [Checkpoint(**data) for data in response.json()["checkpoints"]]

    def get_task_names(self) -> list[str]:
        response = requests.get(
            f"{self.url}/get_task_names",
            headers=self._get_headers(),
        )
        if response.status_code != 200:
            logger.debug(f"Failed to get task names: {response.status_code}")
            return []
        return response.json()["task_names"]

    def edit_checkpoint(
        self, task_name: str, checkpoint_id: str, edit_dict: dict[str, Any]
    ) -> None:
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
    ) -> list[Checkpoint]:
        if task_name not in self.tasks:
            return []
        task = self.tasks[task_name]
        if input_hash not in task.input_hashes:
            return []

        return [
            task.checkpoints[checkpoint_id]
            for checkpoint_id in sorted(task.input_hashes[input_hash])[
                -num_checkpoints:
            ]
        ]

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

    def record_eval_params(
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
        key = f"{result.older_checkpoint_id},{result.newer_checkpoint_id}"
        task.comparison_results[key] = result

        if input_hash not in task.input_hash_to_comparison_results:
            task.input_hash_to_comparison_results[input_hash] = set()
        task.input_hash_to_comparison_results[input_hash].add(key)

    def record_output_score(
        self,
        task_name: str,
        checkpoint_id: str,
        score: OutputScore,
    ) -> None:
        checkpoint = self._fetch_or_create_checkpoint(
            task_name=task_name, checkpoint_id=checkpoint_id
        )
        checkpoint.output_score = score

    def get_comparison_result(
        self,
        task_name: str,
        older_checkpoint_id: str,
        newer_checkpoint_id: str,
    ) -> Optional[ComparisonResult]:
        task = self.tasks[task_name]
        key = f"{older_checkpoint_id},{newer_checkpoint_id}"
        if key not in task.comparison_results:
            return None
        return task.comparison_results[key]

    def get_comparison_results_for_input_hash(
        self,
        task_name: str,
        input_hash: str,
        num_results: int = 3,
    ) -> list[ComparisonResult]:
        task = self.tasks[task_name]
        if input_hash not in task.input_hash_to_comparison_results:
            return []
        comparison_results = task.input_hash_to_comparison_results[input_hash]
        return [task.comparison_results[key] for key in comparison_results][
            :num_results
        ]

    def delete_checkpoints_for_input_hash(
        self,
        task_name: str,
        input_hash: str,
    ) -> None:
        task = self.tasks[task_name]
        if input_hash not in task.input_hashes:
            return
        for checkpoint_id in task.input_hashes[input_hash]:
            del task.checkpoints[checkpoint_id]
        del task.input_hashes[input_hash]

    def edit_checkpoint(
        self, task_name: str, checkpoint_id: str, edit_dict: dict[str, Any]
    ) -> None:
        # TODO: implement
        return None


class FileRecorder(EvalRecorder):
    def __init__(self, dir_path: str) -> None:
        self.dir_path = dir_path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        self.yaml_dicts: dict[str, dict[str, str]] = {}

    def edit_checkpoint(
        self, task_name: str, checkpoint_id: str, edit_dict: dict[str, Any]
    ) -> None:
        old_checkpoint = self.get_checkpoint(task_name, checkpoint_id)
        if old_checkpoint is None:
            return
        old_input_hash = old_checkpoint.get_input_hash()
        checkpoint_dict = old_checkpoint.as_dict()
        checkpoint_dict.update(edit_dict)
        new_checkpoint = Checkpoint.from_dict(checkpoint_dict)
        new_input_hash = new_checkpoint.get_input_hash()
        checkpoint_dict["input_hash"] = new_input_hash

        dir_name = os.path.join(self.dir_path, task_name, "checkpoints")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = os.path.join(dir_name, f"{checkpoint_id}.yaml")
        yaml.dump(checkpoint_dict, open(file_name, "w+"))

        if old_input_hash != new_input_hash:
            inputs_dir_name = os.path.join(self.dir_path, task_name, "inputs")
            if not os.path.exists(inputs_dir_name):
                os.makedirs(inputs_dir_name)

            input_file_name = os.path.join(
                self.dir_path, task_name, "inputs", f"{new_input_hash}.yaml"
            )
            yaml.dump(new_checkpoint.input_variables, open(input_file_name, "w+"))
            try:
                os.remove(os.path.join(inputs_dir_name, f"{old_input_hash}.yaml"))
            except FileNotFoundError:
                pass

    def _record_checkpoint(
        self,
        task_name: str,
        checkpoint_id: str,
        prefixes: list[str],
        name: str,
        value: Any,
        flush: bool = False,
    ) -> dict[str, Any]:
        dir_name = os.path.join(self.dir_path, task_name, "checkpoints")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = os.path.join(dir_name, f"{checkpoint_id}.yaml")

        if checkpoint_id not in self.yaml_dicts:
            if os.path.exists(file_name):
                yaml_dict = yaml.load(open(file_name, "r"), Loader=yaml.FullLoader)
            else:
                yaml_dict = {}
        else:
            yaml_dict = self.yaml_dicts[checkpoint_id]

        current_dict = yaml_dict
        for prefix in prefixes:
            if prefix not in current_dict:
                current_dict[prefix] = {}
            current_dict = current_dict[prefix]
        current_dict[name] = value
        if flush:
            yaml.dump(yaml_dict, open(file_name, "w+"))
            try:
                del self.yaml_dicts[checkpoint_id]
            except KeyError:
                pass
        else:
            self.yaml_dicts[checkpoint_id] = dict(yaml_dict)
        return yaml_dict

    def record_input_variable(
        self, task_name: str, checkpoint_id: str, variable_name: str, value: str
    ) -> None:
        self._record_checkpoint(
            task_name=task_name,
            checkpoint_id=checkpoint_id,
            prefixes=["input_variables"],
            name=variable_name,
            value=value,
        )

    def record_output(
        self,
        task_name: str,
        checkpoint_id: str,
        output: str,
    ) -> None:
        yaml_dict = self._record_checkpoint(
            task_name=task_name,
            checkpoint_id=checkpoint_id,
            prefixes=[],
            name="output",
            value=output,
        )

        input_hash = get_input_hash(yaml_dict["input_variables"])
        inputs_dir_name = os.path.join(self.dir_path, task_name, "inputs")
        if not os.path.exists(inputs_dir_name):
            os.makedirs(inputs_dir_name)

        input_file_name = os.path.join(
            self.dir_path, task_name, "inputs", f"{input_hash}.yaml"
        )
        yaml.dump(yaml_dict["input_variables"], open(input_file_name, "w+"))
        self._record_checkpoint(
            task_name=task_name,
            checkpoint_id=checkpoint_id,
            prefixes=[],
            name="input_hash",
            value=input_hash,
            flush=True,
        )

    def record_eval_params(
        self,
        task_name: str,
        checkpoint_id: str,
        eval_params: dict[str, Any],
        rerun_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._record_checkpoint(
            task_name=task_name,
            checkpoint_id=checkpoint_id,
            prefixes=[],
            name="eval_params",
            value=eval_params,
        )
        if rerun_metadata is not None:
            self._record_checkpoint(
                task_name=task_name,
                checkpoint_id=checkpoint_id,
                prefixes=[],
                name="rerun_metadata",
                value=rerun_metadata,
            )

    def record_output_feedback(
        self,
        task_name: str,
        checkpoint_id: str,
        feedback: OutputFeedback,
    ) -> None:
        self._record_checkpoint(
            task_name=task_name,
            checkpoint_id=checkpoint_id,
            prefixes=[],
            name="output_feedback",
            value=feedback.as_dict(),
            flush=True,
        )

    def record_output_score(
        self,
        task_name: str,
        checkpoint_id: str,
        score: OutputScore,
    ) -> None:
        self._record_checkpoint(
            task_name=task_name,
            checkpoint_id=checkpoint_id,
            prefixes=[],
            name="output_score",
            value=score.as_dict(),
            flush=True,
        )

    def _checkpoint_from_yaml_dict(
        self, yaml_dict: dict[str, Any], checkpoint_id: str
    ) -> Checkpoint:
        checkpoint = Checkpoint(checkpoint_id=checkpoint_id)
        checkpoint.input_variables = yaml_dict.get("input_variables", {})
        checkpoint.eval_params = yaml_dict.get("eval_params", {})
        checkpoint.output = yaml_dict.get("output")
        checkpoint.output_feedback = (
            OutputFeedback.from_dict(yaml_dict.get("output_feedback"))
            if "output_feedback" in yaml_dict
            else None
        )
        checkpoint.output_score = (
            OutputScore.from_dict(yaml_dict.get("output_score"))
            if "output_score" in yaml_dict
            else None
        )
        checkpoint.rerun_metadata = yaml_dict.get("rerun_metadata") or {}
        return checkpoint

    def get_checkpoint(
        self, task_name: str, checkpoint_id: str
    ) -> Optional[Checkpoint]:
        file_name = os.path.join(
            self.dir_path, task_name, "checkpoints", f"{checkpoint_id}.yaml"
        )
        if not os.path.exists(file_name):
            return None
        yaml_dict = yaml.load(open(file_name, "r"), Loader=yaml.FullLoader)
        yaml_dict["checkpoint_id"] = checkpoint_id
        return Checkpoint.from_dict(yaml_dict)

    def _read_checkpoints(self, task_name: str) -> dict[str, list[Checkpoint]]:
        input_hashes_to_checkpoints: dict[str, list[Checkpoint]] = {}
        dir_name = os.path.join(self.dir_path, task_name, "checkpoints")
        for file_name in os.listdir(dir_name):
            if file_name.endswith(".yaml"):
                file_path = os.path.join(dir_name, file_name)
                yaml_dict = yaml.load(open(file_path, "r"), Loader=yaml.FullLoader)
                if input_hash := yaml_dict.get("input_hash"):
                    if input_hash not in input_hashes_to_checkpoints:
                        input_hashes_to_checkpoints[input_hash] = []
                    yaml_dict["checkpoint_id"] = file_name[:-5]
                    input_hashes_to_checkpoints[input_hash].append(
                        Checkpoint.from_dict(yaml_dict)
                    )

        for input_hash in input_hashes_to_checkpoints:
            input_hashes_to_checkpoints[input_hash].sort(
                reverse=True, key=lambda x: x.checkpoint_id
            )
        return input_hashes_to_checkpoints

    def get_latest_checkpoints(
        self, task_name: str, input_hash: str, num_checkpoints: int = 2
    ) -> list[Checkpoint]:
        input_hashes_to_checkpoints = self._read_checkpoints(task_name)
        if input_hash not in input_hashes_to_checkpoints:
            return []
        return input_hashes_to_checkpoints[input_hash][:num_checkpoints]

    def get_input_hashes(self, task_name: str) -> list[str]:
        input_hashes = []
        for file_name in os.listdir(os.path.join(self.dir_path, task_name, "inputs")):
            if file_name.endswith(".yaml"):
                input_hashes.append(file_name[:-5])
        return input_hashes

    def get_task_names(self) -> list[str]:
        task_names = []
        for dir_name in os.listdir(self.dir_path):
            if os.path.isdir(os.path.join(self.dir_path, dir_name)):
                task_names.append(dir_name)
        return task_names

    def record_comparison_result(
        self,
        task_name: str,
        input_hash: str,
        result: ComparisonResult,
    ) -> None:
        comparison_dir = os.path.join(self.dir_path, task_name, "comparison_results")
        if not os.path.exists(comparison_dir):
            os.makedirs(comparison_dir)

        file_name = f"{input_hash}_{result.older_checkpoint_id}_{result.newer_checkpoint_id}.yaml"
        file_path = os.path.join(comparison_dir, file_name)
        yaml.dump(
            result.output_feedback.as_dict(),
            open(file_path, "w+"),
        )

    def get_comparison_result(
        self,
        task_name: str,
        older_checkpoint_id: str,
        newer_checkpoint_id: str,
    ) -> Optional[ComparisonResult]:
        comparison_dir = os.path.join(self.dir_path, task_name, "comparison_results")
        if not os.path.exists(comparison_dir):
            return None

        for file_name in os.listdir(comparison_dir):
            splits = os.path.splitext(file_name)[0].split("_")
            if len(splits) != 3:
                continue

            if splits[1] != older_checkpoint_id:
                continue

            if splits[2] != newer_checkpoint_id:
                continue

            if file_name.endswith(".yaml"):
                file_path = os.path.join(comparison_dir, file_name)
                yaml_dict = yaml.load(open(file_path, "r"), Loader=yaml.FullLoader)
                return ComparisonResult(
                    older_checkpoint_id=older_checkpoint_id,
                    newer_checkpoint_id=newer_checkpoint_id,
                    output_feedback=OutputFeedback.from_dict(yaml_dict),
                )
        return None

    def get_comparison_results_for_input_hash(
        self,
        task_name: str,
        input_hash: str,
        num_results: int = 3,
    ) -> list[ComparisonResult]:
        comparison_dir = os.path.join(self.dir_path, task_name, "comparison_results")
        if not os.path.exists(comparison_dir):
            return []

        results: list[ComparisonResult] = []
        for file_name in os.listdir(comparison_dir):
            splits = os.path.splitext(file_name)[0].split("_")
            if len(splits) != 3:
                continue

            if splits[0] != input_hash:
                continue

            if file_name.endswith(".yaml"):
                file_path = os.path.join(comparison_dir, file_name)
                yaml_dict = yaml.load(open(file_path, "r"), Loader=yaml.FullLoader)
                results.append(
                    ComparisonResult(
                        older_checkpoint_id=splits[1],
                        newer_checkpoint_id=splits[2],
                        output_feedback=OutputFeedback.from_dict(yaml_dict),
                    )
                )
        return sorted(results, key=lambda x: x.newer_checkpoint_id, reverse=True)[
            :num_results
        ]

    def delete_checkpoints_for_input_hash(
        self,
        task_name: str,
        input_hash: str,
    ) -> None:
        ...


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

    def record_eval_params(
        self,
        task_name: str,
        checkpoint_id: str,
        eval_params: dict[str, Any],
        rerun_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.memory_recorder.record_eval_params(
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
    ) -> list[Checkpoint]:
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
        score: OutputScore,
    ) -> None:
        self.memory_recorder.record_output_score(
            task_name=task_name, checkpoint_id=checkpoint_id, score=score
        )
        pickle.dump(self.memory_recorder, open(self.file_name, "wb"))

    def get_comparison_result(
        self,
        task_name: str,
        older_checkpoint_id: str,
        newer_checkpoint_id: str,
    ) -> Optional[ComparisonResult]:
        return self.memory_recorder.get_comparison_result(
            task_name=task_name,
            older_checkpoint_id=older_checkpoint_id,
            newer_checkpoint_id=newer_checkpoint_id,
        )

    def get_comparison_results_for_input_hash(
        self,
        task_name: str,
        input_hash: str,
        num_results: int = 3,
    ) -> list[ComparisonResult]:
        return self.memory_recorder.get_comparison_results_for_input_hash(
            task_name=task_name, input_hash=input_hash, num_results=num_results
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

    def delete_checkpoints_for_input_hash(
        self,
        task_name: str,
        input_hash: str,
    ) -> None:
        self.memory_recorder.delete_checkpoints_for_input_hash(
            task_name=task_name, input_hash=input_hash
        )
        pickle.dump(self.memory_recorder, open(self.file_name, "wb"))

    def compute_latest_comparison_results(self, task_name: str) -> None:
        ...
