import pytest
from unittest.mock import MagicMock

def test_get_example_returns_none_when_task_name_does_not_exist():
    recorder = MemoryRecorder()
    example = recorder.get_example('nonexistent_task', 'example_id', 'checkpoint_id')
    assert example is None

def test_get_example_returns_none_when_example_does_not_exist():
    recorder = MemoryRecorder()
    task = MagicMock()
    task.records = {}
    recorder.tasks = {'task_name': task}
    example = recorder.get_example('task_name', 'nonexistent_example', 'checkpoint_id')
    assert example is None

def test_get_example_returns_none_when_checkpoint_does_not_exist():
    recorder = MemoryRecorder()
    task = MagicMock()
    task.records = {'example_id:checkpoint_id': MagicMock()}
    recorder.tasks = {'task_name': task}
    recorder.get_latest_checkpoints = MagicMock(return_value=[])
    example = recorder.get_example('task_name', 'example_id', 'nonexistent_checkpoint')
    assert example is None

def test_get_example_returns_latest_checkpoint_when_checkpoint_id_is_none():
    recorder = MemoryRecorder()
    task = MagicMock()
    example = Example(id='example_id', checkpoint_id='latest_checkpoint', variables={}, output_variable_names=set(), params={}, feedback=None, feedback_details=None)
    task.records = {'example_id:latest_checkpoint': example}
    recorder.tasks = {'task_name': task}
    recorder.get_latest_checkpoints = MagicMock(return_value=['latest_checkpoint'])
    result = recorder.get_example('task_name', 'example_id', None)
    assert result == example

def test_get_example_returns_specified_checkpoint_when_checkpoint_id_is_not_none():
    recorder = MemoryRecorder()
    task = MagicMock()
    example = Example(id='example_id', checkpoint_id='specified_checkpoint', variables={}, output_variable_names=set(), params={}, feedback=None, feedback_details=None)
    task.records = {'example_id:specified_checkpoint': example}
    recorder.tasks = {'task_name': task}
    result = recorder.get_example('task_name', 'example_id', 'specified_checkpoint')
    assert result == example

def test_get_example_returns_none_when_task_name_is_none():
    recorder = MemoryRecorder()
    example = recorder.get_example(None, 'example_id', 'checkpoint_id')
    assert example is None

def test_get_example_returns_none_when_example_id_is_none():
    recorder = MemoryRecorder()
    example = recorder.get_example('task_name', None, 'checkpoint_id')
    assert example is None