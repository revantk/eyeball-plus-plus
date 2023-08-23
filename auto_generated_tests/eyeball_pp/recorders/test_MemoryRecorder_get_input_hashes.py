import pytest
from unittest.mock import MagicMock
from eyeball_pp.recorders import MemoryRecorder, Task

def test_get_input_hashes_no_task():
    """
    Test get_input_hashes when the task does not exist.
    """
    recorder = MemoryRecorder()
    assert recorder.get_input_hashes('nonexistent_task') == []

def test_get_input_hashes_no_input_names():
    """
    Test get_input_hashes when input_names is not provided.
    """
    recorder = MemoryRecorder()
    task = Task(name='task1')
    task.input_hashes = {'hash1': {'input1'}, 'hash2': {'input2'}}
    recorder.tasks = {'task1': task}
    assert recorder.get_input_hashes('task1') == list(task.input_hashes.keys())

def test_get_input_hashes_with_input_names():
    """
    Test get_input_hashes when input_names is provided.
    """
    recorder = MemoryRecorder()
    task = Task(name='task1')
    task.input_hashes = {'hash1': {'input1', 'input2'}, 'hash2': {'input2'}}
    task.input_hash_to_input_names = {'hash1': {'input1', 'input2'}, 'hash2': {'input2'}}
    recorder.tasks = {'task1': task}
    assert recorder.get_input_hashes('task1', ['input1', 'input2']) == ['hash1']

def test_get_input_hashes_with_limit():
    """
    Test get_input_hashes when limit is provided.
    """
    recorder = MemoryRecorder()
    task = Task(name='task1')
    task.input_hashes = {'hash1': {'input1'}, 'hash2': {'input2'}, 'hash3': {'input3'}}
    recorder.tasks = {'task1': task}
    assert recorder.get_input_hashes('task1', limit=2) == ['hash1', 'hash2']

def test_get_input_hashes_remove_empty_hashes():
    """
    Test get_input_hashes removes input hashes with no corresponding input names.
    """
    recorder = MemoryRecorder()
    task = Task(name='task1')
    task.input_hashes = {'hash1': {'input1'}, 'hash2': set()}
    recorder.tasks = {'task1': task}
    assert recorder.get_input_hashes('task1') == ['hash1']