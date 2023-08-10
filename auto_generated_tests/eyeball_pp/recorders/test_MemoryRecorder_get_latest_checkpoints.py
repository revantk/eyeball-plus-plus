import pytest
from eyeball_pp.recorders import MemoryRecorder, Task, Checkpoint

def test_get_latest_checkpoints_no_task():
    """
    Test that an empty list is returned when the task does not exist.
    """
    recorder = MemoryRecorder()
    assert recorder.get_latest_checkpoints('nonexistent_task', 'input_hash') == []

def test_get_latest_checkpoints_no_input_hash():
    """
    Test that an empty list is returned when the input hash does not exist.
    """
    recorder = MemoryRecorder()
    task = Task(name='task1')
    recorder.tasks['task1'] = task
    assert recorder.get_latest_checkpoints('task1', 'nonexistent_input_hash') == []

def test_get_latest_checkpoints_with_checkpoints():
    """
    Test that the latest checkpoints are returned when they exist.
    """
    recorder = MemoryRecorder()
    task = Task(name='task1')
    checkpoint1 = Checkpoint(checkpoint_id='1')
    checkpoint2 = Checkpoint(checkpoint_id='2')
    task.checkpoints['1'] = checkpoint1
    task.checkpoints['2'] = checkpoint2
    task.input_hashes['input_hash'] = set(['1', '2'])
    recorder.tasks['task1'] = task
    assert recorder.get_latest_checkpoints('task1', 'input_hash') == [checkpoint1, checkpoint2]

def test_get_latest_checkpoints_with_num_checkpoints():
    """
    Test that the specified number of latest checkpoints are returned.
    """
    recorder = MemoryRecorder()
    task = Task(name='task1')
    checkpoint1 = Checkpoint(checkpoint_id='1')
    checkpoint2 = Checkpoint(checkpoint_id='2')
    checkpoint3 = Checkpoint(checkpoint_id='3')
    task.checkpoints['1'] = checkpoint1
    task.checkpoints['2'] = checkpoint2
    task.checkpoints['3'] = checkpoint3
    task.input_hashes['input_hash'] = set(['1', '2', '3'])
    recorder.tasks['task1'] = task
    assert recorder.get_latest_checkpoints('task1', 'input_hash', num_checkpoints=2) == [checkpoint2, checkpoint3]