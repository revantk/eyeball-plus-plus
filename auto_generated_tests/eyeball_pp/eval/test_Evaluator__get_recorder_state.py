import pytest
from contextlib import contextmanager

@contextmanager
def mock_recording_session(evaluator, task_name, example_id, checkpoint_id):
    evaluator.current_recorder_state.task_name = task_name
    evaluator.current_recorder_state.example_id = example_id
    evaluator.current_recorder_state.recorder_checkpoint_id = checkpoint_id
    yield
    delattr(evaluator.current_recorder_state, 'task_name')
    delattr(evaluator.current_recorder_state, 'example_id')
    delattr(evaluator.current_recorder_state, 'recorder_checkpoint_id')

def test_get_recorder_state_with_task_name_and_example_id():
    """
    Test that _get_recorder_state returns the correct task name, example id, and recorder checkpoint id when provided with a task name and example id.
    """
    evaluator = Evaluator()
    with mock_recording_session(evaluator, 'test_task', 'test_example', 'test_checkpoint'):
        (task_name, example_id, recorder_checkpoint_id) = evaluator._get_recorder_state()
        assert task_name == 'test_task'
        assert example_id == 'test_example'
        assert recorder_checkpoint_id == 'test_checkpoint'

def test_get_recorder_state_with_task_name_only():
    """
    Test that _get_recorder_state raises a ValueError when only provided with a task name.
    """
    evaluator = Evaluator()
    with mock_recording_session(evaluator, 'test_task', None, None):
        with pytest.raises(ValueError):
            evaluator._get_recorder_state()

def test_get_recorder_state_with_example_id_only():
    """
    Test that _get_recorder_state raises a ValueError when only provided with an example id.
    """
    evaluator = Evaluator()
    with mock_recording_session(evaluator, None, 'test_example', None):
        with pytest.raises(ValueError):
            evaluator._get_recorder_state()

def test_get_recorder_state_with_recorder_checkpoint_id_only():
    """
    Test that _get_recorder_state raises a ValueError when only provided with a recorder checkpoint id.
    """
    evaluator = Evaluator()
    with mock_recording_session(evaluator, None, None, 'test_checkpoint'):
        with pytest.raises(ValueError):
            evaluator._get_recorder_state()

def test_get_recorder_state_with_no_arguments():
    """
    Test that _get_recorder_state raises a ValueError when not provided with any arguments.
    """
    evaluator = Evaluator()
    with pytest.raises(ValueError):
        evaluator._get_recorder_state()