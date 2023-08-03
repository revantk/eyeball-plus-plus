import pytest
from unittest.mock import patch, MagicMock
from eyeball_pp.eval import Evaluator, EvaluatorMode
from eyeball_pp.classes import OutputFeedback, FeedbackResult

class MockCheckpoint:

    def __init__(self):
        self.output_feedback = None
        self.output = 'output'
        self.checkpoint_id = 'checkpoint_id'

    def get_input_var_str(self):
        return 'input_var_str'

def test_rate_recorded_examples_no_task_name_single_task():
    """
    Test rate_recorded_examples when no task_name is provided and there is only one task recorded
    """
    evaluator = Evaluator()
    evaluator.recorder = MagicMock()
    evaluator.recorder.get_task_names.return_value = ['task1']
    evaluator.recorder.get_input_hashes.return_value = ['hash1', 'hash2']
    evaluator.recorder.get_latest_checkpoints.return_value = [MockCheckpoint(), MockCheckpoint()]
    evaluator.request_user_feedback = MagicMock()
    evaluator.request_user_feedback.return_value = OutputFeedback(FeedbackResult.POSITIVE, 'Good job!')
    evaluator.rate_recorded_examples()
    assert evaluator.mode == EvaluatorMode.RECORD
    evaluator.recorder.get_task_names.assert_called_once()
    evaluator.recorder.get_input_hashes.assert_called_once_with(task_name='task1')
    assert evaluator.request_user_feedback.call_count == 2

def test_rate_recorded_examples_no_task_name_multiple_tasks():
    """
    Test rate_recorded_examples when no task_name is provided and there are multiple tasks recorded
    """
    evaluator = Evaluator()
    evaluator.recorder = MagicMock()
    evaluator.recorder.get_task_names.return_value = ['task1', 'task2']
    with pytest.raises(ValueError):
        evaluator.rate_recorded_examples()
    evaluator.recorder.get_task_names.assert_called_once()

def test_rate_recorded_examples_with_task_name():
    """
    Test rate_recorded_examples when task_name is provided
    """
    evaluator = Evaluator()
    evaluator.recorder = MagicMock()
    evaluator.recorder.get_input_hashes.return_value = ['hash1']
    evaluator.recorder.get_latest_checkpoints.return_value = [MockCheckpoint()]
    evaluator.request_user_feedback = MagicMock()
    evaluator.request_user_feedback.return_value = OutputFeedback(FeedbackResult.POSITIVE, 'Good job!')
    evaluator.rate_recorded_examples(task_name='task1')
    assert evaluator.mode == EvaluatorMode.RECORD
    evaluator.recorder.get_input_hashes.assert_called_once_with(task_name='task1')
    evaluator.request_user_feedback.assert_called_once()

def test_rate_recorded_examples_no_input_hashes():
    """
    Test rate_recorded_examples when no input_hashes are provided
    """
    evaluator = Evaluator()
    evaluator.recorder = MagicMock()
    evaluator.recorder.get_input_hashes.return_value = []
    evaluator.request_user_feedback = MagicMock()
    evaluator.rate_recorded_examples(task_name='task1')
    assert evaluator.mode == EvaluatorMode.RECORD
    evaluator.recorder.get_input_hashes.assert_called_once_with(task_name='task1')
    evaluator.request_user_feedback.assert_not_called()