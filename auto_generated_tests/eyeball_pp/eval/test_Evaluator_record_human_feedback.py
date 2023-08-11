import pytest
from unittest.mock import MagicMock
from eyeball_pp.eval import Evaluator
from eyeball_pp.classes import FeedbackResult, OutputFeedback

def test_record_human_feedback_positive():
    """
    Test that record_human_feedback correctly retrieves the recorder state for positive feedback.
    """
    evaluator = Evaluator()
    evaluator._get_recorder_state = MagicMock(return_value=('task1', 'checkpoint1'))
    feedback = OutputFeedback(FeedbackResult.POSITIVE, 'Good job!')
    evaluator.record_human_feedback(feedback, 'Additional details', 'task1', 'checkpoint1')
    evaluator._get_recorder_state.assert_called_once_with('task1', 'checkpoint1')

def test_record_human_feedback_negative():
    """
    Test that record_human_feedback correctly retrieves the recorder state for negative feedback.
    """
    evaluator = Evaluator()
    evaluator._get_recorder_state = MagicMock(return_value=('task2', 'checkpoint2'))
    feedback = OutputFeedback(FeedbackResult.NEGATIVE, 'Needs improvement.')
    evaluator.record_human_feedback(feedback, 'Additional details', 'task2', 'checkpoint2')
    evaluator._get_recorder_state.assert_called_once_with('task2', 'checkpoint2')

def test_record_human_feedback_neutral():
    """
    Test that record_human_feedback correctly retrieves the recorder state for neutral feedback.
    """
    evaluator = Evaluator()
    evaluator._get_recorder_state = MagicMock(return_value=('task3', 'checkpoint3'))
    feedback = OutputFeedback(FeedbackResult.NEUTRAL, 'Average performance.')
    evaluator.record_human_feedback(feedback, 'Additional details', 'task3', 'checkpoint3')
    evaluator._get_recorder_state.assert_called_once_with('task3', 'checkpoint3')

def test_record_human_feedback_no_task_checkpoint():
    """
    Test that record_human_feedback correctly retrieves the recorder state when no task name or checkpoint id is provided.
    """
    evaluator = Evaluator()
    evaluator._get_recorder_state = MagicMock(return_value=(None, None))
    feedback = OutputFeedback(FeedbackResult.POSITIVE, 'Good job!')
    evaluator.record_human_feedback(feedback, 'Additional details')
    evaluator._get_recorder_state.assert_called_once_with(None, None)