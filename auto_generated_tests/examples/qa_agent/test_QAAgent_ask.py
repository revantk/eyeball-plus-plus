import pytest
from unittest.mock import MagicMock, patch
from examples.qa_agent import QAAgent

@patch('openai.ChatCompletion.create')
@patch('eyeball_pp.get_eval_param')
@patch('eyeball_pp.record_intermediary_state')
@patch.object(QAAgent, '_get_context')
def test_ask(mock_get_context, mock_record_intermediary_state, mock_get_eval_param, mock_chat_create):
    """
    Test the ask function with a specific question and context.
    This test checks if the function correctly generates the prompt and passes it to the openai.ChatCompletion.create method.
    """
    mock_get_context.return_value = 'This is the context.'
    mock_get_eval_param.side_effect = ['gpt-3.5-turbo', 0.5]
    mock_chat_create.return_value = {'choices': [{'message': {'content': 'This is the answer.'}}]}
    agent = QAAgent()
    question = 'What is the meaning of life?'
    answer = agent.ask(question)
    mock_get_context.assert_called_once_with(question)
    mock_record_intermediary_state.assert_called_once_with('context', 'This is the context.')
    mock_get_eval_param.assert_any_call('model')
    mock_get_eval_param.assert_any_call('temperature')
    mock_chat_create.assert_called_once()
    assert answer == 'This is the answer.'

@patch('openai.ChatCompletion.create')
@patch('eyeball_pp.get_eval_param')
@patch('eyeball_pp.record_intermediary_state')
@patch.object(QAAgent, '_get_context')
def test_ask_no_context(mock_get_context, mock_record_intermediary_state, mock_get_eval_param, mock_chat_create):
    """
    Test the ask function with a specific question but no context.
    This test checks if the function correctly handles the case where there is no context for the question.
    """
    mock_get_context.return_value = ''
    mock_get_eval_param.side_effect = ['gpt-3.5-turbo', 0.5]
    mock_chat_create.return_value = {'choices': [{'message': {'content': "I don't know."}}]}
    agent = QAAgent()
    question = 'What is the meaning of life?'
    answer = agent.ask(question)
    mock_get_context.assert_called_once_with(question)
    mock_record_intermediary_state.assert_called_once_with('context', '')
    mock_get_eval_param.assert_any_call('model')
    mock_get_eval_param.assert_any_call('temperature')
    mock_chat_create.assert_called_once()
    assert answer == "I don't know."