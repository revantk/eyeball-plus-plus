import pytest
import json
from unittest.mock import MagicMock
from typing import Any, TypeVar, Optional
T = TypeVar('T', int, float, str, bool, bytes, dict, list, None)

def test_fetch_variable_value_with_existing_variable():
    """
    Test that _fetch_variable_value returns the correct value when the variable exists in the example.
    """
    example_mock = MagicMock()
    example_mock.variables = {'test_var': 'test_value'}
    recorder_mock = MagicMock()
    recorder_mock.get_example.return_value = example_mock
    evaluator = Evaluator()
    evaluator.recorder = recorder_mock
    result = evaluator._fetch_variable_value('test_task', 'test_var', 'default_value', 'test_example', 'test_checkpoint')
    assert result == 'test_value'

def test_fetch_variable_value_with_nonexistent_variable():
    """
    Test that _fetch_variable_value returns the passed-in value when the variable does not exist in the example.
    """
    example_mock = MagicMock()
    example_mock.variables = {}
    recorder_mock = MagicMock()
    recorder_mock.get_example.return_value = example_mock
    evaluator = Evaluator()
    evaluator.recorder = recorder_mock
    result = evaluator._fetch_variable_value('test_task', 'test_var', 'default_value', 'test_example', 'test_checkpoint')
    assert result == 'default_value'

def test_fetch_variable_value_with_nonexistent_example():
    """
    Test that _fetch_variable_value returns the passed-in value when the example does not exist in the recorder.
    """
    recorder_mock = MagicMock()
    recorder_mock.get_example.return_value = None
    evaluator = Evaluator()
    evaluator.recorder = recorder_mock
    result = evaluator._fetch_variable_value('test_task', 'test_var', 'default_value', 'test_example', 'test_checkpoint')
    assert result == 'default_value'

def test_fetch_variable_value_with_json_serializable_passed_in_value():
    """
    Test that _fetch_variable_value returns the correct value when the passed-in value is a JsonSerializable object.
    """
    example_mock = MagicMock()
    example_mock.variables = {'test_var': '{"test_key": "test_value"}'}
    recorder_mock = MagicMock()
    recorder_mock.get_example.return_value = example_mock

    class JsonSerializableMock(JsonSerializable):

        @staticmethod
        def from_json(json_str):
            return json.loads(json_str)
    evaluator = Evaluator()
    evaluator.recorder = recorder_mock
    result = evaluator._fetch_variable_value('test_task', 'test_var', JsonSerializableMock(), 'test_example', 'test_checkpoint')
    assert result == {'test_key': 'test_value'}

def test_fetch_variable_value_with_non_json_serializable_passed_in_value():
    """
    Test that _fetch_variable_value returns the correct value when the passed-in value is not a JsonSerializable object.
    """
    example_mock = MagicMock()
    example_mock.variables = {'test_var': '{"test_key": "test_value"}'}
    recorder_mock = MagicMock()
    recorder_mock.get_example.return_value = example_mock
    evaluator = Evaluator()
    evaluator.recorder = recorder_mock
    result = evaluator._fetch_variable_value('test_task', 'test_var', 'default_value', 'test_example', 'test_checkpoint')
    assert result == {'test_key': 'test_value'}