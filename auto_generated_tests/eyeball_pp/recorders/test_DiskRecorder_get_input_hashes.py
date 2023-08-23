import pytest
from unittest.mock import MagicMock
from eyeball_pp.recorders import DiskRecorder

def test_get_input_hashes_no_limit_no_input_names():
    """
    Test get_input_hashes with no limit and no input names.
    """
    mock_memory_recorder = MagicMock()
    mock_memory_recorder.get_input_hashes.return_value = ['hash1', 'hash2', 'hash3']
    recorder = DiskRecorder('dir_path')
    recorder.memory_recorder = mock_memory_recorder
    result = recorder.get_input_hashes('task1')
    mock_memory_recorder.get_input_hashes.assert_called_once_with('task1', input_names=None, limit=None)
    assert result == ['hash1', 'hash2', 'hash3']

def test_get_input_hashes_with_limit_no_input_names():
    """
    Test get_input_hashes with a limit and no input names.
    """
    mock_memory_recorder = MagicMock()
    mock_memory_recorder.get_input_hashes.return_value = ['hash1', 'hash2']
    recorder = DiskRecorder('dir_path')
    recorder.memory_recorder = mock_memory_recorder
    result = recorder.get_input_hashes('task1', limit=2)
    mock_memory_recorder.get_input_hashes.assert_called_once_with('task1', input_names=None, limit=2)
    assert result == ['hash1', 'hash2']

def test_get_input_hashes_no_limit_with_input_names():
    """
    Test get_input_hashes with no limit and input names.
    """
    mock_memory_recorder = MagicMock()
    mock_memory_recorder.get_input_hashes.return_value = ['hash1', 'hash2', 'hash3']
    recorder = DiskRecorder('dir_path')
    recorder.memory_recorder = mock_memory_recorder
    result = recorder.get_input_hashes('task1', input_names=['input1', 'input2'])
    mock_memory_recorder.get_input_hashes.assert_called_once_with('task1', input_names=['input1', 'input2'], limit=None)
    assert result == ['hash1', 'hash2', 'hash3']

def test_get_input_hashes_with_limit_with_input_names():
    """
    Test get_input_hashes with a limit and input names.
    """
    mock_memory_recorder = MagicMock()
    mock_memory_recorder.get_input_hashes.return_value = ['hash1']
    recorder = DiskRecorder('dir_path')
    recorder.memory_recorder = mock_memory_recorder
    result = recorder.get_input_hashes('task1', input_names=['input1'], limit=1)
    mock_memory_recorder.get_input_hashes.assert_called_once_with('task1', input_names=['input1'], limit=1)
    assert result == ['hash1']