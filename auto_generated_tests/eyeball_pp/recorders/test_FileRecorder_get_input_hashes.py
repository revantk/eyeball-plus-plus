import pytest
import os
from unittest.mock import patch, mock_open
from pyfakefs.fake_filesystem_unittest import TestCase
from eyeball_pp.recorders import FileRecorder

class TestFileRecorder(TestCase):

    def setUp(self):
        self.setUpPyfakefs()

    @patch('os.listdir')
    @patch('os.path.join')
    @patch('builtins.open', new_callable=mock_open, read_data='input1: value1\ninput2: value2')
    @patch('yaml.load')
    def test_get_input_hashes_no_input_names(self, mock_yaml_load, mock_open, mock_path_join, mock_listdir):
        """
        Test get_input_hashes when input_names is None. It should return the input hashes for all input files.
        """
        mock_listdir.return_value = ['input1.yaml', 'input2.yaml']
        mock_path_join.return_value = '/path/to/inputs'
        mock_yaml_load.return_value = {'input1': 'value1', 'input2': 'value2'}
        recorder = FileRecorder('/path/to')
        input_hashes = recorder.get_input_hashes('task1')
        assert input_hashes == ['input1', 'input2']

    @patch('os.listdir')
    @patch('os.path.join')
    @patch('builtins.open', new_callable=mock_open, read_data='input1: value1\ninput2: value2')
    @patch('yaml.load')
    def test_get_input_hashes_with_input_names(self, mock_yaml_load, mock_open, mock_path_join, mock_listdir):
        """
        Test get_input_hashes when input_names is not None. It should return the input hashes for input files that contain all the specified input names.
        """
        mock_listdir.return_value = ['input1.yaml', 'input2.yaml']
        mock_path_join.return_value = '/path/to/inputs'
        mock_yaml_load.return_value = {'input1': 'value1', 'input2': 'value2'}
        recorder = FileRecorder('/path/to')
        input_hashes = recorder.get_input_hashes('task1', ['input1', 'input2'])
        assert input_hashes == ['input1', 'input2']

    @patch('os.listdir')
    @patch('os.path.join')
    @patch('builtins.open', new_callable=mock_open, read_data='input1: value1\ninput2: value2')
    @patch('yaml.load')
    def test_get_input_hashes_with_missing_input_names(self, mock_yaml_load, mock_open, mock_path_join, mock_listdir):
        """
        Test get_input_hashes when input_names is not None and some input names are missing from the input files. It should return the input hashes for input files that contain all the specified input names.
        """
        mock_listdir.return_value = ['input1.yaml', 'input2.yaml']
        mock_path_join.return_value = '/path/to/inputs'
        mock_yaml_load.return_value = {'input1': 'value1'}
        recorder = FileRecorder('/path/to')
        input_hashes = recorder.get_input_hashes('task1', ['input1', 'input2'])
        assert input_hashes == []