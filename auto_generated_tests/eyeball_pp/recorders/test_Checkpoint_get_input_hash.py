import pytest
from eyeball_pp.recorders import Checkpoint


def test_get_input_hash_with_same_input_variables():
    """

    Test that get_input_hash returns the same hash for two checkpoints with the same input variables

    """
    checkpoint1 = Checkpoint(
        checkpoint_id="1",
        variables={"a": "1", "b": "2", "c": "3"},
        output_variable_names={"c"},
        eval_params={},
    )
    checkpoint2 = Checkpoint(
        checkpoint_id="2",
        variables={"a": "1", "b": "2", "c": "3"},
        output_variable_names={"c"},
        eval_params={},
    )
    assert checkpoint1.get_input_hash() == checkpoint2.get_input_hash()


def test_get_input_hash_with_different_input_variables():
    """

    Test that get_input_hash returns different hashes for two checkpoints with different input variables

    """
    checkpoint1 = Checkpoint(
        checkpoint_id="1",
        variables={"a": "1", "b": "2", "c": "3"},
        output_variable_names={"c"},
        eval_params={},
    )
    checkpoint2 = Checkpoint(
        checkpoint_id="2",
        variables={"a": "1", "b": "3", "c": "3"},
        output_variable_names={"c"},
        eval_params={},
    )
    assert checkpoint1.get_input_hash() != checkpoint2.get_input_hash()


def test_get_input_hash_with_different_output_variables():
    """

    Test that get_input_hash returns the same hash for two checkpoints with different output variables but same input variables

    """
    checkpoint1 = Checkpoint(
        checkpoint_id="1",
        variables={"a": "1", "b": "2", "c": "3", "d": "4"},
        output_variable_names={"c"},
        eval_params={},
    )
    checkpoint2 = Checkpoint(
        checkpoint_id="2",
        variables={"a": "1", "b": "2", "c": "4", "d": "4"},
        output_variable_names={"c"},
        eval_params={},
    )
    assert checkpoint1.get_input_hash() == checkpoint2.get_input_hash()
