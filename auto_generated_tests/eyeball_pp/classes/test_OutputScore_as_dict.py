import pytest
from eyeball_pp.classes import OutputScore

def test_as_dict_basic():
    """
    Test that the as_dict function returns a dictionary with the correct keys and values.
    """
    output_score = OutputScore(score=0.8, message='Test message')
    result = output_score.as_dict()
    assert isinstance(result, dict)
    assert result == {'score': 0.8, 'message': 'Test message'}

def test_as_dict_with_different_values():
    """
    Test that the as_dict function works correctly with different values for score and message.
    """
    output_score = OutputScore(score=0.5, message='Another test message')
    result = output_score.as_dict()
    assert result == {'score': 0.5, 'message': 'Another test message'}

def test_as_dict_with_edge_cases():
    """
    Test that the as_dict function works correctly with edge cases, such as a score of 0 or 1 and an empty message.
    """
    output_score = OutputScore(score=0, message='')
    result = output_score.as_dict()
    assert result == {'score': 0, 'message': ''}
    output_score = OutputScore(score=1, message=' ')
    result = output_score.as_dict()
    assert result == {'score': 1, 'message': ' '}