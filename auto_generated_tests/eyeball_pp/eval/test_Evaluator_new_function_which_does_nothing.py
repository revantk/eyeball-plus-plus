import pytest

def test_new_function_which_does_nothing():
    assert isinstance(Evaluator().new_function_which_does_nothing('world'), str)
    assert Evaluator().new_function_which_does_nothing('world') == 'Hello world'
    assert Evaluator().new_function_which_does_nothing('') == 'Hello '
    assert Evaluator().new_function_which_does_nothing('  world  ') == 'Hello   world  '