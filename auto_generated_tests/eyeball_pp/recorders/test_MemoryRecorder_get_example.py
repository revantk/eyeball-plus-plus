import pytest

def test_get_example_returns_none_if_task_does_not_exist():
    recorder = MemoryRecorder()
    assert recorder.get_example('nonexistent_task', 'example_id') is None

def test_get_example_returns_none_if_example_does_not_exist():
    recorder = MemoryRecorder()
    task = Task(name='task_name')
    recorder.tasks['task_name'] = task
    assert recorder.get_example('task_name', 'nonexistent_example_id') is None

def test_get_example_returns_none_if_checkpoint_does_not_exist():
    recorder = MemoryRecorder()
    task = Task(name='task_name')
    example = Example(id='example_id', checkpoint_id='checkpoint_id', variables={}, output_variable_names=set(), params={})
    task.records['example_id:checkpoint_id'] = example
    recorder.tasks['task_name'] = task
    assert recorder.get_example('task_name', 'example_id', 'nonexistent_checkpoint_id') is None

def test_get_example_returns_latest_checkpoint_if_checkpoint_id_not_provided():
    recorder = MemoryRecorder()
    task = Task(name='task_name')
    example1 = Example(id='example_id', checkpoint_id='checkpoint_id1', variables={}, output_variable_names=set(), params={})
    example2 = Example(id='example_id', checkpoint_id='checkpoint_id2', variables={}, output_variable_names=set(), params={})
    task.records['example_id:checkpoint_id1'] = example1
    task.records['example_id:checkpoint_id2'] = example2
    task.checkpoints['example_id'] = {'checkpoint_id1', 'checkpoint_id2'}
    recorder.tasks['task_name'] = task
    assert recorder.get_example('task_name', 'example_id') == example2

def test_get_example_returns_specified_checkpoint_if_checkpoint_id_provided():
    recorder = MemoryRecorder()
    task = Task(name='task_name')
    example1 = Example(id='example_id', checkpoint_id='checkpoint_id1', variables={}, output_variable_names=set(), params={})
    example2 = Example(id='example_id', checkpoint_id='checkpoint_id2', variables={}, output_variable_names=set(), params={})
    task.records['example_id:checkpoint_id1'] = example1
    task.records['example_id:checkpoint_id2'] = example2
    task.checkpoints['example_id'] = {'checkpoint_id1', 'checkpoint_id2'}
    recorder.tasks['task_name'] = task
    assert recorder.get_example('task_name', 'example_id', 'checkpoint_id1') == example1

def test_get_example_returns_example_with_feedback():
    recorder = MemoryRecorder()
    task = Task(name='task_name')
    example = Example(id='example_id', checkpoint_id='checkpoint_id', variables={}, output_variable_names=set(), params={}, feedback=ResponseFeedback.POSITIVE, feedback_details='Good job!')
    task.records['example_id:checkpoint_id'] = example
    recorder.tasks['task_name'] = task
    assert recorder.get_example('task_name', 'example_id') == example