# Eyeball++ 
_Ridiculously simple evaluation for LLM agents_

eyeball_pp is a framework to evaluate and benchmark tasks which use llms.

This framework helps you answer questions like: "Which llm is the best for my specific task?" or "This prompt change looks good on the example I just tested, does it work with all the other things I've tested?"

Your task can be arbitrary, the framework only cares about initial inputs and final outputs.

If you've been eyeballing how well your changes are working, this framework should fit right in and help you evaluate your task in a more methodical manner.

# Installation
eyeball_pp is a python library which can be installed via pip 

`pip install eyeball_pp`

# Concepts 
eyeball_pp has 3 simple concepts -- record, rerun and compare. To see a detailed example check out the examples/ folder in the repo

## Record
eyeball_pp consists of a recorder which records the inputs and outputs of your task runs as you are running it and saves them as checkpoints. You can record this locally while developing or from a production system. You can optionally record human feedback for the task output too.

You can record your task using the `record_task` decorator. This will record every run of this function call as a `Checkpoint` for future comparison. The args_to_record specify which inputs to record and the function return value is saved as the output.
```python
import eyeball_pp

@eyeball_pp.record_task(args_to_record=['input_a', 'input_b'])
def your_task_function(input_a, input_b):
  # Your task can run arbitrary code
  ...
  return task_output
```
If you want to record additional inputs within your function call you can just call `eyeball_pp.record_input('variable_name', value)` inside your function.

If your sytem is more complicated, you can also use the `record_input` and `record_output` functions with the start_recording_session context manager
```python
import eyeball_pp

# your task code 
...
with eyeball_pp.start_recording_session(task_name="your_task", checkpoint_id="some_custom_unique_id"):
  eyeball_pp.record_input('input_a', input_a_value)
  eyeball_pp.record_input('input_b', input_b_value)
  ...
  eyeball_pp.record_output(output)
```

OR without the context manager.. 

```python
eyeball_pp.record_input(task_name="your_task", checkpoint_id="some_custom_unique_id", variable_name="input_a", value=input_a_value)
..
eyeball_pp.record_output(task_name="your_task", checkpoint_id="some_custom_unique_id", variable_name='output', output=output)
```

Data by default gets recorded as yaml files in your repo so you can always inspect it, change it or delete it.

## Rerun
You can then re-run these pre-recorded examples as you make changes. This will only re-run each unique set of input variables. eg. if you recorded a run of `your_task_function(1, 2)` 5 times and `your_task_function(3, 4)` 2 times, the re-run would only run 2 examples -- (1, 2) and (3, 4). Each of these re-runs is saved as a new checkpoint for comparison later. 
```python
from eyeball_pp import rerun_recorded_examples 

for input_vars in rerun_recorded_examples():
  your_task_function(input_vars['input_a'], input_vars['input_b'])
```

You can also re-run the examples with a set of parameters you want to evaluate (eg. model, temperature etc.) -- These params will show up in the comparison later and help you decide which params result in the best performance on your task.
```python
from eyeball_pp import get_eval_param, rerun_recorded_examples 

for vars in rerun_recorded_examples({'model': 'gpt-4', 'temperature': 0.7}, {'model': 'mpt-30b-chat'}):
  your_task_function(vars['input_a'], vars['input_b'])

# You can access these eval params from anywhere in your code
@eyeball_pp.record_task(args_to_record=['input_a', 'input_b'])
def your_task_function(input_a, input_b):
  ...
  model = get_eval_param('model') or 'gpt-3.5-turbo'
  temperature = get_eval_param('temperature') or 0.5
```

## Compare
eyeball_pp lets you run comparisons across various checkpoints and tells you how your changes are performing. If the output of your task can be evaluated objectively then you can supply a custom comparator and if not you can just use the built in model graded eval. This will use a model to figure out if your task output is solving the objective you want it to. And if you've been recording human feedback for your task runs, it will use this feedback to fine-tune the evaluator llm.

```python
# The example below uses the built in model graded eval
eyeball_pp.compare_recorded_checkpoints(task_objective="The task should answer questions based on the context provided and also show the sources")
```

Example output of the above command would be something like
```
Comparing last 3 checkpoints of Example(input_a=1, input_b=2)
[improvement] `your_task_function.return_val` got better in checkpoint 2023-06-21T20:29:17.909680 (model=gpt-4, temperature=0.7) vs the older checkpoint 2023-06-21T20:29:16.189539 (model=None, temperature=None)
[neutral] `your_task_function.return_val` is the same between checkpoints 2023-06-21T20:29:17.237979 (model=claude-v1, temperature=None) and 2023-06-21T20:29:19.286249 (model=gpt-4, temperature=0.7)

Comparing last 3 checkpoints of Example(input_a=3, input_b=4)
[improvement] `your_task_function.return_val` got better in checkpoint 2023-06-21T20:29:17.909680 (model=gpt-4, temperature=0.7) vs an older checkpoint 2023-06-21T20:29:16.189539 (model=None, temperature=None)
[neutral] `your_task_function.return_val` is the same between checkpoints 2023-06-21T20:29:17.237979 (model=claude-v1, temperature=None) and 2023-06-21T20:29:19.286249 (model=gpt-4, temperature=0.7)

Summary:
2/2 examples got better in their most recent runs
The param combination (model=gpt-4, temperature=0.7) works better for 2/2 examples than the default params
The param combination (model=claude-v1, temperature=None) works equally as good as the (model=gpt-4, temperature=0.7) combination for 2/2 examples
```

The comparison will also output a benchmark.md file in your repo with the comparison results in a tabular format.

# Configuration 

## Serialization
For the `record_task` decorator you need to ensure that the inputs to the function and outputs are json serialable. If the variables are custom classes you can define the `to_json` and `from_json` functions on that object. If you want to skip serializing some inputs you can specify that in the decorator as `args_to_skip` 
eg. 
```python
@eyeball_pp.record_task(args_to_skip=["input_a"])
def your_task_function(input_a, input_b: SomeComplexType) -> str:
  ...
  return task_output

class SomeComplexType:
  def to_json(self) -> str:
    ...

  def from_json(json_str: str) -> 'SomeComplexType':
    ...  
```

## Sample rate 
You can set the sample rate for recording, by default it's 1.0
```python
# Set separate config for dev and production 
if is_dev_build():
  eyeball_pp.set_config(sample_rate=1.0, dir_path="/path/to/store/recorded/examples")
else:
  # More details on the api_key integration below
  eyeball_pp.set_config(sample_rate=0.1, api_key=<eyeball_pp_apikey>)
```

## Custom example_id
By default a unique example_id is created based on the values of the input variables, but that might not always work.

```python
@eyeball_pp.record_task(example_id_arg_name='request_id', args_to_skip=['request_id'])
def your_task_function(request_id, input_a, input_b):
  ...
  return task_output
```

## Refining the evaluator with Human feedback
If you record human feedback from your production system you can always log that via the library
```python
import eyeball_pp
from eyeball_pp import ResponseFeedback

eyeball_pp.record_human_feedback(example_id, response_feedback=ResponseFeedback.POSITIVE, feedback_details="I liked the response as it selected the sources correctly")
#TODO handle the case where example_id is not known to the user 
```

But you can also rate examples yourself 

```python
eyeball_pp.rate_examples()
```
This will let you compare and rate examples yourself via the command line
Example output of this command would look like:

```
Consider Example(input_a=1, input_b=2):
Does the output `4` fulfil the objecive of the task? (Y/n)

Does the output `3` fulfil the objecive of the task? (Y/n)

Which response is better? 
A) `3` is better than `4`
B) `4` is better than `3`
C) Both are equivalent
```

## Api key 
eyeball_pp works out of the box locally but if you want to record information from production you can get an apikey from https://api.tark.ai

