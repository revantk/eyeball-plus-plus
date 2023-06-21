# Eyeball++
eyeball_pp is a framework to evaluate and benchmark tasks which use llms.

This framework helps you answer questions like:
"Which llm is the best for my specific task?" 
"Is my prompt change better for the different scenarios I've been testing?"

Your task can be arbitrary, the framework only cares about inputs and outputs.
If you've been eyeballing how well your changes are working, this framework should fit right in and help you evaluate your task in a more methodical manner. Think of it as eyeball++

# Concepts 
## Record:
eyeball_pp consists of a recorder which records your task runs as you are testing them and saves them as checkpoints. You can record task runs locally while developing them or from a production system. task_eval makes it very easy to do this. You can optionally record human feedback for those tasks too.

## Rerun:
You can then re-run these recorded examples as you make changes or re-run the recorded example with a set of parameters you want to evaluate (eg. model, temperature etc.). Each of these re-runs is saved as a new checkpoint.

## Compare:
eyeball_pp lets you run comparisons across various checkpoints and tells you how your changes are performing. If the output of your task can be evaluated objectively then you can supply a custom comparator and if not you can just use the built in model graded eval. This will use a llm to figure out if your task output is solving the objective you want it to. And if you've been recording human feedback for your task runs, it will use this feedback to figure out what is a good output and what is not.

# Example

