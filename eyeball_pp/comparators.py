import openai
from typing import Callable

# Similar to the standard comparator if both are the same return 0, if a is better than b return -1, otherwise return 1
# First input is the objective name, second is the value for a, third is the value for b
# Where a is the older checkpoint and b is the newer checkpoint for this example
OutputComparator = Callable[[str, dict[str, str], str, str], int]


def model_graded_comparator(
    objective: str, inputs: dict[str, str], output_a: str, output_b: str
) -> int:
    input_str = "\n".join([f"{key}: {val}" for key, val in inputs.items()])

    system_msg = f"""
You are a human evaluator trying to grade the response of a function based on the following objective and inputs.

Objective:
{objective}

Inputs: 
{input_str}
"""

    is_b_better = f"""
Keeping the objectives and the inputs in mind, which of the following responses is better?

Response A:
{output_a}

Response B:
{output_b}

Give your reasoning followed by one of the following options:
Yes -- if Response A is better than B
No -- if Response B is better than A
Same -- if both are equally good
"""
    response = openai.ChatCompletion.create(  # type: ignore
        model="gpt-4",
        temperature=0.5,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": is_b_better},
        ],
    )["choices"][0]["message"]["content"]
    response_last_line = response.split("\n")[-1].lower()
    if "yes" in response_last_line:
        return -1
    elif "no" in response_last_line:
        return 1
    else:
        return 0
