from .classes import FeedbackResult, OutputFeedback, OutputScore
import openai


def comparator_from_scores(
    score_a: OutputScore, score_b: OutputScore
) -> OutputFeedback:
    if score_a == score_b:
        return OutputFeedback(
            FeedbackResult.NEUTRAL,
            f"Score {score_a} is equal to {score_b}",
        )
    elif score_a > score_b:
        return OutputFeedback(
            FeedbackResult.NEGATIVE,
            f"Score {score_b} is worse than {score_b}",
        )
    else:
        return OutputFeedback(
            FeedbackResult.POSITIVE,
            f"Score {score_b} is better than {score_a}",
        )


def model_graded_comparator(
    objective: str, inputs: dict[str, str], output_a: str, output_b: str
) -> OutputFeedback:
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
        return OutputFeedback(FeedbackResult.NEGATIVE, response)
    elif "no" in response_last_line:
        return OutputFeedback(FeedbackResult.POSITIVE, response)
    else:
        return OutputFeedback(FeedbackResult.NEUTRAL, response)
