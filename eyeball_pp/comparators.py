from .classes import FeedbackResult, OutputFeedback, OutputScore
import openai


def output_feedback_from_scores(
    older_score: OutputScore, newer_score: OutputScore
) -> OutputFeedback:
    if older_score == newer_score:
        return OutputFeedback(
            FeedbackResult.NEUTRAL,
            f"Score {older_score} is equal to {newer_score}",
        )
    elif older_score > newer_score:
        return OutputFeedback(
            FeedbackResult.NEGATIVE,
            f"Score {newer_score} is worse than {newer_score}",
        )
    else:
        return OutputFeedback(
            FeedbackResult.POSITIVE,
            f"Score {newer_score} is better than {older_score}",
        )


def model_graded_comparator(
    objective: str,
    input_variables: dict[str, str],
    older_checkpoint_output: str,
    newer_checkpoint_output: str,
) -> OutputFeedback:
    input_str = "\n".join([f"{key}: {val}" for key, val in input_variables.items()])

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
{older_checkpoint_output}

Response B:
{newer_checkpoint_output}

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
