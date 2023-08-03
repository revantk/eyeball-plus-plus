from dataclasses import dataclass, asdict
import json
import openai
from typing import Optional, Any
from .classes import (
    FeedbackResult,
    MultiOutputFeedback,
    OutputFeedback,
    OutputScore,
    TASK_OUTPUT_KEY,
)


def output_feedback_from_scores(
    older_scores: dict[str, OutputScore], newer_scores: dict[str, OutputScore]
) -> dict[str, OutputFeedback]:
    feedback = {}
    for output_name, older_score in older_scores.items():
        newer_score = newer_scores.get(output_name)
        if newer_scores is None:
            continue

        if older_scores == newer_score:
            feedback[output_name] = OutputFeedback(
                FeedbackResult.NEUTRAL,
                f"Score {older_scores} is equal to {newer_scores}",
            )
        elif older_score > newer_score:
            feedback[output_name] = OutputFeedback(
                FeedbackResult.NEGATIVE,
                f"Score {newer_scores} is worse than {newer_scores}",
            )
        else:
            feedback[output_name] = OutputFeedback(
                FeedbackResult.POSITIVE,
                f"Score {newer_scores} is better than {older_scores}",
            )
    return feedback


@dataclass
class LLMRequest:
    objective: str
    inputs: dict[str, str]
    responses: list[dict[str, str]]


def _execute_comparator(
    objective: str,
    input_variables: dict[str, str],
    older_checkpoint_response: str,
    newer_checkpoint_response: str,
) -> OutputFeedback:
    system_msg = "You are an evaluator trying to grade the response of two agents based on provided JSON data. Keeping the objectives and the inputs in mind, decide which response is better and provide a reason. You always use the function provided."
    responses = [
        {
            "name": "Andrew",
            "response": older_checkpoint_response,
        },
        {
            "name": "Barry",
            "response": newer_checkpoint_response,
        },
    ]
    llm_request = LLMRequest(
        objective=objective, inputs=input_variables, responses=responses
    )
    user_msg = f"""{json.dumps(asdict(llm_request))}

    Given the above objective, inputs and responses, report your decision on the best response along with the reasoning. Think step by step.
    """
    functions = [
        {
            "name": "report_decision",
            "description": "report the results of the evaluation",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["Andrew", "Barry", "Same"],
                        "description": "The name of the agent with the best response. 'Same' if both are equally good.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "The reason for the decision.",
                    },
                },
                "required": ["decision", "reason"],
            },
        }
    ]

    response = openai.ChatCompletion.create(  # type: ignore
        model="gpt-4",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        functions=functions,
        function_call={"name": "report_decision"},
    )["choices"][0]["message"]

    assert response["content"] is None
    assert response["function_call"]["name"] == "report_decision"

    decision = json.loads(response["function_call"]["arguments"])["decision"]
    reason = json.loads(response["function_call"]["arguments"])["reason"]
    if "Andrew" in decision:
        return OutputFeedback(FeedbackResult.NEGATIVE, reason)
    elif "Barry" in decision:
        return OutputFeedback(FeedbackResult.POSITIVE, reason)
    else:
        return OutputFeedback(FeedbackResult.NEUTRAL, reason)


def model_graded_comparator(
    task_objective: str,
    input_variables: dict[str, str],
    older_checkpoint_output: str,
    newer_checkpoint_output: str,
    objectives_intermediary_state: Optional[dict[str, str]] = None,
    older_checkpoint_intermediary_state: Optional[dict[str, str]] = None,
    newer_checkpoint_intermediary_state: Optional[dict[str, str]] = None,
) -> MultiOutputFeedback:
    feedback = {}

    if older_checkpoint_intermediary_state and newer_checkpoint_intermediary_state:
        for key, value in older_checkpoint_intermediary_state.items():
            older_int_state = value
            newer_int_state = newer_checkpoint_intermediary_state[key]
            if objectives_intermediary_state and key in objectives_intermediary_state:
                feedback[key] = _execute_comparator(
                    objective=objectives_intermediary_state[key],
                    input_variables=input_variables,
                    older_checkpoint_response=older_int_state,
                    newer_checkpoint_response=newer_int_state,
                )
            else:
                print(f"Warning: No objective found for {key} in intermediary state")

    feedback[TASK_OUTPUT_KEY] = _execute_comparator(
        objective=task_objective,
        input_variables=input_variables,
        older_checkpoint_response=older_checkpoint_output,
        newer_checkpoint_response=newer_checkpoint_output,
    )

    return MultiOutputFeedback(feedback)
