from dataclasses import dataclass, asdict
import json
import openai
from typing import Optional, Any
from .classes import FeedbackResult, OutputFeedback, OutputScore, OUTPUT_KEY


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
        }
    ]
    llm_request = LLMRequest(
        objective=objective, inputs=input_variables, responses=responses
    )
    user_msg = f"""{json.dumps(asdict(llm_request))}

    Given the above objective, inputs and responses, report your decision on the best response along with the reasoning. Think step by step.
    """
    functions = [{
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
        }
    }]

    response = openai.ChatCompletion.create(  # type: ignore
        model="gpt-4",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        functions=functions,
        function_call={"name": "report_decision"}
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
    input_variables: dict[str, str],
    older_checkpoint_output: str,
    newer_checkpoint_output: str,
    objective_output: str,
    objectives_intermediary_state: Optional[dict[str, str]] = None,
    older_checkpoint_intermediary_state: Optional[dict[str, str]] = None,
    newer_checkpoint_intermediary_state: Optional[dict[str, str]] = None,
) -> dict[str, OutputFeedback]:
    feedback = {}

    if older_checkpoint_intermediary_state and newer_checkpoint_intermediary_state:
        for key, value in older_checkpoint_intermediary_state.items():
            older_int_state = value
            newer_int_state = newer_checkpoint_intermediary_state[key]
            feedback[key] = _execute_comparator(
                objective=objectives_intermediary_state[key],
                input_variables=input_variables,
                older_checkpoint_response=older_int_state,
                newer_checkpoint_response=newer_int_state)

    feedback[OUTPUT_KEY] = _execute_comparator(
        objective=objective_output,
        input_variables=input_variables,
        older_checkpoint_response=older_checkpoint_output,
        newer_checkpoint_response=newer_checkpoint_output)

    return feedback
