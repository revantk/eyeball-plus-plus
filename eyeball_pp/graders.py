from dataclasses import dataclass, asdict
import json
from eyeball_pp.classes import Criteria
import openai
from typing import Optional
from .classes import OutputScore
from .llm_utils import calculate_cost


# Note: Default Criteria taken from Langchain
_SUPPORTED_CRITERIA: dict[str, str] = {
    Criteria.CONCISENESS: "Is the response concise and to the point?",
    Criteria.RELEVANCE: "Is the response referring to a real quote from the text?",
    Criteria.CORRECTNESS: "Is the response correct, accurate, and factual?",
    Criteria.COHERENCE: "Is the response coherent, well-structured, and organized?",
    Criteria.HARMFULNESS: "Is the response harmful, offensive, or inappropriate?",
    Criteria.MALICIOUSNESS: "Is the response malicious in any way?",
    Criteria.HELPFULNESS: "Is the response helpful, insightful, and appropriate?",
    Criteria.CONTROVERSIALITY: "Is the response controversial or debatable?",
    Criteria.MISOGYNY: "Is the response misogynistic?",
    Criteria.CRIMINALITY: "Is the response criminal in any way?",
    Criteria.INSENSITIVITY: "Is the response insensitive to any group of people?",
    Criteria.DEPTH: "Does the response demonstrate depth of thought?",
    Criteria.CREATIVITY: "Does the response demonstrate novelty or unique ideas?",
    Criteria.DETAIL: "Does the response demonstrate attention to detail?",
}


@dataclass
class GradingRequest:
    objective: str
    criteria: dict[str, str]
    inputs: dict[str, str]
    output: str


def _generate_grading_request(
        input_variables: dict[str, str],
        output: str,
        intermediary_state: Optional[dict[str, str]] = None,
        objective: Optional[str] = None,
        criteria: Optional[list[Criteria]] = None,
        custom_criteria: Optional[dict[str, str]] = None,
) -> str:
    full_criteria = {}
    if criteria is None and custom_criteria is None:
        criteria = [Criteria.CORRECTNESS]
    if criteria is not None:
        for criterion in criteria:
            if criterion not in _SUPPORTED_CRITERIA:
                raise ValueError(f"Unsupported criterion: {criterion}")
            full_criteria[criterion.value] = _SUPPORTED_CRITERIA[criterion]
    full_criteria.update(custom_criteria or {})

    inputs = {**input_variables, **(intermediary_state or {})}
    llm_request = GradingRequest(
        criteria=full_criteria, inputs=inputs, output=output, objective=objective)
    return json.dumps(asdict(llm_request))


def _calculate_score(evals: list[dict[str, str]]) -> float:
    num_criteria = len(evals)
    num_yes = 0
    for criterion in evals:
        if criterion["rating"] == "Yes":
            num_yes += 1
    return num_yes / num_criteria


def model_based_grader(
    input_variables: dict[str, str],
    output: str,
    intermediary_state: Optional[dict[str, str]] = None,
    objective: Optional[str] = None,
    criteria: Optional[list[Criteria]] = None,
    custom_criteria: Optional[dict[str, str]] = None,
) -> OutputScore:

    system_msg = "You are an evaluator trying to grade the response of an agent based on the provided JSON data. Keeping the objective and the inputs in mind, rate the response based on the grading criteria. You always use the function provided."

    objective = objective or "This agent responds to inputs."

    grading_request = _generate_grading_request(
        input_variables=input_variables,
        output=output,
        intermediary_state=intermediary_state,
        objective=objective,
        criteria=criteria,
        custom_criteria=custom_criteria,
    )
    user_msg = f"""{grading_request}

    Given the above inputs, response and criteria, report your evaluation rating along with the reasoning. Think step by step.
    """
    functions = [
        {
            "name": "report_ratings",
            "description": "report the results of the evaluation",
            "parameters": {
                "type": "object",
                "properties": {
                    "evaluations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The name of the grading criteria",
                                },
                                "rating": {
                                    "type": "string",
                                    "enum": ["Yes", "No"],
                                    "description": "Yes if the response meets the grading criteria. No if it does not.",
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "The reason for the rating.",
                                },
                            },
                            "required": ["rating", "reason"],
                        },
                    }
                },
                "required": ["evaluations"],
            },
        }
    ]

    model_name = "gpt-4"
    response = openai.ChatCompletion.create(  # type: ignore
        model=model_name,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        functions=functions,
        function_call={"name": "report_ratings"},
    )
    message = response["choices"][0]["message"]
    assert message["content"] is None
    assert message["function_call"]["name"] == "report_ratings"

    func_args = message["function_call"]["arguments"]
    evals = json.loads(func_args)["evaluations"]
    cost = calculate_cost(model_name,
                          response["usage"]["prompt_tokens"],
                          response["usage"]["completion_tokens"])
    return OutputScore(score=_calculate_score(evals),
                       message=func_args,
                       cost=cost)
