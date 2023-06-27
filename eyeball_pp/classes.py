from enum import IntEnum
from typing import Callable
from dataclasses import dataclass


class FeedbackResult(IntEnum):
    POSITIVE = 1
    NEGATIVE = -1
    NEUTRAL = 0


@dataclass
class OutputFeedback:
    result: FeedbackResult
    message: str


@dataclass
class OutputScore:
    score: float
    message: str

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, OutputScore) and self.score == __value.score

    def __lt__(self, __value: object) -> bool:
        return isinstance(__value, OutputScore) and self.score < __value.score

    def __gt__(self, __value: object) -> bool:
        return isinstance(__value, OutputScore) and self.score > __value.score

    def __str__(self) -> str:
        return f"{self.score:.2f} ({self.message})"


# Compare the output of two checkpoints for a given objective
# The comparator should return the output of -- How would you rate checkpoint b compared to checkpoint a? Is it positive, negative or neutral?
# First input is the objective , second is a dictionary of input variables, third is the value for a, fourth is the value for b
OutputComparator = Callable[[str, dict[str, str], str, str], OutputFeedback]

# The output scorer is used to score the output of a model given the objective and inputs
# The scorer should return a float with a higher value indicating a better output
# First input is the objective , second is a dictionary of input variables, third is the value for the output
OutputScorer = Callable[[str, dict[str, str], str], OutputScore]
