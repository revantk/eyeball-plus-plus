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


# Compare the output of two checkpoints for a given objective
# The comparator should return the output of -- How would you rate checkpoint b compared to checkpoint a? Is it positive, negative or neutral?
# First input is the objective name, second is the value for a, third is the value for b
OutputComparator = Callable[[str, dict[str, str], str, str], OutputFeedback]

# The output scorer is used to score the output of a model given the objective and inputs
# The scorer should return a float with a higher value indicating a better output
OutputScorer = Callable[[str, dict[str, str], str], float]
