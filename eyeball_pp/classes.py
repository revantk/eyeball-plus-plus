from enum import IntEnum
from typing import Any, Callable
from dataclasses import dataclass
from typing import Protocol


class FeedbackResult(IntEnum):
    POSITIVE = 1
    NEGATIVE = -1
    NEUTRAL = 0

    def __str__(self) -> str:
        if self == FeedbackResult.POSITIVE:
            return "📈 +ve"
        elif self == FeedbackResult.NEGATIVE:
            return "📉 -ve"
        else:
            return "📊 neutral"


@dataclass
class OutputFeedback:
    result: FeedbackResult
    message: str

    def as_dict(self):
        return {"result": self.result.name, "message": self.message}

    @staticmethod
    def from_dict(data):
        return OutputFeedback(
            result=FeedbackResult[data["result"]], message=data["message"]
        )


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

    def as_dict(self):
        return {"score": self.score, "message": self.message}

    @staticmethod
    def from_dict(data):
        return OutputScore(score=data["score"], message=data["message"])


# Compare the output of two checkpoints for a given objective
# The comparator should return the output of -- How would you rate the newer_checkpoint output compared to older_checkpoint output? Is it positive, negative or neutral?
class OutputComparator(Protocol):
    def __call__(
        self,
        objective: str,
        input_variables: dict[str, str],
        older_checkpoint_output: str,
        newer_checkpoint_output: str,
    ) -> OutputFeedback:
        ...


# The output scorer is used to score the output of a model given the objective and inputs
# The scorer should return a float with a higher value indicating a better output
class OutputScorer(Protocol):
    def __call__(
        self, objective: str, input_variables: dict[str, str], output: str
    ) -> OutputScore:
        ...
