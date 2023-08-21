from enum import Enum, IntEnum
from typing import Optional
from dataclasses import dataclass, asdict, fields
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


TASK_OUTPUT_KEY = "task_output"


class Criteria(str, Enum):
    """A Criteria to evaluate."""

    CONCISENESS = "conciseness"
    RELEVANCE = "relevance"
    CORRECTNESS = "correctness"
    COHERENCE = "coherence"
    HARMFULNESS = "harmfulness"
    MALICIOUSNESS = "maliciousness"
    HELPFULNESS = "helpfulness"
    CONTROVERSIALITY = "controversiality"
    MISOGYNY = "misogyny"
    CRIMINALITY = "criminality"
    INSENSITIVITY = "insensitivity"
    DEPTH = "depth"
    CREATIVITY = "creativity"
    DETAIL = "detail"


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

    def __str__(self) -> str:
        if self.message:
            return f"{self.result} ({self.message})"
        else:
            return f"{self.result}"


class MultiOutputFeedback(dict[str, OutputFeedback]):
    def as_dict(self):
        return {key: value.as_dict() for key, value in self.items()}

    @staticmethod
    def from_dict(data):
        return MultiOutputFeedback(
            {key: OutputFeedback.from_dict(value) for key, value in data.items()}
        )

    def __str__(self) -> str:
        return "\n".join([f"{key}: {value}" for key, value in self.items()])


@dataclass
class OutputScore:
    score: float
    message: str
    cost: float = 0.0

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, OutputScore) and self.score == __value.score

    def __lt__(self, __value: object) -> bool:
        return isinstance(__value, OutputScore) and self.score < __value.score

    def __gt__(self, __value: object) -> bool:
        return isinstance(__value, OutputScore) and self.score > __value.score

    def __str__(self) -> str:
        if self.message:
            return f"{self.score:.2f} ({self.message})"
        else:
            return f"{self.score: .2f}"

    def as_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data):
        return OutputScore(**data)


class MultiOutputScores(dict[str, OutputScore]):
    def as_dict(self):
        return {key: value.as_dict() for key, value in self.items()}

    @staticmethod
    def from_dict(data):
        return MultiOutputScores(
            {key: OutputScore.from_dict(value) for key, value in data.items()}
        )


# Compare the output of two checkpoints for a given objective
# The comparator should return the output of -- How would you rate the newer_checkpoint output compared to older_checkpoint output? Is it positive, negative or neutral?
class OutputComparator(Protocol):
    def __call__(
        self,
        task_objective: str,
        input_variables: dict[str, str],
        older_checkpoint_output: str,
        newer_checkpoint_output: str,
        objectives_intermediary_state: Optional[dict[str, str]] = None,
        older_checkpoint_intermediary_state: Optional[dict[str, str]] = None,
        newer_checkpoint_intermediary_state: Optional[dict[str, str]] = None,
    ) -> MultiOutputFeedback:
        ...


class OutputGrader(Protocol):
    def __call__(
        self,
        input_variables: dict[str, str],
        output: str,
        intermediary_state: Optional[dict[str, str]] = None,
        criteria: Optional[list[Criteria]] = None,
        custom_criteria: Optional[dict[str, str]] = None,
    ) -> OutputScore:
        ...


@dataclass
class EvaluatorConfig:
    sample_rate: float = 1.0
    dir_path: str = "."
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    record_in_memory: bool = False

    @staticmethod
    def _merge(original_config: "EvaluatorConfig", **kwargs) -> "EvaluatorConfig":
        """Kwargs should be a subset of the fields of the original config."""
        if len(kwargs) == 0:
            return original_config

        new_config = EvaluatorConfig()

        for field in fields(original_config):
            if (field_val := kwargs.get(field.name)) is not None:
                setattr(new_config, field.name, field_val)
            else:
                setattr(new_config, field.name, getattr(original_config, field.name))

        return new_config
