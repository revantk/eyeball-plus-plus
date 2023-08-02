from collections import defaultdict
import threading
from typing import Any, Optional, Protocol, Iterator
import openai
import dataclasses
import time
import tiktoken
from enum import Enum
import anthropic
import logging
import os

# Secret keys
openai.api_key = os.environ.get("OPENAI_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = "https://tark.openai.azure.com/"

CODE_TEMPERATURE = 0.2
TEXT_TEMPERATURE = 0.4

# terminal colors
BLUE = "\x1b[34m"
END_CLR = "\x1b[0m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"

logger = logging.getLogger()

PRICING = {
    "gpt-3.5-turbo": (0.002 / 1000.0, 0.002 / 1000.0),
    "gpt-4": (0.03 / 1000.0, 0.06 / 1000.0),
    "claude-2": (11.02 / 1000000.0, 32.68 / 1000000.0),
    "claude-instant-1": (1.63 / 1000000.0, 5.51 / 1000000.0),
}


@dataclasses.dataclass
class TokensUsed:
    input_tokens: int
    output_tokens: int

    def __add__(self, other: "TokensUsed") -> "TokensUsed":
        return TokensUsed(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclasses.dataclass
class LLMResponse:
    output: str
    model_name: str
    num_input_tokens: int
    num_output_tokens: int

    @staticmethod
    def empty() -> "LLMResponse":
        return LLMResponse("", "", 0, 0)

    def __add__(self, other: "LLMResponse") -> "LLMResponse":
        if (
            self.model_name != other.model_name
            and self.model_name != ""
            and other.model_name != ""
        ):
            raise ValueError(
                f"Cannot add LLMResponse objects from different models: "
                f"{self.model_name} and {other.model_name}"
            )
        return LLMResponse(
            self.output + other.output,
            max(self.model_name, other.model_name),
            max(self.num_input_tokens, other.num_input_tokens),
            self.num_output_tokens + other.num_output_tokens,
        )


@dataclasses.dataclass
class ModelDetails:
    name: str
    max_tokens: int
    default_stop_tokens: list[str] = dataclasses.field(default_factory=list)
    temperature: float = 0.4
    additional_kwargs: dict[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.name)
        except KeyError:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def get_tokens_in_prompt(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))
        # else:
        #     return int(float(len(re.split("[ \n\.\t]", prompt))) / 0.7)

    def _get_max_tokens_to_request(self, current_prompt: str) -> int:
        return self.max_tokens - int(self.get_tokens_in_prompt(current_prompt) / 0.9)


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclasses.dataclass
class Message:
    role: Role
    content: str

    def __str__(self) -> str:
        return f"--- {self.role.value}:\n{self.content}"

    def to_json_dict(self) -> dict[str, Any]:
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def from_json_dict(cls, json_dict) -> "Message":
        return Message(role=Role(json_dict["role"]), content=json_dict["content"])


class StreamableModel(Protocol):
    def stream_with_messages(self, messages: list[Message]) -> Iterator[LLMResponse]:
        ...

    def predict_with_messages(self, messages: list[Message]) -> LLMResponse:
        ...

    def get_name(self) -> str:
        ...

    def set_temperature(self, temperature: float) -> None:
        ...


class AnthropicChatModel(StreamableModel):
    def __init__(self, name: str, temperature: float, *models: ModelDetails) -> None:
        self.models = list(models)
        self.models.sort(key=lambda md: md.max_tokens)
        self.temperature = temperature
        self.name = name

    def get_name(self) -> str:
        return self.name

    def _get_model_name(self, prompt: str) -> tuple[str, int]:
        num_input_tokens = anthropic_client.count_tokens(prompt)
        if len(self.models) == 1:
            return self.models[0].name, num_input_tokens
        else:
            for model in self.models:
                if int(num_input_tokens * 1.5) < model.max_tokens:
                    return model.name, num_input_tokens
            return self.models[-1].name, num_input_tokens

    def _create_prompt(self, messages: list[Message]) -> str:
        prompt = ""
        last_role = None
        for message in messages:
            if message.role == last_role:
                prompt += f"\n{message.content}"
            if message.role in (Role.USER, Role.SYSTEM):
                prompt += f"{anthropic.HUMAN_PROMPT} {message.content}"
            if message.role == Role.ASSISTANT:
                prompt += f"{anthropic.AI_PROMPT} {message.content}"
            last_role = Role.USER if message.role == Role.SYSTEM else message.role
        prompt += anthropic.AI_PROMPT
        return prompt

    def _create_completion(self, messages: list[Message], stream: bool) -> Any:
        prompt = "\n".join(
            f"{anthropic.AI_PROMPT} {msg.content}"
            if msg.role == Role.ASSISTANT
            else f"{anthropic.HUMAN_PROMPT} {msg.content}"
            for msg in messages
        )
        return anthropic_client.completions.create(
            prompt=prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=self._get_model_name(prompt)[0],
            max_tokens_to_sample=4000,
            temperature=self.temperature,
        )

    def predict_with_messages(self, messages: list[Message]) -> LLMResponse:
        prompt = self._create_prompt(messages)
        model_name, num_input_tokens = self._get_model_name(prompt)
        response = anthropic_client.completions.create(
            prompt=prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=model_name,
            max_tokens_to_sample=4000,
            temperature=self.temperature,
        )
        completion = response.completion
        num_output_tokens = anthropic_client.count_tokens(completion)
        return LLMResponse(
            output=completion,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            model_name=model_name,
        )

    def stream_with_messages(self, messages: list[Message]) -> Iterator[LLMResponse]:
        prompt = self._create_prompt(messages)
        model_name, num_input_tokens = self._get_model_name(prompt)
        response = anthropic_client.completions.create(
            prompt=prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=model_name,
            max_tokens_to_sample=4000,
            temperature=self.temperature,
            stream=True,
        )
        last_completion = ""
        for event in response:
            completion = event.completion
            delta = completion[len(last_completion) :]
            num_output_tokens = anthropic_client.count_tokens(delta)
            yield LLMResponse(
                output=delta,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                model_name=model_name,
            )
            last_completion = completion

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature


class OpenaiChatModel(StreamableModel):
    def __init__(self, name: str, temperature: float, *models: ModelDetails) -> None:
        self.models: list[ModelDetails] = list(models)
        self.models.sort(key=lambda md: md.max_tokens)
        self.temperature = temperature
        self.name = name

    def get_name(self) -> str:
        return self.name

    def _count_tokens(self, model_name: str, prompt: str) -> int:
        return len(tiktoken.encoding_for_model(model_name).encode_ordinary(prompt))

    def _get_model_to_use(self, messages: list[Message]) -> tuple[ModelDetails, int]:
        num_input_tokens = 0
        for model in self.models:
            num_input_tokens = self._count_tokens(
                model.name, "\n".join(msg.content for msg in messages)
            )
            if int(num_input_tokens * 1.5) < model.max_tokens:
                return model, num_input_tokens
        return self.models[-1], num_input_tokens

    def _create_completion(
        self, model: ModelDetails, messages: list[Message], stream: bool
    ) -> Any:
        return openai.ChatCompletion.create(  # type: ignore
            model=model.name,
            messages=[
                {"role": message.role.value, "content": message.content}
                for message in messages
            ],
            temperature=self.temperature,
            stream=stream,
            **model.additional_kwargs,
        )

    def predict_with_messages(self, messages: list[Message]) -> LLMResponse:
        model_to_use, num_input_tokens = self._get_model_to_use(messages)
        output = self._create_completion(model_to_use, messages, stream=False)[
            "choices"
        ][0]["message"]["content"]
        num_output_tokens = self._count_tokens(model_to_use.name, output)
        return LLMResponse(
            output=output,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            model_name=model_to_use.name,
        )

    def stream_with_messages(self, messages: list[Message]) -> Iterator[LLMResponse]:
        model_to_use, num_input_tokens = self._get_model_to_use(messages)
        response = self._create_completion(model_to_use, messages, stream=True)
        for event in response:
            delta = event["choices"][0]["delta"].get("content", "")
            num_output_tokens = self._count_tokens(model_to_use.name, delta)
            yield LLMResponse(
                output=delta,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                model_name=model_to_use.name,
            )

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature


# Base model details
TURBO = ModelDetails(
    "gpt-3.5-turbo",
    4000,
    additional_kwargs={
        "engine": "gpt-35-turbo",
        "api_key": AZURE_OPENAI_KEY,
        "api_base": AZURE_OPENAI_ENDPOINT,
        "api_type": "azure",
        "api_version": "2023-03-15-preview",
    },
)
TURBO_16k = ModelDetails("gpt-3.5-turbo-16k", 16000)
GPT4 = ModelDetails("gpt-4", 8000)
CLAUDE = ModelDetails("claude-2", 100000)
CLAUDE_INSTANT = ModelDetails("claude-instant-1", 100000)

# Anthropic models
ANTHROPIC_TEXT = AnthropicChatModel("anthropic_text", TEXT_TEMPERATURE, CLAUDE_INSTANT)
ANTHROPIC_CODE = AnthropicChatModel("anthropic_code", CODE_TEMPERATURE, CLAUDE)

# Openai models
OPENAI_TEXT_MODEL = OpenaiChatModel("openai_text", TEXT_TEMPERATURE, TURBO)
OPENAI_CODE_MODEL = OpenaiChatModel(
    "openai_code", CODE_TEMPERATURE, TURBO, TURBO_16k, GPT4
)
OPENAI_COMPLICATED_CODE_MODEL = OpenaiChatModel(
    "openai_complicated_code", CODE_TEMPERATURE, GPT4
)

# Default models
TEXT_MODEL = OPENAI_TEXT_MODEL
CODE_MODEL = OPENAI_CODE_MODEL
COMPLICATED_CODE_MODEL = OPENAI_COMPLICATED_CODE_MODEL


def get_streamable_model_from_name(name: str) -> StreamableModel:
    if name == "anthropic_text":
        return ANTHROPIC_TEXT
    elif name == "anthropic_code":
        return ANTHROPIC_CODE
    elif name == "openai_text":
        return OPENAI_TEXT_MODEL
    elif name == "openai_code":
        return OPENAI_CODE_MODEL
    elif name == "openai_complicated_code":
        return OPENAI_COMPLICATED_CODE_MODEL
    else:
        raise ValueError(f"Unknown model name: {name}")


def predict_with_messages(
    messages: list[Message],
    model: StreamableModel,
    print_output: bool = False,
    print_input: Optional[bool] = None,
    retries: int = 3,
    stream: bool = False,
    kill_signal: Optional[threading.Event] = None,
) -> LLMResponse:
    if print_input is None:
        print_input = print_output

    if print_input:
        to_log = "\n".join(str(message) for message in messages)
        logger.info(f"LLM input:\n{to_log}")

    llm_response = LLMResponse.empty()

    start_time = time.time()
    if stream:
        try_num = 0
        while try_num < retries:
            try:
                for delta in model.stream_with_messages(messages):
                    if kill_signal is not None and kill_signal.is_set():
                        return LLMResponse.empty()
                    llm_response += delta
                break
            except Exception as e:
                if print_output:
                    logger.error(f"{RED}{e}\nRetrying...{END_CLR}")
                time.sleep(2 ^ try_num)
                try_num += 1
                llm_response = LLMResponse.empty()
    else:
        try_num = 0
        while try_num < retries:
            try:
                llm_response = model.predict_with_messages(messages)
                break
            except Exception as e:
                if print_output:
                    logger.error(f"{RED}{e}\nRetrying...{END_CLR}")
                time.sleep(2 ^ try_num)
                try_num += 1

    if print_output:
        logger.info(f"LLM output:\n{GREEN}{llm_response.output}{END_CLR}")

    # Log llm stats
    time_taken = time.time() - start_time
    cost_msg = ""
    if pricing_model := PRICING.get(llm_response.model_name):
        cost = (
            pricing_model[0] * llm_response.num_input_tokens
            + pricing_model[1] * llm_response.num_output_tokens
        )
        cost_msg += f"cost: ${cost:.3f}, "

    logger.info(
        f"{BLUE}{llm_response.model_name} time taken:{time_taken:.2f}s, {cost_msg}input_tokens:{llm_response.num_input_tokens}, output_tokens:{llm_response.num_output_tokens}, retries:{try_num} {END_CLR}"
    )

    return llm_response


def predict(
    prompt: str,
    model: StreamableModel,
    print_progress: bool = False,
    print_output: bool = False,
    extra_strip_tokens: str = "",
    retries: int = 3,
) -> str:
    llm_response = predict_with_messages(
        [Message(Role.USER, prompt)],
        model,
        print_output=print_output,
        print_input=False,
        stream=print_progress,
        retries=retries,
    )

    return llm_response.output.strip(' \n"' + extra_strip_tokens)


class ChatThread:
    def __init__(
        self,
        system_message: Optional[str] = None,
        print_output: bool = False,
        messages: Optional[list[Message]] = None,
    ) -> None:
        self.messages: list[Message] = []
        self.print_output = print_output
        self.system_msg = system_message
        self.kill_signal = threading.Event()
        self.last_model_used: Optional[str] = None

        # input and output tokens per model name
        self.tokens_used: dict[str, TokensUsed] = defaultdict(lambda: TokensUsed(0, 0))

        if messages is not None:
            self.messages.extend(messages)
        elif system_message is not None:
            msg = Message(Role.SYSTEM, system_message)
            self.messages.append(msg)

    @staticmethod
    def clone(chat_thread: "ChatThread", clear_messages: bool) -> "ChatThread":
        return ChatThread(
            system_message=chat_thread.system_msg,
            print_output=chat_thread.print_output,
            messages=None if clear_messages else list(chat_thread.messages),
        )

    def add_agent_message(self, message: str) -> None:
        msg = Message(Role.ASSISTANT, message)
        self.messages.append(msg)

    def add_user_message(self, message: str) -> None:
        msg = Message(Role.USER, message)
        self.messages.append(msg)

    def ask(
        self,
        prompt: str,
        model: StreamableModel,
        log_prefix: Optional[str] = None,
    ) -> str:
        user_msg = Message(Role.USER, prompt)
        self.messages.append(user_msg)

        if log_prefix is not None:
            logger.info(
                f"{log_prefix} -- Initiating llm request with model {model.get_name()}"
            )

        response = predict_with_messages(
            self.messages,
            model,
            print_input=self.print_output,
            stream=False,
            print_output=self.print_output,
            kill_signal=self.kill_signal,
        )
        self.tokens_used[response.model_name] += TokensUsed(
            response.num_input_tokens, response.num_output_tokens
        )

        self.last_model_used = model.get_name()
        self.messages.append(Message(Role.ASSISTANT, response.output))
        return response.output
