from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ALLOWED_ROLES = {"system", "user", "assistant"}

# Currently just supports Qwen / ChatML format
# TODO: Abstract to common interface and add support for other formats


@dataclass(frozen=True)
class Message:
    role: str
    content: str


def format_chatml(messages: Iterable[Message]) -> str:
    """Turn role-labelled messages into the exact text Qwen was trained on."""
    parts: list[str] = []
    for message in messages:
        if message.role not in ALLOWED_ROLES:
            raise ValueError(f"unsupported chat role: {message.role}")
        if IM_START in message.content or IM_END in message.content:
            raise ValueError("message content cannot contain a ChatML control token")
        parts.append(f"{IM_START}{message.role}\n{message.content}{IM_END}\n")
    parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


class QwenTokenizer:
    def __init__(self, model_dir: str | Path):
        tokenizer_path = Path(model_dir) / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"missing tokenizer: {tokenizer_path}")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encode_chat(self, messages: Iterable[Message]) -> list[int]:
        return self.encode(format_chatml(messages))

    def incremental_decoder(self) -> IncrementalDecoder:
        return IncrementalDecoder(self._tokenizer)


class IncrementalDecoder:
    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        self._stream = DecodeStream(skip_special_tokens=True)

    def step(self, token_id: int) -> str:
        return self._stream.step(self._tokenizer, token_id) or ""
