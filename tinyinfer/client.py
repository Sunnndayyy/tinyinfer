from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass


class ClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class ServerInfo:
    runtime: str
    model: str
    architecture: str
    layers: int | None
    context_length: int | None
    device: str
    dtype: str
    kv_cache: str
    sampling_strategy: str
    temperature: float
    decoding: str = "unknown"
    attention: str = "unknown"
    quantization: str = "unknown"
    thinking: bool | None = None


@dataclass(frozen=True)
class ChatCompletion:
    text: str
    finish_reason: str
    generated_tokens: int
    client_ttft: float
    server_ttft: float
    output_tokens_per_second: float

    @property
    def time_to_first_token(self) -> float:
        """Deprecated alias for the server-only TTFT."""
        return self.server_ttft


def _error_message(error: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode("utf-8"))
        message = payload["error"]["message"]
    except (KeyError, TypeError, ValueError):
        message = error.reason
    return f"server returned HTTP {error.code}: {message}"


class TinyInferClient:
    def __init__(self, host: str, *, timeout: float = 30.0):
        self.host = host.rstrip("/")
        if "://" not in self.host:
            self.host = f"http://{self.host}"
        self.timeout = timeout

    def get_server_info(self) -> ServerInfo:
        request = urllib.request.Request(f"{self.host}/health", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            raise ClientError(_error_message(error)) from error
        except urllib.error.URLError as error:
            raise ClientError(
                f"could not reach TinyInfer at {self.host}: {error.reason}"
            ) from error
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ClientError("the TinyInfer health response was not valid") from error

        try:
            return ServerInfo(
                runtime=str(payload["runtime"]),
                model=str(payload["model"]),
                architecture=str(payload["architecture"]),
                layers=_optional_int(payload.get("layers")),
                context_length=_optional_int(payload.get("context_length")),
                device=str(payload.get("device", "unknown")),
                dtype=str(payload.get("dtype", "unknown")),
                quantization=str(payload.get("quantization", "unknown")),
                attention=str(payload.get("attention", "unknown")),
                kv_cache=str(payload.get("kv_cache", "unknown")),
                decoding=str(payload.get("decoding", "unknown")),
                sampling_strategy=str(payload["sampling"]["strategy"]),
                temperature=float(payload["sampling"]["temperature"]),
                thinking=_optional_bool(payload.get("thinking")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ClientError("the TinyInfer health response is missing model metadata") from error

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        on_text: Callable[[str], None],
    ) -> ChatCompletion:
        payload = json.dumps(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": True,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        text_parts: list[str] = []
        completion: ChatCompletion | None = None
        request_started_at = time.perf_counter()
        first_token_at: float | None = None

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(line[6:])
                        choice = chunk["choices"][0]
                        delta = choice["delta"]
                        if not isinstance(choice, dict) or not isinstance(delta, dict):
                            raise TypeError("invalid stream choice")
                        has_content = "content" in delta
                        content = delta.get("content", "")
                        if not isinstance(content, str):
                            raise TypeError("invalid stream content")
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
                        raise ClientError("the server sent an invalid chat stream") from error
                    if has_content and first_token_at is None:
                        first_token_at = time.perf_counter()
                    if content:
                        text_parts.append(content)
                        on_text(content)
                    if choice.get("finish_reason") is not None:
                        try:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            client_ttft = first_token_at - request_started_at
                            timings = chunk["tinyinfer"]
                            if "server_ttft_seconds" in timings:
                                server_ttft = timings["server_ttft_seconds"]
                            else:
                                server_ttft = timings["time_to_first_token_seconds"]
                            completion = ChatCompletion(
                                text="".join(text_parts),
                                finish_reason=str(choice["finish_reason"]),
                                generated_tokens=int(chunk["usage"]["completion_tokens"]),
                                client_ttft=client_ttft,
                                server_ttft=float(server_ttft),
                                output_tokens_per_second=float(
                                    chunk["tinyinfer"]["output_tokens_per_second"]
                                ),
                            )
                        except (KeyError, TypeError, ValueError) as error:
                            raise ClientError(
                                "the server chat stream is missing completion metrics"
                            ) from error
        except urllib.error.HTTPError as error:
            raise ClientError(_error_message(error)) from error
        except urllib.error.URLError as error:
            raise ClientError(
                f"could not reach TinyInfer at {self.host}: {error.reason}"
            ) from error
        except (OSError, http.client.HTTPException, UnicodeError) as error:
            raise ClientError("the server chat stream was interrupted") from error

        if completion is None:
            raise ClientError("the server chat stream ended before completion")
        return completion


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError("expected a boolean")
    return value
