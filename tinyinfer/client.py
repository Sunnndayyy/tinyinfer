from __future__ import annotations

import http.client
import json  
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
    sampling_strategy: str
    temperature: float


@dataclass(frozen=True)
class ChatCompletion:
    text: str
    finish_reason: str
    generated_tokens: int
    time_to_first_token: float
    output_tokens_per_second: float


def _error_message(error: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode("utf-8"))
        message = payload["error"]["message"]
    except (KeyError, TypeError, ValueError):
        message = error.reason
    return f"server returned HTTP {error.code}: {message}"


class TinyInferClient:
    def __init__(self, host: str):
        self.host = host.rstrip("/")
        if "://" not in self.host:
            self.host = f"http://{self.host}"

    def get_server_info(self) -> ServerInfo:
        request = urllib.request.Request(f"{self.host}/health", method="GET")
        try:
            with urllib.request.urlopen(request) as response:
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
                sampling_strategy=str(payload["sampling"]["strategy"]),
                temperature=float(payload["sampling"]["temperature"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ClientError(
                "the TinyInfer health response is missing model metadata"
            ) from error

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

        try:
            with urllib.request.urlopen(request) as response:
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
                        content = delta.get("content", "")
                        if not isinstance(content, str):
                            raise TypeError("invalid stream content")
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
                        raise ClientError("the server sent an invalid chat stream") from error
                    if content:
                        text_parts.append(content)
                        on_text(content)
                    if choice.get("finish_reason") is not None:
                        try:
                            completion = ChatCompletion(
                                text="".join(text_parts),
                                finish_reason=str(choice["finish_reason"]),
                                generated_tokens=int(chunk["usage"]["completion_tokens"]),
                                time_to_first_token=float(
                                    chunk["tinyinfer"]["time_to_first_token_seconds"]
                                ),
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