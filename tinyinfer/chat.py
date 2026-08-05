from __future__ import annotations

import http.client
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import TextIO

TINYINFER_LOGO = """
\x1b[0;97;40m▐\x1b[0;97;47m▓▓▓▓▓\x1b[0;97;40m▌▐\x1b[0;97;47m▓▓▓\x1b[0;97;40m▐\x1b[0;97;47m▓\x1b[0;97;40m▄    \x1b[0;97;47m▓▓▓\x1b[0;97;40m▐\x1b[0;97;47m▓\x1b[0;97;40m▄  ▄\x1b[0;97;47m▓▓\x1b[0;97;40m▐\x1b[0;97;47m▓▓▓\x1b[0;97;40m▐\x1b[0;97;47m▓\x1b[0;97;40m▄    \x1b[0;97;47m▓▓▓\x1b[0;97;40m▐\x1b[0;97;47m▓▓▓▓▓\x1b[0;97;40m▐\x1b[0;97;47m▓▓▓▓▓\x1b[0;97;40m▌▐\x1b[0;97;47m▓▓▓▓▓\x1b[0;97;40m▄\x1b[0;37;40m  \x1b[0m
\x1b[0;97;40m  ▐\x1b[0;97;47m▒\x1b[0;37;40m▌  \x1b[0;97;40m▐\x1b[0;97;47m▒▒▒\x1b[0;37;40m▐\x1b[0;97;47m▒▒▒\x1b[0;97;40m▄  \x1b[0;97;47m▒▒▒\x1b[0;97;40m ▀\x1b[0;97;47m▒▒▒▒\x1b[0;97;40m▀\x1b[0;37;40m \x1b[0;97;40m▐\x1b[0;97;47m▒▒▒\x1b[0;37;40m▐\x1b[0;97;47m▒▒▒\x1b[0;97;40m▄  \x1b[0;97;47m▒▒▒\x1b[0;97;40m▐\x1b[0;97;47m▒▒▒\x1b[0;37;40m  \x1b[0;97;40m▐\x1b[0;97;47m▒▒▒\x1b[0;37;40m   ▐\x1b[0;97;47m▒▒\x1b[0;97;40m  ▀\x1b[0;97;47m▒▒\x1b[0;97;40m▄\x1b[0m
\x1b[0;97;40m \x1b[0;90;40m \x1b[0;37;40m▐\x1b[0;97;47m░\x1b[0;37;40m▌  ▐\x1b[0;97;47m░░░\x1b[0;37;40m▐\x1b[0;97;47m░░░\x1b[0;37;40m▀\x1b[0;97;47m░\x1b[0;37;40m▄\x1b[0;97;47m░░░\x1b[0;97;40m   \x1b[0;97;47m░░\x1b[0;37;40m   ▐\x1b[0;97;47m░░░\x1b[0;37;40m▐\x1b[0;97;47m░░░\x1b[0;37;40m▀\x1b[0;97;47m░\x1b[0;37;40m▄\x1b[0;97;47m░░░\x1b[0;37;40m▐\x1b[0;97;47m░░░\x1b[0;37;40m▀ ▐\x1b[0;97;47m░░░\x1b[0;37;40m▀  ▐\x1b[0;97;47m░░\x1b[0;37;40m████▀ \x1b[0m
\x1b[0;97;40m \x1b[0;90;40m \x1b[0;37;40m▐\x1b[0;97;47m \x1b[0;37;40m▌  \x1b[0;90;40m▐\x1b[0;97;47m  \x1b[0;37;40m█\x1b[0;90;40m▐\x1b[0;97;47m  \x1b[0;37;40m█\x1b[0;97;40m  \x1b[0;37;40m▀███\x1b[0;97;40m   \x1b[0;37;40m██   \x1b[0;90;40m▐\x1b[0;97;47m  \x1b[0;37;40m█\x1b[0;90;40m▐\x1b[0;97;47m  \x1b[0;37;40m█\x1b[0;97;40m  \x1b[0;37;40m▀███▐\x1b[0;97;47m  \x1b[0;37;40m█  ▐\x1b[0;97;47m  \x1b[0;37;40m█\x1b[0;97;47m  \x1b[0;37;40m▌\x1b[0;90;40m▐\x1b[0;97;47m  \x1b[0;97;40m  \x1b[0;37;40m▀██▄\x1b[0m"""


PLAIN_TINYINFER_LOGO = r"""TINYINFER
########  ##  ##   ##  ##    ##  ######  ##  ##   ######  ########  ########
   ##     ##  ###  ##   ##  ##     ##    ###  ##   ##      ##        ##     ##
   ##     ##  #### ##    ####      ##    #### ##   #####   #####     ########
   ##     ##  ## ####     ##       ##    ## ####   ##      ##        ##   ##
   ##     ##  ##  ###     ##     ######  ##  ###   ##      ########  ##    ##"""


class ChatClientError(RuntimeError):
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
class ChatTurn:
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


class ChatClient:
    def __init__(self, host: str):
        self.host = host.rstrip("/")
        if "://" not in self.host:
            self.host = f"http://{self.host}"

    def server_info(self) -> ServerInfo:
        request = urllib.request.Request(f"{self.host}/health", method="GET")
        try:
            with urllib.request.urlopen(request) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            raise ChatClientError(_error_message(error)) from error
        except urllib.error.URLError as error:
            raise ChatClientError(
                f"could not reach TinyInfer at {self.host}: {error.reason}"
            ) from error
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ChatClientError("the TinyInfer health response was not valid") from error

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
            raise ChatClientError(
                "the TinyInfer health response is missing model metadata"
            ) from error

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        on_text: Callable[[str], None],
    ) -> ChatTurn:
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
        turn: ChatTurn | None = None

        try:
            with urllib.request.urlopen(request) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(line[6:])
                        choice = chunk["choices"][0]
                        content = choice["delta"].get("content", "")
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
                        raise ChatClientError("the server sent an invalid chat stream") from error
                    if content:
                        text_parts.append(content)
                        on_text(content)
                    if choice.get("finish_reason") is not None:
                        try:
                            turn = ChatTurn(
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
                            raise ChatClientError(
                                "the server chat stream is missing completion metrics"
                            ) from error
        except urllib.error.HTTPError as error:
            raise ChatClientError(_error_message(error)) from error
        except urllib.error.URLError as error:
            raise ChatClientError(
                f"could not reach TinyInfer at {self.host}: {error.reason}"
            ) from error
        except (OSError, http.client.HTTPException, UnicodeError) as error:
            raise ChatClientError("the server chat stream was interrupted") from error

        if turn is None:
            raise ChatClientError("the server chat stream ended before completion")
        return turn


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _supports_color(output: TextIO) -> bool:
    return "NO_COLOR" not in os.environ and hasattr(output, "isatty") and output.isatty()


def _render_logo(output: TextIO) -> str:
    if _supports_color(output):
        return TINYINFER_LOGO.lstrip("\n")
    return PLAIN_TINYINFER_LOGO


def _known(value: object | None) -> str:
    return "unknown" if value is None else str(value)


def render_banner(info: ServerInfo, *, host: str, max_tokens: int, output: TextIO) -> str:
    architecture = (
        f"{info.architecture} · {_known(info.layers)} layers · "
        f"{_known(info.context_length)} context"
    )
    return "\n".join(
        [
            _render_logo(output),
            "",
            "  a tiny inference server and chat client",
            "",
            f"  runtime    v{info.runtime} · {info.device} · {info.dtype}",
            f"  model      {info.model}",
            f"  arch       {architecture}",
            (
                f"  sampling   {info.sampling_strategy} "
                f"(temperature {info.temperature:g}) · max output {max_tokens}"
            ),
            f"  server     {host}",
            "",
            "  commands   /quit exit · /clear reset conversation",
            "",
        ]
    )


def run_chat_session(
    client: ChatClient,
    *,
    system_prompt: str,
    max_tokens: int,
    input_fn: Callable[[str], str] = input,
    output: TextIO,
) -> int:
    try:
        info = client.server_info()
    except ChatClientError as error:
        output.write(f"Error: {error}\n")
        return 1

    output.write(render_banner(info, host=client.host, max_tokens=max_tokens, output=output))
    messages = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            prompt = input_fn("> ")
        except (EOFError, KeyboardInterrupt, StopIteration):
            output.write("\nGoodbye.\n")
            return 0

        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt.lower() in {"/quit", "/exit"}:
            output.write("Goodbye.\n")
            return 0
        if prompt.lower() == "/clear":
            messages = [{"role": "system", "content": system_prompt}]
            output.write("Conversation cleared.\n\n")
            continue

        candidate_messages = [*messages, {"role": "user", "content": prompt}]
        output.write("\n")
        output.flush()
        try:
            turn = client.complete(
                candidate_messages,
                max_tokens=max_tokens,
                on_text=lambda text: _write_stream(output, text),
            )
        except ChatClientError as error:
            output.write(f"\nError: {error}\n\n")
            continue
        except KeyboardInterrupt:
            output.write("\nGeneration interrupted.\n\n")
            continue

        if not turn.text:
            output.write("\nNo text was generated; conversation history was unchanged.\n\n")
            output.flush()
            continue

        candidate_messages.append({"role": "assistant", "content": turn.text})
        messages = candidate_messages
        output.write(
            "\n\n"
            f"[{turn.generated_tokens} generated · TTFT {turn.time_to_first_token:.2f}s · "
            f"{turn.output_tokens_per_second:.1f} tok/s · {turn.finish_reason}]\n\n"
        )
        output.flush()


def _write_stream(output: TextIO, text: str) -> None:
    output.write(text)
    output.flush()
