from __future__ import annotations

import os

from collections.abc import Callable
from typing import TextIO

from tinyinfer.client import ClientError, ServerInfo, TinyInferClient

TINYINFER_LOGO = """
\x1b[0;97;40m▐\x1b[0;97;47m▓▓▓▓▓\x1b[0;97;40m▌▐\x1b[0;97;47m▓▓▓\x1b[0;97;40m▐\x1b[0;97;47m▓\x1b[0;97;40m▄    \x1b[0;97;47m▓▓▓\x1b[0;97;40m▐\x1b[0;97;47m▓\x1b[0;97;40m▄  ▄\x1b[0;97;47m▓▓\x1b[0;97;40m▐\x1b[0;97;47m▓▓▓\x1b[0;97;40m▐\x1b[0;97;47m▓\x1b[0;97;40m▄    \x1b[0;97;47m▓▓▓\x1b[0;97;40m▐\x1b[0;97;47m▓▓▓▓▓\x1b[0;97;40m▐\x1b[0;97;47m▓▓▓▓▓\x1b[0;97;40m▌▐\x1b[0;97;47m▓▓▓▓▓\x1b[0;97;40m▄\x1b[0;37;40m  \x1b[0m
\x1b[0;97;40m  ▐\x1b[0;97;47m▒\x1b[0;37;40m▌  \x1b[0;97;40m▐\x1b[0;97;47m▒▒▒\x1b[0;37;40m▐\x1b[0;97;47m▒▒▒\x1b[0;97;40m▄  \x1b[0;97;47m▒▒▒\x1b[0;97;40m ▀\x1b[0;97;47m▒▒▒▒\x1b[0;97;40m▀\x1b[0;37;40m \x1b[0;97;40m▐\x1b[0;97;47m▒▒▒\x1b[0;37;40m▐\x1b[0;97;47m▒▒▒\x1b[0;97;40m▄  \x1b[0;97;47m▒▒▒\x1b[0;97;40m▐\x1b[0;97;47m▒▒▒\x1b[0;37;40m  \x1b[0;97;40m▐\x1b[0;97;47m▒▒▒\x1b[0;37;40m   ▐\x1b[0;97;47m▒▒\x1b[0;97;40m  ▀\x1b[0;97;47m▒▒\x1b[0;97;40m▄\x1b[0m
\x1b[0;97;40m \x1b[0;90;40m \x1b[0;37;40m▐\x1b[0;97;47m░\x1b[0;37;40m▌  ▐\x1b[0;97;47m░░░\x1b[0;37;40m▐\x1b[0;97;47m░░░\x1b[0;37;40m▀\x1b[0;97;47m░\x1b[0;37;40m▄\x1b[0;97;47m░░░\x1b[0;97;40m   \x1b[0;97;47m░░\x1b[0;37;40m   ▐\x1b[0;97;47m░░░\x1b[0;37;40m▐\x1b[0;97;47m░░░\x1b[0;37;40m▀\x1b[0;97;47m░\x1b[0;37;40m▄\x1b[0;97;47m░░░\x1b[0;37;40m▐\x1b[0;97;47m░░░\x1b[0;37;40m▀ ▐\x1b[0;97;47m░░░\x1b[0;37;40m▀  ▐\x1b[0;97;47m░░\x1b[0;37;40m████▀ \x1b[0m
\x1b[0;97;40m \x1b[0;90;40m \x1b[0;37;40m▐\x1b[0;97;47m \x1b[0;37;40m▌  \x1b[0;90;40m▐\x1b[0;97;47m  \x1b[0;37;40m█\x1b[0;90;40m▐\x1b[0;97;47m  \x1b[0;37;40m█\x1b[0;97;40m  \x1b[0;37;40m▀███\x1b[0;97;40m   \x1b[0;37;40m██   \x1b[0;90;40m▐\x1b[0;97;47m  \x1b[0;37;40m█\x1b[0;90;40m▐\x1b[0;97;47m  \x1b[0;37;40m█\x1b[0;97;40m  \x1b[0;37;40m▀███▐\x1b[0;97;47m  \x1b[0;37;40m█  ▐\x1b[0;97;47m  \x1b[0;37;40m█\x1b[0;97;47m  \x1b[0;37;40m▌\x1b[0;90;40m▐\x1b[0;97;47m  \x1b[0;97;40m  \x1b[0;37;40m▀██▄\x1b[0m"""


PLAIN_TINYINFER_LOGO = r"""TINYINFER"""

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
    client: TinyInferClient,
    *,
    system_prompt: str,
    max_tokens: int,
    input_fn: Callable[[str], str] = input,
    output: TextIO,
) -> int:
    try:
        info = client.get_server_info()
    except ClientError as error:
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
            completion = client.create_chat_completion(
                candidate_messages,
                max_tokens=max_tokens,
                on_text=lambda text: _write_stream(output, text),
            )
        except ClientError as error:
            output.write(f"\nError: {error}\n\n")
            continue
        except KeyboardInterrupt:
            output.write("\nGeneration interrupted.\n\n")
            continue

        if not completion.text:
            output.write("\nNo text was generated; conversation history was unchanged.\n\n")
            output.flush()
            continue

        candidate_messages.append({"role": "assistant", "content": completion.text})
        messages = candidate_messages
        output.write(
            "\n\n"
            f"[{completion.generated_tokens} generated · TTFT {completion.time_to_first_token:.2f}s · "
            f"{completion.output_tokens_per_second:.1f} tok/s · {completion.finish_reason}]\n\n"
        )
        output.flush()


def _write_stream(output: TextIO, text: str) -> None:
    output.write(text)
    output.flush()
