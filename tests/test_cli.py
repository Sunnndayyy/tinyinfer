from argparse import Namespace

import pytest

from tinyinfer import benchmark, chat, cli
from tinyinfer.cli import build_parser, run_chat, run_leaderboard


def test_chat_command_is_interactive_and_only_needs_a_host() -> None:
    args = build_parser().parse_args(["chat", "--host", "http://localhost:9000"])

    assert args.command == "chat"
    assert args.host == "http://localhost:9000"
    assert args.max_tokens == 128
    assert not hasattr(args, "prompt")


@pytest.mark.parametrize("command", ["generate", "serve", "bench"])
def test_model_commands_default_to_reference_runtime(command: str) -> None:
    arguments = [command]
    if command == "generate":
        arguments.extend(("--prompt", "hello"))

    args = build_parser().parse_args(arguments)

    assert args.kv_cache == "contiguous"
    assert args.decoding == "autoregressive"


@pytest.mark.parametrize("cache_name", ["none", "contiguous", "paged"])
def test_model_commands_accept_each_kv_cache(cache_name: str) -> None:
    args = build_parser().parse_args(["bench", "--kv-cache", cache_name])

    assert args.kv_cache == cache_name


def test_model_commands_accept_autoregressive_decoding() -> None:
    args = build_parser().parse_args(["bench", "--decoding", "autoregressive"])

    assert args.decoding == "autoregressive"


def test_benchmark_can_save_a_named_profile() -> None:
    args = build_parser().parse_args(
        ["bench", "--profile", "decode", "--kv-cache", "contiguous", "--save"]
    )

    assert args.profile == "decode"
    assert args.save is True


def test_leaderboard_command_needs_no_arguments() -> None:
    args = build_parser().parse_args(["leaderboard"])

    assert args.command == "leaderboard"


def test_leaderboard_prints_generated_markdown(monkeypatch, capsys) -> None:
    monkeypatch.setattr(benchmark, "write_leaderboard", lambda: "# Results\n")

    assert run_leaderboard() == 0
    assert capsys.readouterr().out == "# Results\n"


def test_chat_command_rejects_output_limits_outside_server_bounds(capsys) -> None:
    for max_tokens in (0, 513):
        result = run_chat(
            Namespace(
                host="http://localhost:9000",
                system="system",
                max_tokens=max_tokens,
            )
        )

        assert result == 2
    assert capsys.readouterr().err.count("max-tokens must be between 1 and 512") == 2


def test_chat_command_starts_interactive_session(monkeypatch) -> None:
    calls = []
    client = object()
    monkeypatch.setattr(
        cli,
        "TinyInferClient",
        lambda host: calls.append(("host", host)) or client,
    )

    def fake_session(received_client, **options):
        calls.append(("session", received_client, options))
        return 7

    monkeypatch.setattr(chat, "run_chat_session", fake_session)

    result = run_chat(
        Namespace(
            host="localhost:9000",
            system="Be concise.",
            max_tokens=64,
        )
    )

    assert result == 7
    assert calls[0] == ("host", "localhost:9000")
    assert calls[1][0:2] == ("session", client)
    assert calls[1][2]["system_prompt"] == "Be concise."
    assert calls[1][2]["max_tokens"] == 64
