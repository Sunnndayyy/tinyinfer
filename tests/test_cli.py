from argparse import Namespace

from tinyinfer import chat
from tinyinfer.cli import build_parser, run_chat


def test_chat_command_is_interactive_and_only_needs_a_host() -> None:
    args = build_parser().parse_args(["chat", "--host", "http://localhost:9000"])

    assert args.command == "chat"
    assert args.host == "http://localhost:9000"
    assert args.max_tokens == 128
    assert not hasattr(args, "prompt")


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
    monkeypatch.setattr(chat, "ChatClient", lambda host: calls.append(("host", host)) or client)

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
