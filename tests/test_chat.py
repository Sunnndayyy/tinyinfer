import io
import json
import urllib.error

import pytest

from tinyinfer.chat import (
    ChatClient,
    ChatClientError,
    ChatTurn,
    ServerInfo,
    render_banner,
    run_chat_session,
)

HEALTH_PAYLOAD = {
    "status": "ok",
    "runtime": "0.1.0",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "architecture": "qwen2",
    "layers": 28,
    "context_length": 32_768,
    "device": "mps",
    "dtype": "bfloat16",
    "sampling": {"strategy": "greedy", "temperature": 0.0},
}


class HTTPResponse:
    def __init__(self, lines=(), *, body=b"", iteration_error=None) -> None:
        self.lines = list(lines)
        self.body = body
        self.iteration_error = iteration_error

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.body

    def __iter__(self):
        yield from self.lines
        if self.iteration_error is not None:
            raise self.iteration_error


def health_response() -> HTTPResponse:
    return HTTPResponse(body=json.dumps(HEALTH_PAYLOAD).encode("utf-8"))


def completion_response(*, text="Hello", finish_reason="stop") -> HTTPResponse:
    content_chunk = {"choices": [{"delta": {"content": text}, "finish_reason": None}]}
    terminal_chunk = {
        "choices": [{"delta": {}, "finish_reason": finish_reason}],
        "usage": {"completion_tokens": 2},
        "tinyinfer": {
            "time_to_first_token_seconds": 0.125,
            "output_tokens_per_second": 16.0,
        },
    }
    return HTTPResponse(
        [
            f"data: {json.dumps(content_chunk)}\n\n".encode(),
            f"data: {json.dumps(terminal_chunk)}\n\n".encode(),
            b"data: [DONE]\n\n",
        ]
    )


class RecordingClient:
    def __init__(self) -> None:
        self.calls = []
        self.host = "http://localhost:8000"

    def server_info(self) -> ServerInfo:
        return ServerInfo(
            runtime="0.1.0",
            model="Qwen/Qwen2.5-1.5B-Instruct",
            architecture="qwen2",
            layers=28,
            context_length=32_768,
            device="mps",
            dtype="bfloat16",
            sampling_strategy="greedy",
            temperature=0.0,
        )

    def complete(self, messages, *, max_tokens, on_text) -> ChatTurn:
        self.calls.append((list(messages), max_tokens))
        reply = f"reply {len(self.calls)}"
        on_text(reply)
        return ChatTurn(
            text=reply,
            finish_reason="stop",
            generated_tokens=3,
            time_to_first_token=0.125,
            output_tokens_per_second=24.0,
        )


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def scripted_input(*values):
    prompts = iter(values)
    return lambda _: next(prompts)


def test_banner_renders_shaded_logo_in_color_terminals(monkeypatch) -> None:
    client = RecordingClient()
    output = TTYBuffer()
    monkeypatch.delenv("NO_COLOR", raising=False)

    banner = render_banner(
        client.server_info(),
        host=client.host,
        max_tokens=128,
        output=output,
    )

    assert "\033[0;97;47m" in banner
    assert "▓▓▓▓▓" in banner


def test_banner_uses_plain_logo_when_color_is_disabled(monkeypatch) -> None:
    client = RecordingClient()
    output = TTYBuffer()
    monkeypatch.setenv("NO_COLOR", "1")

    banner = render_banner(
        client.server_info(),
        host=client.host,
        max_tokens=128,
        output=output,
    )

    assert banner.startswith("TINYINFER\n########")
    assert "\033[" not in banner


def test_chat_session_keeps_history_and_clear_resets_it() -> None:
    client = RecordingClient()
    output = io.StringIO()

    result = run_chat_session(
        client,
        system_prompt="Be concise.",
        max_tokens=64,
        input_fn=scripted_input("hello", "remember that", "/clear", "fresh start", "/quit"),
        output=output,
    )

    assert result == 0
    assert [[message["content"] for message in call[0]] for call in client.calls] == [
        ["Be concise.", "hello"],
        ["Be concise.", "hello", "reply 1", "remember that"],
        ["Be concise.", "fresh start"],
    ]
    rendered = output.getvalue()
    assert "TINYINFER" in rendered
    assert "Qwen/Qwen2.5-1.5B-Instruct" in rendered
    assert "qwen2 · 28 layers · 32768 context" in rendered
    assert "greedy (temperature 0) · max output 64" in rendered
    assert "Conversation cleared." in rendered
    assert "3 generated · TTFT 0.12s · 24.0 tok/s" in rendered


class RejectFirstClient(RecordingClient):
    def complete(self, messages, *, max_tokens, on_text) -> ChatTurn:
        if not self.calls:
            self.calls.append((list(messages), max_tokens))
            raise ChatClientError("prompt exceeds the model context")
        return super().complete(messages, max_tokens=max_tokens, on_text=on_text)


def test_failed_turn_is_not_added_to_conversation_history() -> None:
    client = RejectFirstClient()
    output = io.StringIO()

    result = run_chat_session(
        client,
        system_prompt="system",
        max_tokens=8,
        input_fn=scripted_input("too much", "try again", "/quit"),
        output=output,
    )

    assert result == 0
    assert [message["content"] for message in client.calls[1][0]] == ["system", "try again"]
    assert "Error: prompt exceeds the model context" in output.getvalue()


class EmptyFirstClient(RecordingClient):
    def complete(self, messages, *, max_tokens, on_text) -> ChatTurn:
        if not self.calls:
            self.calls.append((list(messages), max_tokens))
            return ChatTurn(
                text="",
                finish_reason="stop",
                generated_tokens=0,
                time_to_first_token=0.0,
                output_tokens_per_second=0.0,
            )
        return super().complete(messages, max_tokens=max_tokens, on_text=on_text)


def test_empty_turn_is_not_added_to_conversation_history() -> None:
    client = EmptyFirstClient()
    output = io.StringIO()

    result = run_chat_session(
        client,
        system_prompt="system",
        max_tokens=8,
        input_fn=scripted_input("say nothing", "try again", "/quit"),
        output=output,
    )

    assert result == 0
    assert [message["content"] for message in client.calls[1][0]] == [
        "system",
        "try again",
    ]
    assert "no text was generated" in output.getvalue().lower()


def test_chat_client_reads_health_metadata(monkeypatch) -> None:
    requests = []

    def urlopen(request):
        requests.append(request)
        return health_response()

    monkeypatch.setattr("tinyinfer.chat.urllib.request.urlopen", urlopen)

    info = ChatClient("localhost:8000/").server_info()

    assert requests[0].full_url == "http://localhost:8000/health"
    assert requests[0].get_method() == "GET"
    assert info == ServerInfo(
        runtime="0.1.0",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        architecture="qwen2",
        layers=28,
        context_length=32_768,
        device="mps",
        dtype="bfloat16",
        sampling_strategy="greedy",
        temperature=0.0,
    )


def test_chat_client_posts_history_and_parses_terminal_metrics(monkeypatch) -> None:
    requests = []

    def urlopen(request):
        requests.append(request)
        return completion_response()

    monkeypatch.setattr("tinyinfer.chat.urllib.request.urlopen", urlopen)
    streamed = []
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]

    turn = ChatClient("http://localhost:8000").complete(
        messages,
        max_tokens=24,
        on_text=streamed.append,
    )

    request = requests[0]
    assert request.full_url == "http://localhost:8000/v1/chat/completions"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data) == {
        "messages": messages,
        "max_tokens": 24,
        "stream": True,
    }
    assert streamed == ["Hello"]
    assert turn == ChatTurn(
        text="Hello",
        finish_reason="stop",
        generated_tokens=2,
        time_to_first_token=0.125,
        output_tokens_per_second=16.0,
    )


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (HTTPResponse([b"data: not-json\n\n"]), "invalid chat stream"),
        (
            HTTPResponse([b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n']),
            "ended before completion",
        ),
    ],
)
def test_chat_client_rejects_incomplete_streams(monkeypatch, response, message) -> None:
    monkeypatch.setattr("tinyinfer.chat.urllib.request.urlopen", lambda request: response)

    with pytest.raises(ChatClientError, match=message):
        ChatClient("localhost:8000").complete(
            [{"role": "user", "content": "Hello"}],
            max_tokens=4,
            on_text=lambda _: None,
        )


def test_chat_client_translates_http_errors(monkeypatch) -> None:
    error = urllib.error.HTTPError(
        "http://localhost:8000/v1/chat/completions",
        503,
        "Service Unavailable",
        {},
        io.BytesIO(b'{"error":{"message":"model is busy"}}'),
    )
    monkeypatch.setattr(
        "tinyinfer.chat.urllib.request.urlopen",
        lambda request: (_ for _ in ()).throw(error),
    )

    with pytest.raises(ChatClientError, match="server returned HTTP 503: model is busy"):
        ChatClient("localhost:8000").complete(
            [{"role": "user", "content": "Hello"}],
            max_tokens=4,
            on_text=lambda _: None,
        )


def test_interrupted_stream_does_not_commit_partial_history(monkeypatch) -> None:
    post_payloads = []

    def urlopen(request):
        if request.get_method() == "GET":
            return health_response()
        post_payloads.append(json.loads(request.data))
        if len(post_payloads) == 1:
            partial_chunk = {"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]}
            return HTTPResponse(
                [f"data: {json.dumps(partial_chunk)}\n\n".encode()],
                iteration_error=OSError("connection reset"),
            )
        return completion_response(text="recovered")

    monkeypatch.setattr("tinyinfer.chat.urllib.request.urlopen", urlopen)
    output = io.StringIO()

    result = run_chat_session(
        ChatClient("localhost:8000"),
        system_prompt="system",
        max_tokens=8,
        input_fn=scripted_input("first", "second", "/quit"),
        output=output,
    )

    assert result == 0
    assert post_payloads[1]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "second"},
    ]
    assert "partial" in output.getvalue()
    assert "chat stream was interrupted" in output.getvalue()


def test_eof_exits_chat_session_cleanly() -> None:
    client = RecordingClient()
    output = io.StringIO()

    result = run_chat_session(
        client,
        system_prompt="system",
        max_tokens=8,
        input_fn=lambda _: (_ for _ in ()).throw(EOFError),
        output=output,
    )

    assert result == 0
    assert client.calls == []
    assert output.getvalue().endswith("Goodbye.\n")
