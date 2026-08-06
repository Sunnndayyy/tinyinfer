import io
import json
import urllib.error

import pytest

from tinyinfer.chat import render_banner, run_chat_session
from tinyinfer.client import ChatCompletion, ClientError, ServerInfo, TinyInferClient

HEALTH_PAYLOAD = {
    "status": "ok",
    "runtime": "0.1.0",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "architecture": "qwen2",
    "layers": 28,
    "context_length": 32_768,
    "device": "mps",
    "dtype": "bfloat16",
    "quantization": "q8",
    "kv_cache": "contiguous",
    "decoding": "autoregressive",
    "attention": "sdpa",
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


class MutableClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class TimedHTTPResponse(HTTPResponse):
    def __init__(self, lines, *, clock: MutableClock, advances: list[float]) -> None:
        super().__init__(lines)
        self.clock = clock
        self.advances = advances

    def __iter__(self):
        for line, seconds in zip(self.lines, self.advances, strict=True):
            self.clock.advance(seconds)
            yield line


def health_response() -> HTTPResponse:
    return HTTPResponse(body=json.dumps(HEALTH_PAYLOAD).encode("utf-8"))


def test_server_info_keeps_the_previous_constructor_signature() -> None:
    info = ServerInfo(
        "0.1.0",
        "tiny",
        "qwen2",
        2,
        128,
        "cpu",
        "float32",
        "contiguous",
        "greedy",
        0.0,
    )

    assert info.decoding == "unknown"
    assert info.attention == "unknown"
    assert info.quantization == "unknown"


def completion_response(
    *,
    text="Hello",
    extra_texts=(),
    finish_reason="stop",
    server_ttft=0.125,
    legacy_ttft=0.125,
) -> HTTPResponse:
    terminal_chunk = {
        "choices": [{"delta": {}, "finish_reason": finish_reason}],
        "usage": {"completion_tokens": 2},
        "tinyinfer": {
            "output_tokens_per_second": 16.0,
        },
    }
    if server_ttft is not None:
        terminal_chunk["tinyinfer"]["server_ttft_seconds"] = server_ttft
    if legacy_ttft is not None:
        terminal_chunk["tinyinfer"]["time_to_first_token_seconds"] = legacy_ttft
    content_chunks = []
    if text is not None:
        content_chunks = [text, *extra_texts]
    lines = [
        f"data: {json.dumps({'choices': [{'delta': {'content': content}, 'finish_reason': None}]})}\n\n".encode()
        for content in content_chunks
    ]
    lines.extend(
        [
            f"data: {json.dumps(terminal_chunk)}\n\n".encode(),
            b"data: [DONE]\n\n",
        ]
    )
    return HTTPResponse(lines)


class RecordingClient:
    def __init__(self) -> None:
        self.calls = []
        self.host = "http://localhost:8000"

    def get_server_info(self) -> ServerInfo:
        return ServerInfo(
            runtime="0.1.0",
            model="Qwen/Qwen2.5-1.5B-Instruct",
            architecture="qwen2",
            layers=28,
            context_length=32_768,
            device="mps",
            dtype="bfloat16",
            quantization="q8",
            kv_cache="contiguous",
            decoding="autoregressive",
            attention="sdpa",
            sampling_strategy="greedy",
            temperature=0.0,
        )

    def create_chat_completion(self, messages, *, max_tokens, on_text) -> ChatCompletion:
        self.calls.append((list(messages), max_tokens))
        reply = f"reply {len(self.calls)}"
        on_text(reply)
        return ChatCompletion(
            text=reply,
            finish_reason="stop",
            generated_tokens=3,
            client_ttft=0.5,
            server_ttft=0.125,
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
        client.get_server_info(),
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
        client.get_server_info(),
        host=client.host,
        max_tokens=128,
        output=output,
    )

    assert banner.startswith("TINYINFER\n")
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
    assert "autoregressive" in rendered
    assert "weights    q8" in rendered
    assert "greedy (temperature 0) · max output 64" in rendered
    assert "Conversation cleared." in rendered
    assert "3 generated · TTFT 0.50s · 24.0 tok/s" in rendered


class RejectFirstClient(RecordingClient):
    def create_chat_completion(self, messages, *, max_tokens, on_text) -> ChatCompletion:
        if not self.calls:
            self.calls.append((list(messages), max_tokens))
            raise ClientError("prompt exceeds the model context")
        return super().create_chat_completion(messages, max_tokens=max_tokens, on_text=on_text)


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
    def create_chat_completion(self, messages, *, max_tokens, on_text) -> ChatCompletion:
        if not self.calls:
            self.calls.append((list(messages), max_tokens))
            return ChatCompletion(
                text="",
                finish_reason="stop",
                generated_tokens=0,
                client_ttft=0.0,
                server_ttft=0.0,
                output_tokens_per_second=0.0,
            )
        return super().create_chat_completion(messages, max_tokens=max_tokens, on_text=on_text)


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

    def urlopen(request, *, timeout):
        requests.append(request)
        return health_response()

    monkeypatch.setattr("tinyinfer.client.urllib.request.urlopen", urlopen)

    info = TinyInferClient("localhost:8000/").get_server_info()

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
        quantization="q8",
        kv_cache="contiguous",
        decoding="autoregressive",
        attention="sdpa",
        sampling_strategy="greedy",
        temperature=0.0,
    )


def test_server_info_keeps_decoding_optional_for_older_callers() -> None:
    info = ServerInfo(
        "tinyinfer",
        "test-model",
        "qwen2",
        2,
        32768,
        "cpu",
        "float32",
        "contiguous",
        "greedy",
        0.0,
    )

    assert info.decoding == "unknown"
    assert info.attention == "unknown"


def test_chat_client_posts_history_and_parses_terminal_metrics(monkeypatch) -> None:
    requests = []
    clock = MutableClock(10.0)

    def urlopen(request, *, timeout):
        requests.append(request)
        clock.advance(0.2)
        response = completion_response(legacy_ttft=9.0)
        return TimedHTTPResponse(response.lines, clock=clock, advances=[0.3, 1.0, 0.0])

    monkeypatch.setattr("tinyinfer.client.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("tinyinfer.client.time.perf_counter", clock)
    streamed = []
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]

    completion = TinyInferClient("http://localhost:8000").create_chat_completion(
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
    assert completion == ChatCompletion(
        text="Hello",
        finish_reason="stop",
        generated_tokens=2,
        client_ttft=0.5,
        server_ttft=0.125,
        output_tokens_per_second=16.0,
    )
    assert completion.time_to_first_token == 0.125


def test_chat_client_accepts_legacy_server_ttft_metric(monkeypatch) -> None:
    clock = iter([30.0, 30.1])
    monkeypatch.setattr(
        "tinyinfer.client.urllib.request.urlopen",
        lambda request, *, timeout: completion_response(
            server_ttft=None,
            legacy_ttft=0.25,
        ),
    )
    monkeypatch.setattr("tinyinfer.client.time.perf_counter", lambda: next(clock))

    completion = TinyInferClient("localhost:8000").create_chat_completion(
        [{"role": "user", "content": "Hello"}],
        max_tokens=4,
        on_text=lambda _: None,
    )

    assert completion.server_ttft == 0.25
    assert completion.time_to_first_token == 0.25


def test_chat_client_times_empty_first_token_without_rendering_it(monkeypatch) -> None:
    clock = MutableClock(40.0)

    def urlopen(request, *, timeout):
        clock.advance(0.1)
        response = completion_response(text="", extra_texts=("Hello",))
        return TimedHTTPResponse(response.lines, clock=clock, advances=[0.2, 0.4, 0.5, 0.0])

    monkeypatch.setattr("tinyinfer.client.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("tinyinfer.client.time.perf_counter", clock)
    streamed = []

    completion = TinyInferClient("localhost:8000").create_chat_completion(
        [{"role": "user", "content": "Hello"}],
        max_tokens=4,
        on_text=streamed.append,
    )

    assert completion.client_ttft == pytest.approx(0.3)
    assert completion.text == "Hello"
    assert streamed == ["Hello"]


def test_chat_client_uses_terminal_arrival_when_no_text_is_streamed(monkeypatch) -> None:
    clock = iter([20.0, 20.25])
    monkeypatch.setattr(
        "tinyinfer.client.urllib.request.urlopen",
        lambda request, *, timeout: completion_response(text=None),
    )
    monkeypatch.setattr("tinyinfer.client.time.perf_counter", lambda: next(clock))

    completion = TinyInferClient("localhost:8000").create_chat_completion(
        [{"role": "user", "content": "Stop immediately"}],
        max_tokens=4,
        on_text=lambda _: None,
    )

    assert completion.client_ttft == 0.25
    assert completion.server_ttft == 0.125


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
    monkeypatch.setattr(
        "tinyinfer.client.urllib.request.urlopen",
        lambda request, *, timeout: response,
    )

    with pytest.raises(ClientError, match=message):
        TinyInferClient("localhost:8000").create_chat_completion(
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
        "tinyinfer.client.urllib.request.urlopen",
        lambda request, *, timeout: (_ for _ in ()).throw(error),
    )

    with pytest.raises(ClientError, match="server returned HTTP 503: model is busy"):
        TinyInferClient("localhost:8000").create_chat_completion(
            [{"role": "user", "content": "Hello"}],
            max_tokens=4,
            on_text=lambda _: None,
        )


def test_interrupted_stream_does_not_commit_partial_history(monkeypatch) -> None:
    post_payloads = []

    def urlopen(request, *, timeout):
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

    monkeypatch.setattr("tinyinfer.client.urllib.request.urlopen", urlopen)
    output = io.StringIO()

    result = run_chat_session(
        TinyInferClient("localhost:8000"),
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
