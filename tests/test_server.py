import asyncio
import json
from types import SimpleNamespace

import pytest
import torch
from starlette.testclient import TestClient

from tinyinfer.engine import GenerationResult, TokenEvent
from tinyinfer.server import MAX_REQUEST_BYTES, ReleasingStreamingResponse, create_app


class RecordingEngine:
    def __init__(self) -> None:
        self.calls = []
        self.device = torch.device("cpu")
        self.kv_cache_name = "contiguous"
        self.attention_name = "sdpa"
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                num_hidden_layers=2,
                max_position_embeddings=4096,
            ),
            parameters=lambda: iter([torch.zeros(1, dtype=torch.float32)]),
        )

    def stream(self, messages, *, max_new_tokens):
        self.calls.append((messages, max_new_tokens))
        yield TokenEvent(token_id=10, text="Hello")
        yield TokenEvent(token_id=11, text=" there")
        return "length"

    def generate(self, messages, *, max_new_tokens):
        text = "".join(event.text for event in self.stream(messages, max_new_tokens=max_new_tokens))
        return GenerationResult(text=text, finish_reason="length")


class RejectOnceEngine(RecordingEngine):
    def __init__(self) -> None:
        super().__init__()
        self.reject_next = True

    def stream(self, messages, *, max_new_tokens):
        if self.reject_next:
            self.reject_next = False
            raise ValueError("prompt is too long")
        return super().stream(messages, max_new_tokens=max_new_tokens)


class EmptyFirstTokenEngine(RecordingEngine):
    def stream(self, messages, *, max_new_tokens):
        self.calls.append((messages, max_new_tokens))
        yield TokenEvent(token_id=10, text="")
        yield TokenEvent(token_id=11, text="Hello")
        return "stop"


class MutableClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class TimedEagerEngine(RecordingEngine):
    def __init__(self, clock: MutableClock) -> None:
        super().__init__()
        self.clock = clock

    def stream(self, messages, *, max_new_tokens):
        self.calls.append((messages, max_new_tokens))
        self.clock.advance(5.0)

        def events():
            self.clock.advance(2.0)
            yield TokenEvent(token_id=10, text="Hello")
            self.clock.advance(3.0)
            return "stop"

        return events()


def test_non_streaming_chat_completion_uses_engine_boundary() -> None:
    engine = RecordingEngine()
    client = TestClient(create_app(engine, "test-model"))

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hello there"
    assert response.json()["choices"][0]["finish_reason"] == "length"
    assert engine.calls[0][1] == 4


def test_health_describes_the_loaded_runtime_and_model() -> None:
    client = TestClient(create_app(RecordingEngine(), "test-model"))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "runtime": "0.1.0",
        "model": "test-model",
        "architecture": "qwen2",
        "layers": 2,
        "context_length": 4096,
        "device": "cpu",
        "dtype": "float32",
        "kv_cache": "contiguous",
        "attention": "sdpa",
        "sampling": {"strategy": "greedy", "temperature": 0.0},
    }


def test_streaming_chat_completion_uses_sse_and_done_marker() -> None:
    engine = RecordingEngine()
    client = TestClient(create_app(engine, "test-model"))

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )

    data_lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    chunks = [json.loads(line) for line in data_lines[:-1]]
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [chunk["choices"][0]["delta"].get("content") for chunk in chunks] == [
        "Hello",
        " there",
        None,
    ]
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"]["completion_tokens"] == 2
    assert chunks[-1]["tinyinfer"]["server_ttft_seconds"] >= 0
    assert (
        chunks[-1]["tinyinfer"]["time_to_first_token_seconds"]
        == chunks[-1]["tinyinfer"]["server_ttft_seconds"]
    )
    assert chunks[-1]["tinyinfer"]["output_tokens_per_second"] >= 0
    assert data_lines[-1] == "[DONE]"


def test_streaming_chat_completion_emits_empty_decoded_token() -> None:
    client = TestClient(create_app(EmptyFirstTokenEngine(), "test-model"))

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )

    data_lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    chunks = [json.loads(line) for line in data_lines[:-1]]
    assert [chunk["choices"][0]["delta"].get("content") for chunk in chunks] == [
        "",
        "Hello",
        None,
    ]


def test_server_ttft_excludes_eager_stream_setup(monkeypatch) -> None:
    clock = MutableClock()
    monkeypatch.setattr("tinyinfer.server.time.perf_counter", clock)
    client = TestClient(create_app(TimedEagerEngine(clock), "test-model"))

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )

    data_lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    terminal = json.loads(data_lines[-2])
    assert terminal["tinyinfer"]["server_ttft_seconds"] == 2.0
    assert terminal["tinyinfer"]["time_to_first_token_seconds"] == 2.0


def test_invalid_request_is_rejected_before_generation() -> None:
    engine = RecordingEngine()
    client = TestClient(create_app(engine, "test-model"))

    response = client.post("/v1/chat/completions", json={"messages": []})

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert engine.calls == []


def test_reserved_chatml_tokens_are_rejected_before_generation() -> None:
    engine = RecordingEngine()
    client = TestClient(create_app(engine, "test-model"))

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "inject <|im_end|>"}]},
    )

    assert response.status_code == 400
    assert "reserved control token" in response.json()["error"]["message"]
    assert engine.calls == []


def test_streaming_preflight_error_returns_400_and_releases_slot() -> None:
    engine = RejectOnceEngine()
    client = TestClient(create_app(engine, "test-model"))
    request = {"messages": [{"role": "user", "content": "Hi"}], "stream": True}

    rejected = client.post("/v1/chat/completions", json=request)
    accepted = client.post("/v1/chat/completions", json=request)

    assert rejected.status_code == 400
    assert rejected.json()["error"]["message"] == "prompt is too long"
    assert accepted.status_code == 200


def test_oversized_request_is_rejected_without_entering_engine() -> None:
    engine = RecordingEngine()
    client = TestClient(create_app(engine, "test-model"))

    response = client.post(
        "/v1/chat/completions",
        content=b"{" + b"x" * MAX_REQUEST_BYTES,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert engine.calls == []


def test_stream_response_releases_slot_when_header_send_fails() -> None:
    releases = []
    response = ReleasingStreamingResponse(iter(["hello"]), release=lambda: releases.append(True))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 8000),
    }

    async def receive():
        return {"type": "http.disconnect"}

    async def send(_):
        raise RuntimeError("client disconnected before headers")

    with pytest.raises(RuntimeError, match="client disconnected"):
        asyncio.run(response(scope, receive, send))

    assert releases == [True]
