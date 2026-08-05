from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from threading import BoundedSemaphore

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from tinyinfer import __version__
from tinyinfer.engine import Engine
from tinyinfer.tokenizer import ALLOWED_ROLES, IM_END, IM_START, Message

MAX_REQUEST_BYTES = 1_048_576
MAX_MESSAGE_CHARS = 65_536


class ReleasingStreamingResponse(StreamingResponse):
    """Release the model slot even if streaming fails before iteration begins."""

    def __init__(self, *args, release, **kwargs):
        super().__init__(*args, **kwargs)
        self._release = release

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._release()


def error_response(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error"}},
        status_code=status_code,
    )


def parse_request(payload: object) -> tuple[list[Message], int, bool]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty list")

    messages: list[Message] = []
    message_chars = 0
    for index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        role = raw_message.get("role")
        content = raw_message.get("content")
        if not isinstance(role, str) or not isinstance(content, str) or not content:
            raise ValueError(f"messages[{index}] requires string role and non-empty content")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"messages[{index}] has unsupported role {role!r}")
        if IM_START in content or IM_END in content:
            raise ValueError(f"messages[{index}] contains a reserved control token")
        message_chars += len(content)
        if message_chars > MAX_MESSAGE_CHARS:
            raise ValueError(f"message content exceeds {MAX_MESSAGE_CHARS} characters")
        messages.append(Message(role=role, content=content))

    max_tokens = payload.get("max_tokens", 32)
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or not 1 <= max_tokens <= 512
    ):
        raise ValueError("max_tokens must be an integer between 1 and 512")
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise ValueError("stream must be true or false")
    return messages, max_tokens, stream


async def read_json(request: Request) -> object:
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
            raise OverflowError(f"request body exceeds {MAX_REQUEST_BYTES} bytes")
        body.extend(chunk)
    return json.loads(body)


def completion_chunk(
    *,
    completion_id: str,
    created: int,
    model_name: str,
    content: str | None,
    finish_reason: str | None,
    usage: dict[str, int] | None = None,
    timings: dict[str, float] | None = None,
) -> dict[str, object]:
    delta: dict[str, str] = {}
    if content is not None:
        delta["content"] = content
    chunk: dict[str, object] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage
    if timings is not None:
        chunk["tinyinfer"] = timings
    return chunk


def engine_metadata(engine: Engine, model_name: str) -> dict[str, object]:
    config = engine.model.config
    dtype = str(next(engine.model.parameters()).dtype).removeprefix("torch.")
    return {
        "status": "ok",
        "runtime": __version__,
        "model": model_name,
        "architecture": "qwen2",
        "layers": config.num_hidden_layers,
        "context_length": config.max_position_embeddings,
        "device": str(engine.device),
        "dtype": dtype,
        "attention": engine.attention_name,
        "kv_cache": engine.kv_cache_name,
        "sampling": {"strategy": "greedy", "temperature": 0.0},
    }


def create_app(engine: Engine, model_name: str) -> Starlette:
    generation_slot = BoundedSemaphore(value=1)
    metadata = engine_metadata(engine, model_name)

    async def health(_: Request) -> JSONResponse:
        return JSONResponse(metadata)

    async def chat_completions(request: Request):
        if not generation_slot.acquire(blocking=False):
            return error_response("the model is already serving another request", status_code=503)

        release_in_response = False
        try:
            try:
                payload = await read_json(request)
                messages, max_tokens, stream = parse_request(payload)
                token_events = engine.stream(messages, max_new_tokens=max_tokens)
            except OverflowError as error:
                return error_response(str(error), status_code=413)
            except json.JSONDecodeError:
                return error_response("request body must contain valid JSON")
            except ValueError as error:
                return error_response(str(error))

            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            if not stream:
                try:
                    result = await run_in_threadpool(
                        engine.generate, messages, max_new_tokens=max_tokens
                    )
                except ValueError as error:
                    return error_response(str(error))
                return JSONResponse(
                    {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": result.text},
                                "finish_reason": result.finish_reason,
                            }
                        ],
                    }
                )

            def events() -> Iterator[str]:
                started_at = time.perf_counter()
                first_token_at: float | None = None
                generated_tokens = 0
                while True:
                    try:
                        event = next(token_events)
                    except StopIteration as stopped:
                        finish_reason = stopped.value or "stop"
                        break
                    generated_tokens += 1
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    chunk = completion_chunk(
                        completion_id=completion_id,
                        created=created,
                        model_name=model_name,
                        content=event.text,
                        finish_reason=None,
                    )
                    yield f"data: {json.dumps(chunk)}\n\n"
                completed_at = time.perf_counter()
                server_ttft = (
                    first_token_at if first_token_at is not None else completed_at
                ) - started_at
                output_seconds = max(completed_at - started_at, 1e-9)
                terminal = completion_chunk(
                    completion_id=completion_id,
                    created=created,
                    model_name=model_name,
                    content=None,
                    finish_reason=finish_reason,
                    usage={"completion_tokens": generated_tokens},
                    timings={
                        "server_ttft_seconds": server_ttft,
                        "time_to_first_token_seconds": server_ttft,
                        "output_tokens_per_second": generated_tokens / output_seconds,
                    },
                )
                yield f"data: {json.dumps(terminal)}\n\n"
                yield "data: [DONE]\n\n"

            release_in_response = True
            return ReleasingStreamingResponse(
                events(), media_type="text/event-stream", release=generation_slot.release
            )
        finally:
            if not release_in_response:
                generation_slot.release()

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        ]
    )
