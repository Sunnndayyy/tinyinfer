from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from tinyinfer.artifacts import DEFAULT_MODEL, resolve_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinyinfer",
        description="Run a real, readable LLM inference engine.",
    )
    commands = parser.add_subparsers(dest="command")

    download = commands.add_parser("download", help="download model artifacts")
    download.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    download.add_argument("--cache-dir")

    generate = commands.add_parser("generate", help="generate text locally with TinyInfer")
    generate.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--system", default="You are a helpful assistant.")
    generate.add_argument("--max-new-tokens", type=int, default=32)
    generate.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    generate.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    generate.add_argument("--cache-dir")

    serve = commands.add_parser("serve", help="serve the real model over HTTP")
    serve.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    serve.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    serve.add_argument("--cache-dir")
    serve.add_argument(
        "--allow-remote",
        action="store_true",
        help="acknowledge that a non-loopback server has no authentication",
    )

    chat = commands.add_parser("chat", help="chat with a running TinyInfer server")
    chat.add_argument("--host", default="http://127.0.0.1:8000")
    chat.add_argument("--system", default="You are a helpful assistant.")
    chat.add_argument("--max-tokens", type=int, default=128)

    bench = commands.add_parser("bench", help="benchmark local generation")
    bench.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    bench.add_argument("--prompt", default="Explain what a KV cache saves in one sentence.")
    bench.add_argument("--system", default="You are a helpful assistant.")
    bench.add_argument("--max-new-tokens", type=int, default=16)
    bench.add_argument("--warmup", type=int, default=1)
    bench.add_argument("--repetitions", type=int, default=3)
    bench.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    bench.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    bench.add_argument("--cache-dir")
    bench.add_argument("--json", action="store_true", dest="json_output")
    return parser


def run_download(args: argparse.Namespace) -> int:
    print(resolve_model(args.model, args.cache_dir))
    return 0


def run_generate(args: argparse.Namespace) -> int:
    from tinyinfer.engine import Engine
    from tinyinfer.tokenizer import Message

    model_dir = resolve_model(args.model, args.cache_dir)
    engine = Engine.from_pretrained(
        model_dir,
        device_name=args.device,
        dtype_name=args.dtype,
    )
    messages = [
        Message(role="system", content=args.system),
        Message(role="user", content=args.prompt),
    ]
    for event in engine.stream(messages, max_new_tokens=args.max_new_tokens):
        print(event.text, end="", flush=True)
    print()
    return 0


def run_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from tinyinfer.engine import Engine
    from tinyinfer.server import create_app

    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_remote:
        print(
            "refusing to expose an unauthenticated server; add --allow-remote to acknowledge",
            file=sys.stderr,
        )
        return 2
    if args.allow_remote:
        print("WARNING: TinyInfer has no authentication or TLS.", file=sys.stderr)

    model_dir = resolve_model(args.model, args.cache_dir)
    print(f"Loading {args.model} from {model_dir}...", file=sys.stderr)
    engine = Engine.from_pretrained(
        model_dir,
        device_name=args.device,
        dtype_name=args.dtype,
    )
    app = create_app(engine, args.model)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def run_chat(args: argparse.Namespace) -> int:
    from tinyinfer.chat import ChatClient, run_chat_session

    if not 1 <= args.max_tokens <= 512:
        print("max-tokens must be between 1 and 512", file=sys.stderr)
        return 2
    return run_chat_session(
        ChatClient(args.host),
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        output=sys.stdout,
    )


def run_bench(args: argparse.Namespace) -> int:
    from tinyinfer.benchmark import benchmark, format_summary
    from tinyinfer.engine import Engine
    from tinyinfer.tokenizer import Message

    if args.warmup < 0 or args.repetitions < 1 or args.max_new_tokens < 1:
        print("warmup must be >= 0; repetitions and max-new-tokens must be >= 1", file=sys.stderr)
        return 2
    model_dir = resolve_model(args.model, args.cache_dir)
    engine = Engine.from_pretrained(
        model_dir,
        device_name=args.device,
        dtype_name=args.dtype,
    )
    messages = [
        Message(role="system", content=args.system),
        Message(role="user", content=args.prompt),
    ]
    result = benchmark(
        engine,
        messages,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
        repetitions=args.repetitions,
        metadata={
            "model": args.model,
            "device": str(engine.device),
            "dtype": str(next(engine.model.parameters()).dtype),
            "max_new_tokens": args.max_new_tokens,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
        },
    )
    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        print(format_summary(result))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if args.command == "download":
        return run_download(args)
    if args.command == "generate":
        return run_generate(args)
    if args.command == "serve":
        return run_serve(args)
    if args.command == "chat":
        return run_chat(args)
    if args.command == "bench":
        return run_bench(args)
    parser.error(f"unknown command: {args.command}")
    return 2
