from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence

from tinyinfer.artifacts import DEFAULT_MODEL, resolve_model
from tinyinfer.client import TinyInferClient
from tinyinfer.runtime import (
    ATTENTION_NAMES,
    DECODING_NAMES,
    DEFAULT_ATTENTION,
    DEFAULT_DECODING,
    DEFAULT_KV_CACHE,
    DEFAULT_QUANTIZATION,
    KV_CACHE_NAMES,
    QUANTIZATION_NAMES,
)

ROOFLINE_OPTION_DEFAULTS = (
    ("model", "model", DEFAULT_MODEL),
    ("profile", "--profile", None),
    ("prompt", "--prompt", None),
    ("system", "--system", None),
    ("max_new_tokens", "--max-new-tokens", None),
    ("warmup", "--warmup", None),
    ("repetitions", "--repetitions", None),
    ("device", "--device", "auto"),
    ("dtype", "--dtype", "auto"),
    ("decoding", "--decoding", DEFAULT_DECODING),
    ("attention", "--attention", DEFAULT_ATTENTION),
    ("kv_cache", "--kv-cache", DEFAULT_KV_CACHE),
    ("quantization", "--quantization", DEFAULT_QUANTIZATION),
    ("cache_dir", "--cache-dir", None),
    ("json_output", "--json", False),
    ("save", "--save", False),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinyinfer",
        description="a tiny llm inference server and chat client",
    )
    commands = parser.add_subparsers(dest="command")

    download = commands.add_parser("download", help="download model artifacts")
    download.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    download.add_argument("--cache-dir")

    quantize = commands.add_parser("quantize", help="convert a model to packed Q8 weights")
    quantize.add_argument("model")
    quantize.add_argument("--format", choices=("q8",), default="q8")
    quantize.add_argument("--output", required=True)
    quantize.add_argument("--group-size", type=int, choices=(32,), default=32)
    quantize.add_argument("--cache-dir")

    generate = commands.add_parser("generate", help="generate text locally with TinyInfer")
    generate.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--system", default="You are a helpful assistant.")
    generate.add_argument("--max-new-tokens", type=int, default=32)
    generate.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    generate.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    generate.add_argument("--decoding", choices=DECODING_NAMES, default=DEFAULT_DECODING)
    generate.add_argument("--attention", choices=ATTENTION_NAMES, default=DEFAULT_ATTENTION)
    generate.add_argument("--kv-cache", choices=KV_CACHE_NAMES, default=DEFAULT_KV_CACHE)
    generate.add_argument(
        "--quantization",
        choices=("auto", *QUANTIZATION_NAMES),
        default=DEFAULT_QUANTIZATION,
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
    serve.add_argument("--decoding", choices=DECODING_NAMES, default=DEFAULT_DECODING)
    serve.add_argument("--attention", choices=ATTENTION_NAMES, default=DEFAULT_ATTENTION)
    serve.add_argument("--kv-cache", choices=KV_CACHE_NAMES, default=DEFAULT_KV_CACHE)
    serve.add_argument(
        "--quantization",
        choices=("auto", *QUANTIZATION_NAMES),
        default=DEFAULT_QUANTIZATION,
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

    bench = commands.add_parser(
        "bench",
        aliases=("benchmark",),
        help="benchmark local generation and kernels",
    )
    bench.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    bench.add_argument("--profile")
    bench.add_argument("--prompt")
    bench.add_argument("--system")
    bench.add_argument("--max-new-tokens", type=int)
    bench.add_argument("--warmup", type=int)
    bench.add_argument("--repetitions", type=int)
    bench.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    bench.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    bench.add_argument("--decoding", choices=DECODING_NAMES, default=DEFAULT_DECODING)
    bench.add_argument("--attention", choices=ATTENTION_NAMES, default=DEFAULT_ATTENTION)
    bench.add_argument("--kv-cache", choices=KV_CACHE_NAMES, default=DEFAULT_KV_CACHE)
    bench.add_argument(
        "--quantization",
        choices=("auto", *QUANTIZATION_NAMES),
        default=DEFAULT_QUANTIZATION,
    )
    bench.add_argument("--cache-dir")
    bench.add_argument("--json", action="store_true", dest="json_output")
    bench.add_argument("--save", action="store_true")
    bench.add_argument("--roofline", action="store_true", help="run the Q8/BF16 Roofline test")
    roofline_action = bench.add_mutually_exclusive_group()
    roofline_action.add_argument("--capture", action="store_true", help="capture four GPU traces")
    roofline_action.add_argument("--clean", action="store_true", help="remove Roofline artifacts")

    commands.add_parser("leaderboard", help="regenerate BENCHMARKS.md from local results")
    return parser


def run_download(args: argparse.Namespace) -> int:
    print(resolve_model(args.model, args.cache_dir))
    return 0


def run_quantize(args: argparse.Namespace) -> int:
    from tinyinfer.quantization.convert import convert_checkpoint

    source = resolve_model(args.model, args.cache_dir)
    revision = source.name if source.parent.name == "snapshots" else None
    output = convert_checkpoint(
        source,
        args.output,
        source_model=args.model,
        source_revision=revision,
        format_name=args.format,
        group_size=args.group_size,
    )
    print(output)
    return 0


def run_generate(args: argparse.Namespace) -> int:
    from tinyinfer.engine import Engine
    from tinyinfer.tokenizer import Message

    model_dir = resolve_model(args.model, args.cache_dir)
    engine = Engine.from_pretrained(
        model_dir,
        device_name=args.device,
        dtype_name=args.dtype,
        decoding_name=args.decoding,
        attention_name=args.attention,
        kv_cache_name=args.kv_cache,
        quantization_name=getattr(args, "quantization", DEFAULT_QUANTIZATION),
        model_name=args.model,
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
        decoding_name=args.decoding,
        attention_name=args.attention,
        kv_cache_name=args.kv_cache,
        quantization_name=getattr(args, "quantization", DEFAULT_QUANTIZATION),
        model_name=args.model,
    )
    app = create_app(engine, args.model)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def run_chat(args: argparse.Namespace) -> int:
    from tinyinfer.chat import run_chat_session

    if not 1 <= args.max_tokens <= 512:
        print("max-tokens must be between 1 and 512", file=sys.stderr)
        return 2
    return run_chat_session(
        TinyInferClient(args.host),
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        output=sys.stdout,
    )


def run_bench(args: argparse.Namespace) -> int:
    if (getattr(args, "capture", False) or getattr(args, "clean", False)) and not getattr(
        args, "roofline", False
    ):
        print("--capture and --clean require --roofline", file=sys.stderr)
        return 2
    if getattr(args, "roofline", False):
        conflicts = [
            option
            for name, option, default in ROOFLINE_OPTION_DEFAULTS
            if getattr(args, name) != default
        ]
        if conflicts:
            print(
                f"--roofline cannot use model benchmark options: {', '.join(conflicts)}",
                file=sys.stderr,
            )
            return 2
        return run_roofline(args)

    from tinyinfer.benchmark import (
        benchmark,
        benchmark_options,
        format_summary,
        hardware_name,
        save_leaderboard_result,
    )
    from tinyinfer.engine import Engine
    from tinyinfer.tokenizer import Message

    try:
        options = benchmark_options(
            args.profile,
            prompt=args.prompt,
            system=args.system,
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    if options.warmup < 0 or options.repetitions < 1 or options.max_new_tokens < 1:
        print("warmup must be >= 0; repetitions and max-new-tokens must be >= 1", file=sys.stderr)
        return 2
    if args.save and not args.profile:
        print("--save requires --profile so leaderboard rows stay comparable", file=sys.stderr)
        return 2
    model_dir = resolve_model(args.model, args.cache_dir)
    engine = Engine.from_pretrained(
        model_dir,
        device_name=args.device,
        dtype_name=args.dtype,
        decoding_name=args.decoding,
        attention_name=args.attention,
        kv_cache_name=args.kv_cache,
        quantization_name=getattr(args, "quantization", DEFAULT_QUANTIZATION),
        model_name=args.model,
    )
    messages = [
        Message(role="system", content=options.system),
        Message(role="user", content=options.prompt),
    ]
    result = benchmark(
        engine,
        messages,
        max_new_tokens=options.max_new_tokens,
        warmup=options.warmup,
        repetitions=options.repetitions,
        metadata={
            "model": engine.source_model,
            "revision": engine.source_revision,
            "artifact_path": engine.artifact_path,
            "hardware": hardware_name(),
            "device": str(engine.device),
            "dtype": str(engine.activation_dtype),
            "quantization": engine.quantization_name,
            "decoding": engine.decoding_name,
            "attention": engine.attention_name,
            "profile": options.profile,
            "kv_cache": engine.kv_cache_name,
            "max_new_tokens": options.max_new_tokens,
            "warmup": options.warmup,
            "repetitions": options.repetitions,
        },
    )
    if args.save:
        try:
            save_leaderboard_result(result)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 2
    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        print(format_summary(result))
    return 0


def run_roofline(args: argparse.Namespace) -> int:
    from tinyinfer import roofline

    if args.clean:
        roofline.clean_artifacts()
        print(f"removed {roofline.ARTIFACT_DIR}")
        return 0

    if args.capture and os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        environment = {**os.environ, "MTL_CAPTURE_ENABLED": "1"}
        result = subprocess.run(
            [sys.executable, "-m", "tinyinfer", "benchmark", "--roofline", "--capture"],
            env=environment,
            check=False,
        )
        return result.returncode

    return roofline.run_default(capture=args.capture)


def run_leaderboard() -> int:
    from tinyinfer.benchmark import write_leaderboard

    try:
        print(write_leaderboard(), end="")
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if args.command == "download":
        return run_download(args)
    if args.command == "quantize":
        return run_quantize(args)
    if args.command == "generate":
        return run_generate(args)
    if args.command == "serve":
        return run_serve(args)
    if args.command == "chat":
        return run_chat(args)
    if args.command in {"bench", "benchmark"}:
        return run_bench(args)
    if args.command == "leaderboard":
        return run_leaderboard()
    parser.error(f"unknown command: {args.command}")
    return 2
