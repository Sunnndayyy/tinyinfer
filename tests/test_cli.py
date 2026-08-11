import json
import subprocess
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from tinyinfer import benchmark, chat, cli
from tinyinfer.cli import (
    build_parser,
    run_bench,
    run_chat,
    run_generate,
    run_leaderboard,
    run_serve,
)


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
    assert args.attention == "eager"
    assert args.quantization == "auto"


@pytest.mark.parametrize("quantization", ["none", "q8", "q4"])
def test_model_commands_accept_each_quantization(quantization: str) -> None:
    args = build_parser().parse_args(["bench", "--quantization", quantization])

    assert args.quantization == quantization


@pytest.mark.parametrize("cache_name", ["none", "contiguous", "paged"])
def test_model_commands_accept_each_kv_cache(cache_name: str) -> None:
    args = build_parser().parse_args(["bench", "--kv-cache", cache_name])

    assert args.kv_cache == cache_name


def test_model_commands_accept_autoregressive_decoding() -> None:
    args = build_parser().parse_args(["bench", "--decoding", "autoregressive"])

    assert args.decoding == "autoregressive"


@pytest.mark.parametrize("attention_name", ["eager", "sdpa"])
def test_model_commands_accept_each_attention_implementation(attention_name: str) -> None:
    args = build_parser().parse_args(["bench", "--attention", attention_name])

    assert args.attention == attention_name


def test_benchmark_can_save_a_named_profile() -> None:
    args = build_parser().parse_args(
        ["bench", "--profile", "decode", "--kv-cache", "contiguous", "--save"]
    )

    assert args.profile == "decode"
    assert args.save is True


def test_benchmark_alias_exposes_the_opinionated_roofline_command() -> None:
    args = build_parser().parse_args(["benchmark", "--roofline", "--capture"])

    assert args.command == "benchmark"
    assert args.roofline is True
    assert args.capture is True


def test_benchmark_alias_dispatches_the_roofline_command(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "tinyinfer.roofline.run_default",
        lambda *, capture: calls.append(capture) or 0,
    )

    assert cli.main(["benchmark", "--roofline"]) == 0
    assert calls == [False]


@pytest.mark.parametrize(
    "arguments",
    (
        ["model-id"],
        ["--warmup", "1"],
        ["--device", "cpu"],
        ["--json"],
        ["--save"],
    ),
)
def test_roofline_rejects_model_benchmark_options(arguments, capsys) -> None:
    args = build_parser().parse_args(["benchmark", *arguments, "--roofline"])

    assert run_bench(args) == 2
    assert "--roofline cannot use model benchmark options" in capsys.readouterr().err


def test_module_invocation_propagates_cli_failure_status() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tinyinfer", "benchmark", "--capture"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stderr == "--capture and --clean require --roofline\n"


def test_roofline_command_runs_the_default_experiment(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "tinyinfer.roofline.run_default",
        lambda *, capture: calls.append(capture) or 0,
    )

    args = build_parser().parse_args(["benchmark", "--roofline"])

    assert run_bench(args) == 0
    assert calls == [False]


def test_roofline_cleanup_removes_only_its_artifact_directory(tmp_path, monkeypatch) -> None:
    roofline_dir = tmp_path / ".tinyinfer" / "roofline"
    roofline_dir.mkdir(parents=True)
    (roofline_dir / "results.json").write_text("result")
    sibling = tmp_path / ".tinyinfer" / "benchmarks.json"
    sibling.write_text("benchmark")
    monkeypatch.chdir(tmp_path)

    args = build_parser().parse_args(["benchmark", "--roofline", "--clean"])

    assert run_bench(args) == 0
    assert not roofline_dir.exists()
    assert sibling.read_text() == "benchmark"


def test_roofline_capture_restarts_with_metal_capture_enabled(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MTL_CAPTURE_ENABLED", raising=False)
    monkeypatch.setattr(
        "tinyinfer.cli.subprocess.run",
        lambda command, **options: (
            calls.append((command, options)) or SimpleNamespace(returncode=0)
        ),
    )

    args = build_parser().parse_args(["benchmark", "--roofline", "--capture"])

    assert run_bench(args) == 0
    command, options = calls[0]
    assert command[-3:] == ["benchmark", "--roofline", "--capture"]
    assert options["env"]["MTL_CAPTURE_ENABLED"] == "1"


def test_roofline_capture_uses_the_fixed_capture_directory(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MTL_CAPTURE_ENABLED", "1")
    monkeypatch.setattr(
        "tinyinfer.roofline.run_default",
        lambda *, capture: calls.append(capture) or 0,
    )

    args = build_parser().parse_args(["benchmark", "--roofline", "--capture"])

    assert run_bench(args) == 0
    assert calls == [True]


@pytest.mark.parametrize("option", ["--capture", "--clean"])
def test_roofline_actions_require_roofline_mode(option, capsys) -> None:
    args = build_parser().parse_args(["benchmark", option])

    assert run_bench(args) == 2
    assert capsys.readouterr().err == "--capture and --clean require --roofline\n"


@pytest.mark.parametrize(
    ("command", "quantization", "runner"),
    [
        ("generate", "none", run_generate),
        ("serve", "q8", run_serve),
        ("bench", "q4", run_bench),
    ],
)
def test_model_runners_forward_explicit_quantization(
    command, quantization, runner, tmp_path, monkeypatch
) -> None:
    engine = SimpleNamespace(
        stream=lambda *_args, **_kwargs: (),
        source_model="Tiny/Qwen",
        source_revision=None,
        artifact_path=str(tmp_path),
        device=torch.device("cpu"),
        activation_dtype=torch.float32,
        quantization_name=quantization,
        decoding_name="autoregressive",
        attention_name="eager",
        kv_cache_name="contiguous",
    )
    load_options = {}
    monkeypatch.setattr(
        "tinyinfer.engine.Engine.from_pretrained",
        lambda *args, **kwargs: load_options.update(kwargs) or engine,
    )
    monkeypatch.setattr("tinyinfer.server.create_app", lambda *_args: object())
    monkeypatch.setattr("uvicorn.run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark, "hardware_name", lambda: "Test hardware")
    monkeypatch.setattr(benchmark, "benchmark", lambda *_args, **_kwargs: {})

    arguments = [command, str(tmp_path), "--quantization", quantization]
    if command == "generate":
        arguments.extend(("--prompt", "hello"))
    elif command == "bench":
        arguments.append("--json")

    assert runner(build_parser().parse_args(arguments)) == 0
    assert load_options["quantization_name"] == quantization


def test_benchmark_json_keeps_activation_and_weight_identity_separate(
    tmp_path, monkeypatch, capsys
) -> None:
    engine = SimpleNamespace(
        source_model="Tiny/Qwen",
        source_revision="revision-1",
        artifact_path=str(tmp_path),
        device=torch.device("cpu"),
        activation_dtype=torch.bfloat16,
        quantization_name="q8",
        decoding_name="autoregressive",
        attention_name="eager",
        kv_cache_name="contiguous",
    )
    load_options = {}
    monkeypatch.setattr(
        "tinyinfer.engine.Engine.from_pretrained",
        lambda *args, **kwargs: load_options.update(kwargs) or engine,
    )
    monkeypatch.setattr(benchmark, "hardware_name", lambda: "Test hardware")
    monkeypatch.setattr(
        benchmark,
        "benchmark",
        lambda engine, messages, **kwargs: {"metadata": kwargs["metadata"]},
    )
    args = build_parser().parse_args(["bench", str(tmp_path), "--json"])
    del args.quantization  # Direct Namespace callers from before U5 still default to auto.

    assert run_bench(args) == 0

    metadata = json.loads(capsys.readouterr().out)["metadata"]
    assert metadata["model"] == "Tiny/Qwen"
    assert metadata["revision"] == "revision-1"
    assert metadata["artifact_path"] == str(tmp_path)
    assert metadata["dtype"] == "torch.bfloat16"
    assert metadata["quantization"] == "q8"
    assert load_options["quantization_name"] == "auto"


def test_leaderboard_command_needs_no_arguments() -> None:
    args = build_parser().parse_args(["leaderboard"])

    assert args.command == "leaderboard"


def test_quantize_command_accepts_a_q8_output_directory() -> None:
    args = build_parser().parse_args(
        ["quantize", "Tiny/Qwen", "--format", "q8", "--output", "model-q8"]
    )

    assert args.command == "quantize"
    assert args.model == "Tiny/Qwen"
    assert args.format == "q8"
    assert args.output == "model-q8"
    assert args.group_size == 32


def test_quantize_command_rejects_an_unsupported_group_size() -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(
            ["quantize", "Tiny/Qwen", "--output", "model-q8", "--group-size", "64"]
        )

    assert error.value.code == 2


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
