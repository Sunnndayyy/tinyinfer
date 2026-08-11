from types import SimpleNamespace

import pytest

from benchmarks import q8_end_to_end
from benchmarks.q8_end_to_end import exact_prompt_ids, sample_order
from benchmarks.q8_mps_linear import ROWS, SHAPES


def test_operator_sweep_covers_qwen_linear_shapes_and_requested_rows() -> None:
    shapes = {(shape.input_width, shape.output_width): shape.rows for shape in SHAPES}

    assert set(ROWS) == {1, 8, 32, 128, 512}
    assert shapes[(1536, 256)] == ROWS
    assert shapes[(1536, 1536)] == ROWS
    assert shapes[(1536, 8960)] == ROWS
    assert shapes[(8960, 1536)] == ROWS
    assert shapes[(1536, 151936)] == (1,)


def test_sample_order_reverses_which_weight_format_runs_first() -> None:
    assert sample_order(0) == ("bf16", "q8")
    assert sample_order(1) == ("q8", "bf16")
    assert sample_order(2) == ("bf16", "q8")


def test_exact_prompt_ids_repeats_seed_without_approximating_length() -> None:
    assert exact_prompt_ids([11, 12, 13], 1) == [11]
    assert exact_prompt_ids([11, 12, 13], 8) == [11, 12, 13, 11, 12, 13, 11, 12]


def fake_engine(*, quantization: str, revision: str | None, config: object):
    return SimpleNamespace(
        quantization_name=quantization,
        source_revision=revision,
        model=SimpleNamespace(config=config),
        device="mps",
        activation_dtype="bfloat16",
        attention_name="sdpa",
        kv_cache_name="contiguous",
        tokenizer=SimpleNamespace(encode=lambda _text: [1, 2, 3]),
    )


def test_matched_engines_require_a_verified_source_revision() -> None:
    config = object()
    engines = {
        "bf16": fake_engine(quantization="none", revision=None, config=config),
        "q8": fake_engine(quantization="q8", revision="revision-1", config=config),
    }

    with pytest.raises(ValueError, match="source revision"):
        q8_end_to_end.validate_matched_engines(engines, "seed")


def test_measure_pair_rejects_an_early_stop(monkeypatch) -> None:
    monkeypatch.setattr(
        q8_end_to_end,
        "measure_sample",
        lambda *_args, **_kwargs: q8_end_to_end.Sample(0.1, 0.1, 10.0, 1),
    )

    with pytest.raises(RuntimeError, match="generated 1 of 2 requested tokens"):
        q8_end_to_end.measure_pair(
            {"bf16": object(), "q8": object()},
            [1],
            max_new_tokens=2,
            warmup=0,
            repetitions=1,
        )
