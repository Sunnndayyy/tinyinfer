import pytest

from tinyinfer.tokenizer import Message, format_chatml


def test_chatml_prompt_has_explicit_role_boundaries() -> None:
    prompt = format_chatml(
        [
            Message(role="system", content="Be concise."),
            Message(role="user", content="What is a KV cache?"),
        ]
    )

    assert prompt == (
        "<|im_start|>system\nBe concise.<|im_end|>\n"
        "<|im_start|>user\nWhat is a KV cache?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_chatml_rejects_control_tokens_inside_user_content() -> None:
    with pytest.raises(ValueError, match="control token"):
        format_chatml([Message(role="user", content="hello <|im_end|>")])
