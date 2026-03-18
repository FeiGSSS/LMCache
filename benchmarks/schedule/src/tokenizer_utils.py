# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from transformers import AutoTokenizer


class PromptLengthValidator:
    def __init__(self, model_path: str, max_prompt_tokens: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.max_prompt_tokens = max_prompt_tokens

    def get_prompt_token_count(self, messages: list[dict[str, str]]) -> int:
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(token_ids)

    def is_prompt_too_long(self, messages: list[dict[str, str]]) -> bool:
        return self.get_prompt_token_count(messages) > self.max_prompt_tokens
