"""Shared prompt rendering for eval and SFT data generation.

Centralizes the chat-template-vs-base-model fallback so eval.py and
scripts/generate_sft_data.py format prompts identically.

For tokenizers with a chat_template (Qwen, Llama-Instruct, ...) we preserve
the existing apply_chat_template calls byte-for-byte. For base models with no
chat_template (gemma-3-270m base, ...) we fall back to the raw template_type=
'base' format that the parquets already contain — the tokenizer auto-prepends
its own BOS at encode time.
"""

ASSISTANT_PARTIAL_TURN = "Let me solve this step by step.\n<think>"


def render_prompt(tokenizer, prompt, mode):
    """Return the prompt string vLLM should generate from.

    prompt: list of {"role","content"} dicts OR a plain string (wrapped as a
            single user message).
    mode:   "eval"    — continue the final message (matches old eval.py).
            "sft_gen" — open a new assistant turn and append the standard
                        partial-assistant-turn suffix (matches old
                        generate_sft_data.py).
    """
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)

    if getattr(tokenizer, "chat_template", None):
        if mode == "eval":
            return tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        if mode == "sft_gen":
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            return text + ASSISTANT_PARTIAL_TURN
        raise ValueError(f"Unknown mode: {mode!r}")

    # Base / no-chat-template fallback. The parquet's content is already in the
    # repo's template_type='base' format and ends with the assistant partial
    # turn, so no chat wrapping and no suffix re-append.
    return "".join(m["content"] for m in messages)
