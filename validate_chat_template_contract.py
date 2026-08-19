#!/usr/bin/env python3
"""
Validate converted OpenAI-format JSONL against the chat-template contract
required by sagemaker-hyperpod-recipes' GPT-OSS SFT recipes:

- exactly one leading, non-empty system message
- a real (non-empty) user turn is present
- every tool_calls[].function.arguments is a parsed dict, not a string
- every role=tool message's tool_call_id matches a preceding tool_calls[].id
- tools is a sibling top-level array in OpenAI function-spec shape
- no leaked internal chat-template tokens
- rough token-length distribution vs a target context length (outliers flagged,
  not dropped)
"""

from __future__ import annotations

import argparse
import json

_TEMPLATE_TOKENS = [
    "<|begin_internal_thought|>", "<|end_internal_thought|>",
    "<|begin_of_solution|>", "<|end_of_solution|>",
    "<|begin_of_text|>", "<|end_of_text|>",
    "<|nova_user|>", "<|nova_assistant|>",
]


def _rough_token_count(record: dict) -> int:
    # Cheap chars/4 approximation -- good enough to flag gross outliers,
    # not a substitute for the real tokenizer.
    chars = 0
    for msg in record.get("messages", []):
        chars += len(msg.get("content") or "")
        for tc in msg.get("tool_calls", []):
            chars += len(json.dumps(tc))
    chars += len(json.dumps(record.get("tools", [])))
    return chars // 4


def validate_record(record: dict, max_len: int) -> list[str]:
    failures = []
    messages = record.get("messages")
    tools = record.get("tools")

    if not isinstance(messages, list) or not messages:
        return ["messages missing or empty"]

    if messages[0].get("role") != "system" or not (messages[0].get("content") or "").strip():
        failures.append("first message is not a non-empty system message")

    if not any(m.get("role") == "user" and (m.get("content") or "").strip() for m in messages):
        failures.append("no real (non-empty) user turn present")

    if tools is not None:
        if not isinstance(tools, list):
            failures.append("tools is present but not a list (must be a sibling top-level array)")
        else:
            for i, t in enumerate(tools):
                if not (isinstance(t, dict) and t.get("type") == "function" and "function" in t):
                    failures.append(f"tools[{i}] is not an OpenAI function-spec shape")

    call_ids = set()
    for i, msg in enumerate(messages):
        for tc in msg.get("tool_calls", []):
            call_ids.add(tc.get("id"))
            args = tc.get("function", {}).get("arguments")
            if not isinstance(args, dict):
                failures.append(f"messages[{i}].tool_calls arguments is {type(args).__name__}, not a dict")

        if msg.get("role") == "tool":
            tcid = msg.get("tool_call_id")
            if tcid not in call_ids:
                failures.append(f"messages[{i}] role=tool has tool_call_id={tcid!r} with no preceding tool_calls id")

        content = msg.get("content") or ""
        if any(tok in content for tok in _TEMPLATE_TOKENS):
            failures.append(f"messages[{i}] content contains a leaked template token")

    approx_tokens = _rough_token_count(record)
    if approx_tokens > max_len:
        failures.append(f"approx token count {approx_tokens} exceeds max_len {max_len} (outlier, not dropped)")

    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Converted OpenAI-format JSONL to validate")
    parser.add_argument("--max-len", type=int, default=16384, help="Target context length for outlier flagging")
    parser.add_argument("--show-failures", type=int, default=5, help="Number of example failing records to print")
    args = parser.parse_args()

    total = 0
    passed = 0
    failure_examples = []

    with open(args.input) as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            failures = validate_record(record, args.max_len)
            if failures:
                if len(failure_examples) < args.show_failures:
                    failure_examples.append((line_num, failures))
            else:
                passed += 1

    print(f"Total records:   {total}")
    print(f"Passed:          {passed}")
    print(f"Failed:          {total - passed}")
    if failure_examples:
        print("\nExample failures:")
        for line_num, failures in failure_examples:
            print(f"  line {line_num}: {failures}")


if __name__ == "__main__":
    main()
