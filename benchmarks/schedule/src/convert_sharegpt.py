# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_raw_data(input_path: str) -> list[dict[str, Any]]:
    """Load raw ShareGPT data."""
    with open(input_path, encoding="utf-8") as file:
        data = json.load(file)
    print(f"加载了 {len(data)} 条原始记录")
    return data


def merge_conversation_parts(
    data: list[dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    """Merge conversation shards with the same conversation id."""
    conversation_parts: dict[str, list[dict[str, str]]] = defaultdict(list)

    for item in data:
        raw_id = item["id"]
        conv_id = raw_id.rsplit("_", 1)[0] if "_" in raw_id else raw_id

        turns = item["conversations"]
        if not turns:
            continue

        if conv_id in conversation_parts and conversation_parts[conv_id]:
            prev_turn = conversation_parts[conv_id][-1]
            if prev_turn and prev_turn.get("from") == turns[0].get("from"):
                turns = turns[1:]

        if turns:
            conversation_parts[conv_id].extend(turns)

    print(f"合并后得到 {len(conversation_parts)} 个对话")
    return conversation_parts


def convert_role(from_val: str) -> str | None:
    """Convert ShareGPT role names to OpenAI role names."""
    if from_val in {"human", "user"}:
        return "user"
    if from_val in {"gpt", "bing", "chatgpt", "bard"}:
        return "assistant"
    if from_val == "system":
        return "system"
    return None


def is_valid_conversation(messages: list[dict[str, str]]) -> bool:
    """Validate a user/assistant alternating conversation."""
    if not messages:
        return False

    if messages[0].get("role") != "user":
        return False

    expected_role = "user"
    for message in messages:
        if message.get("role") != expected_role:
            return False
        expected_role = "assistant" if expected_role == "user" else "user"

    return True


def count_user_turns(messages: list[dict[str, str]]) -> int:
    """Count user turns in converted messages."""
    return sum(1 for message in messages if message.get("role") == "user")


def convert_to_openai_format(
    conversations: dict[str, list[dict[str, str]]],
    min_turns: int | None = None,
    max_turns: int | None = None,
    min_user_turns: int | None = None,
) -> list[dict[str, Any]]:
    """Convert merged ShareGPT conversations to benchmark format."""
    result = []

    for conv_id, turns in conversations.items():
        if any(turn.get("from") == "system" for turn in turns):
            continue

        messages = []
        for turn in turns:
            role = convert_role(turn.get("from", ""))
            if role is None:
                continue
            if role == "system":
                break
            messages.append({"role": role, "content": turn.get("value", "")})

        if any(message.get("role") == "system" for message in messages):
            continue

        if max_turns is not None and len(messages) > max_turns:
            messages = messages[:max_turns]

        if min_turns is not None and len(messages) < min_turns:
            continue

        if min_user_turns is not None and count_user_turns(messages) < min_user_turns:
            continue

        if not is_valid_conversation(messages):
            continue

        result.append({"id": conv_id, "messages": messages})

    print(f"有效对话数: {len(result)}")
    return result


def print_stats(conversations: list[dict[str, Any]]) -> None:
    """Print summary stats for converted conversations."""
    if not conversations:
        return

    turn_counts = [len(conversation["messages"]) for conversation in conversations]
    user_turn_counts = [
        count_user_turns(conversation["messages"]) for conversation in conversations
    ]
    print("\n=== 统计信息 ===")
    print(f"对话数: {len(conversations)}")
    print(
        f"消息数: min={min(turn_counts)}, max={max(turn_counts)}, avg={sum(turn_counts) / len(turn_counts):.1f}"
    )
    print(
        f"用户轮数: min={min(user_turn_counts)}, max={max(user_turn_counts)}, avg={sum(user_turn_counts) / len(user_turn_counts):.1f}"
    )
    print(f"总轮数 (user轮): {sum(user_turn_counts)}")


def select_conversations(
    conversations: list[dict[str, Any]],
    count: int | None,
    selection: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a subset of conversations by strategy."""
    selected = list(conversations)

    if selection == "random":
        rng = random.Random(seed)
        rng.shuffle(selected)
    elif selection == "length_desc":
        selected.sort(key=lambda item: len(item.get("messages", [])), reverse=True)
    elif selection == "length_asc":
        selected.sort(key=lambda item: len(item.get("messages", [])))
    else:
        raise ValueError(f"Unsupported selection mode: {selection}")

    if count is not None:
        selected = selected[:count]

    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="转换 ShareGPT 数据为 vllm 格式")
    parser.add_argument(
        "--input",
        type=str,
        default="/data/llm-datasets/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json",
        help="输入文件路径",
    )
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--min-turns", type=int, default=None, help="最小消息数")
    parser.add_argument("--max-turns", type=int, default=None, help="最大消息数")
    parser.add_argument(
        "--min-user-turns",
        type=int,
        default=None,
        help="最小 user turn 数，用于保留更长的多轮对话",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="输出对话数量，默认 200",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="length_desc",
        choices=["random", "length_desc", "length_asc"],
        help="对话选择策略，默认按长度降序",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be a positive integer")

    if args.output is None:
        script_dir = Path(__file__).resolve().parent.parent
        args.output = str(script_dir / "dataset" / "sharegpt_conv.json")

    print(f"读取输入文件: {args.input}")
    raw_data = load_raw_data(args.input)

    print("合并对话片段...")
    conversations = merge_conversation_parts(raw_data)

    print("转换格式...")
    result = convert_to_openai_format(
        conversations,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        min_user_turns=args.min_user_turns,
    )

    result = select_conversations(
        result,
        count=args.count,
        selection=args.selection,
        seed=args.seed,
    )

    print_stats(result)

    print(f"\n保存到: {args.output}")
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("完成!")


if __name__ == "__main__":
    main()
