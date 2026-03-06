#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
转换 ShareGPT 数据为 vllm benchmark 格式。

将原始 ShareGPT JSON 转换为:
[
    {"id": "conv_123", "messages": [{"role": "user", "content": "..."}, ...]},
    ...
]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_raw_data(input_path: str) -> list[dict[str, Any]]:
    """加载原始 ShareGPT 数据"""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"加载了 {len(data)} 条原始记录")
    return data


def merge_conversation_parts(
    data: list[dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    """
    合并同一对话的多片段。

    输入数据 ID 格式: "convID_index"，需要按 convID 合并。
    """
    # 按对话ID分组
    conversation_parts: dict[str, list[dict[str, str]]] = defaultdict(list)

    for item in data:
        raw_id = item["id"]
        # 分割 ID: "hRPPgZT_0" -> "hRPPgZT", "0"
        if "_" in raw_id:
            conv_id = raw_id.rsplit("_", 1)[0]
        else:
            conv_id = raw_id

        turns = item["conversations"]
        if not turns:
            continue

        # 如果上一片段的最后一条和当前片段的第一条来自同一人，跳过重复
        if conv_id in conversation_parts and conversation_parts[conv_id]:
            prev_turns = conversation_parts[conv_id][-1]
            if prev_turns and prev_turns.get("from") == turns[0].get("from"):
                turns = turns[1:]

        if turns:
            conversation_parts[conv_id].extend(turns)

    print(f"合并后得到 {len(conversation_parts)} 个对话")
    return conversation_parts


def convert_role(from_val: str) -> str | None:
    """转换 from 字段到 role 字段"""
    if from_val in {"human", "user"}:
        return "user"
    elif from_val in {"gpt", "bing", "chatgpt", "bard"}:
        return "assistant"
    elif from_val == "system":
        return "system"
    return None


def is_valid_conversation(messages: list[dict[str, str]]) -> bool:
    """验证对话是否有效"""
    if not messages:
        return False

    # 第一条必须是 user
    if messages[0].get("role") != "user":
        return False

    # 检查交替
    expected_role = "user"
    for msg in messages:
        if msg.get("role") != expected_role:
            return False
        expected_role = "assistant" if expected_role == "user" else "user"

    return True


def convert_to_openai_format(
    conversations: dict[str, list[dict[str, str]]],
    min_turns: int | None = None,
    max_turns: int | None = None,
) -> list[dict[str, Any]]:
    """转换为 OpenAI 格式"""
    result = []

    for conv_id, turns in conversations.items():
        # 过滤包含 system 的对话
        if any(t.get("from") == "system" for t in turns):
            continue

        # 转换 role
        messages = []
        for i, turn in enumerate(turns):
            role = convert_role(turn.get("from", ""))
            if role is None:
                continue
            if role == "system":
                # 跳过包含 system 的对话
                break
            messages.append({"role": role, "content": turn.get("value", "")})
        else:
            # 如果上面没有 break，继续处理
            pass

        # 检查是否因为 system 被跳过
        if any(m.get("role") == "system" for m in messages):
            continue

        # 限制轮数
        if max_turns is not None and len(messages) > max_turns:
            messages = messages[:max_turns]

        # 跳过短对话
        if min_turns is not None and len(messages) < min_turns:
            continue

        # 验证对话有效性
        if not is_valid_conversation(messages):
            continue

        result.append({"id": conv_id, "messages": messages})

    print(f"有效对话数: {len(result)}")
    return result


def print_stats(conversations: list[dict[str, Any]]) -> None:
    """打印统计信息"""
    if not conversations:
        return

    turn_counts = [len(c["messages"]) for c in conversations]
    print(f"\n=== 统计信息 ===")
    print(f"对话数: {len(conversations)}")
    print(
        f"消息数: min={min(turn_counts)}, max={max(turn_counts)}, avg={sum(turn_counts) / len(turn_counts):.1f}"
    )

    # 轮数分布 (user->assistant 为一轮)
    user_count = sum(
        1 for c in conversations for m in c["messages"] if m["role"] == "user"
    )
    print(f"总轮数 (user轮): {user_count}")


def select_conversations(
    conversations: list[dict[str, Any]],
    count: int | None,
    selection: str,
    seed: int,
) -> list[dict[str, Any]]:
    """按策略选择需要输出的对话子集。"""
    selected = list(conversations)

    if selection == "random":
        random.seed(seed)
        random.shuffle(selected)
    elif selection == "length_desc":
        selected.sort(key=lambda x: len(x.get("messages", [])), reverse=True)
    elif selection == "length_asc":
        selected.sort(key=lambda x: len(x.get("messages", [])))
    else:
        raise ValueError(f"Unsupported selection mode: {selection}")

    if count is not None:
        selected = selected[:count]

    return selected


def main():
    parser = argparse.ArgumentParser(description="转换 ShareGPT 数据为 vllm 格式")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/fei/research/datasets/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json",
        help="输入文件路径",
    )
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--min-turns", type=int, default=None, help="最小轮数")
    parser.add_argument("--max-turns", type=int, default=None, help="最大轮数")
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

    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be a positive integer")

    # 设置输出路径
    if args.output is None:
        script_dir = Path(__file__).parent
        args.output = str(script_dir / "sharegpt_conv.json")

    # 加载数据
    print(f"读取输入文件: {args.input}")
    raw_data = load_raw_data(args.input)

    # 合并多片段
    print("合并对话片段...")
    conversations = merge_conversation_parts(raw_data)

    # 转换格式
    print("转换格式...")
    result = convert_to_openai_format(
        conversations,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
    )

    # 按策略选择输出子集
    result = select_conversations(
        result,
        count=args.count,
        selection=args.selection,
        seed=args.seed,
    )

    # 打印统计
    print_stats(result)

    # 保存
    print(f"\n保存到: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("完成!")


if __name__ == "__main__":
    main()
