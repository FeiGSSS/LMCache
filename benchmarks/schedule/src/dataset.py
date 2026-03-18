# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import random
from pathlib import Path

from .models import ConversationState, Message, UserProfile, UserState


def _is_valid_messages(messages: list[Message]) -> bool:
    if not messages:
        return False

    expected_role = "user"
    for message in messages:
        if set(message.keys()) != {"role", "content"}:
            return False
        if message["role"] != expected_role:
            return False
        expected_role = "assistant" if expected_role == "user" else "user"

    return True


def _count_user_turns(messages: list[Message]) -> int:
    return sum(1 for message in messages if message["role"] == "user")


def load_conversations(
    input_file: str,
    min_user_turns: int,
) -> list[ConversationState]:
    """Load benchmark conversations from convert_sharegpt output."""
    input_path = Path(input_file)
    with input_path.open(encoding="utf-8") as file:
        raw_items = json.load(file)

    conversations: list[ConversationState] = []
    for item in raw_items:
        conversation_id = item["id"]
        messages = item["messages"]

        if not _is_valid_messages(messages):
            continue

        if _count_user_turns(messages) < min_user_turns:
            continue

        conversations.append(
            ConversationState(
                conversation_id=conversation_id,
                messages=[dict(message) for message in messages],
            )
        )

    return conversations


def assign_user_profiles(
    num_users: int,
    vip_ratio: float,
    active_ratio: float,
    vip_weight: float,
    active_weight: float,
    normal_weight: float,
    rng: random.Random,
) -> list[UserProfile]:
    if num_users <= 0:
        raise ValueError("num_users must be positive")

    vip_count = int(num_users * vip_ratio)
    active_count = int(num_users * active_ratio)
    normal_count = num_users - vip_count - active_count
    if normal_count < 0:
        raise ValueError("User tier ratios exceed 1.0")

    profiles = [UserProfile(tier="vip", weight=vip_weight) for _ in range(vip_count)]
    profiles.extend(
        UserProfile(tier="active", weight=active_weight) for _ in range(active_count)
    )
    profiles.extend(
        UserProfile(tier="normal", weight=normal_weight) for _ in range(normal_count)
    )
    rng.shuffle(profiles)
    return profiles


def build_users(
    conversations: list[ConversationState],
    num_users: int,
    conversations_per_user: int,
    profiles: list[UserProfile],
    rng: random.Random,
) -> list[UserState]:
    required = num_users * conversations_per_user
    if len(conversations) < required:
        raise ValueError(
            "Not enough conversations after filtering: "
            f"need {required}, got {len(conversations)}"
        )

    shuffled_conversations = list(conversations)
    rng.shuffle(shuffled_conversations)

    users: list[UserState] = []
    for user_id in range(num_users):
        start = user_id * conversations_per_user
        end = start + conversations_per_user
        user_conversations = shuffled_conversations[start:end]
        users.append(
            UserState(
                user_id=user_id,
                profile=profiles[user_id],
                conversations=user_conversations,
            )
        )

    return users
