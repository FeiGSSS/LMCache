# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


Message = dict[str, str]


class UserStateName(str, Enum):
    IDLE = "idle"
    SLEEPING = "sleeping"
    QUEUED = "queued"
    IN_FLIGHT = "in_flight"
    FINISHED = "finished"


@dataclass(slots=True)
class ConversationState:
    conversation_id: str
    messages: list[Message]
    next_user_message_index: int = 0
    last_used_ts: float = 0.0
    reuse_count: int = 0

    def is_exhausted(self) -> bool:
        return self.next_user_message_index >= len(self.messages)

    def current_messages(self) -> list[Message]:
        if self.is_exhausted():
            raise ValueError(f"Conversation {self.conversation_id} is exhausted")

        end_index = self.next_user_message_index + 1
        return [dict(message) for message in self.messages[:end_index]]

    def advance(self, finished_ts: float) -> None:
        if self.is_exhausted():
            raise ValueError(f"Conversation {self.conversation_id} is exhausted")

        self.last_used_ts = finished_ts
        self.reuse_count += 1
        self.next_user_message_index += 2

    def count_user_turns(self) -> int:
        return sum(1 for message in self.messages if message["role"] == "user")

    def mark_exhausted(self) -> None:
        self.next_user_message_index = len(self.messages)


@dataclass(slots=True)
class UserProfile:
    tier: str
    weight: float


@dataclass(slots=True)
class UserState:
    user_id: int
    profile: UserProfile
    conversations: list[ConversationState]
    state: UserStateName = UserStateName.IDLE
    last_conversation_id: str | None = None
    completed_requests: int = 0
    current_request_id: str | None = None

    def has_available_conversation(self) -> bool:
        return any(
            not conversation.is_exhausted() for conversation in self.conversations
        )

    def get_last_conversation(self) -> ConversationState | None:
        if self.last_conversation_id is None:
            return None

        for conversation in self.conversations:
            if conversation.conversation_id == self.last_conversation_id:
                return conversation

        return None

    def available_conversations(self) -> list[ConversationState]:
        return [
            conversation
            for conversation in self.conversations
            if not conversation.is_exhausted()
        ]


@dataclass(slots=True)
class QueuedRequest:
    request_id: str
    user_id: int
    conversation_id: str
    user_tier: str
    user_weight: float
    messages: list[Message]
    enqueue_ts: float
    selection_reason: str


@dataclass(slots=True)
class RequestResult:
    request_id: str
    user_id: int
    conversation_id: str
    success: bool
    ttft_ms: float
    latency_ms: float
    prompt_tokens: int
    cached_tokens: int
    generated_tokens: int
    queue_wait_ms: float
    selection_reason: str
    user_tier: str
    error: str | None = None


@dataclass(slots=True)
class RunSummary:
    total_requests: int
    succeeded_requests: int
    failed_requests: int
    runtime_sec: float
    requests_per_sec: float
    mean_ttft_ms: float
    p50_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float
    mean_prompt_tokens: float
    mean_cached_tokens: float
    skipped_overlong_conversations: int


@dataclass(slots=True)
class MetricsReport:
    summary: RunSummary
    by_tier: dict[str, dict[str, float]] = field(default_factory=dict)
    by_reason: dict[str, dict[str, float]] = field(default_factory=dict)
    prompt_buckets: dict[str, dict[str, float]] = field(default_factory=dict)
    details: list[dict[str, object]] = field(default_factory=list)
