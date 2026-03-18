# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import random
import time
import uuid

from .models import QueuedRequest, UserState, UserStateName


def choose_next_conversation(
    user: UserState,
    continue_prob: float,
    rng: random.Random,
) -> tuple[object | None, str]:
    available = user.available_conversations()
    if not available:
        user.state = UserStateName.FINISHED
        return None, "finished"

    last_conversation = user.get_last_conversation()
    if (
        last_conversation is not None
        and not last_conversation.is_exhausted()
        and rng.random() < continue_prob
    ):
        return last_conversation, "continue"

    revive_candidates = [
        conversation
        for conversation in available
        if conversation.conversation_id != user.last_conversation_id
    ]
    if revive_candidates:
        return rng.choice(revive_candidates), "revive"

    return available[0], "continue"


async def schedule_user_request(
    user: UserState,
    continue_prob: float,
    request_rate_per_user: float,
    rng: random.Random,
    enqueue_callback,
    stop_event: asyncio.Event,
) -> None:
    if stop_event.is_set() or not user.has_available_conversation():
        user.state = UserStateName.FINISHED
        return

    if request_rate_per_user > 0:
        user.state = UserStateName.SLEEPING
        await asyncio.sleep(rng.expovariate(request_rate_per_user))

    if stop_event.is_set() or not user.has_available_conversation():
        user.state = UserStateName.FINISHED
        return

    conversation, selection_reason = choose_next_conversation(
        user=user,
        continue_prob=continue_prob,
        rng=rng,
    )
    if conversation is None:
        user.state = UserStateName.FINISHED
        return

    request_id = str(uuid.uuid4())
    user.current_request_id = request_id
    user.last_conversation_id = conversation.conversation_id
    user.state = UserStateName.QUEUED

    request = QueuedRequest(
        request_id=request_id,
        user_id=user.user_id,
        conversation_id=conversation.conversation_id,
        user_tier=user.profile.tier,
        user_weight=user.profile.weight,
        messages=conversation.current_messages(),
        enqueue_ts=time.perf_counter(),
        selection_reason=selection_reason,
    )
    await enqueue_callback(request)
