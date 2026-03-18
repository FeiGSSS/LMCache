# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time

from .models import QueuedRequest, RequestResult, UserState, UserStateName
from .queue import WeightedReadyQueue
from .request_client import RequestClient
from .user_actor import schedule_user_request


class BenchmarkScheduler:
    def __init__(
        self,
        users: list[UserState],
        ready_queue: WeightedReadyQueue,
        request_client: RequestClient,
        max_parallel: int,
        max_num_requests: int,
        request_rate_per_user: float,
        continue_prob: float,
        rng,
    ) -> None:
        self.users = {user.user_id: user for user in users}
        self.ready_queue = ready_queue
        self.request_client = request_client
        self.max_parallel = max_parallel
        self.max_num_requests = max_num_requests
        self.request_rate_per_user = request_rate_per_user
        self.continue_prob = continue_prob
        self.rng = rng
        self.stop_event = asyncio.Event()
        self.results: list[RequestResult] = []
        self.in_flight = 0
        self._lock = asyncio.Lock()
        self._running_tasks: set[asyncio.Task] = set()

    async def enqueue(self, request: QueuedRequest) -> None:
        async with self._lock:
            self.ready_queue.append(request)

    async def _start_user(self, user: UserState) -> None:
        await schedule_user_request(
            user=user,
            continue_prob=self.continue_prob,
            request_rate_per_user=self.request_rate_per_user,
            rng=self.rng,
            enqueue_callback=self.enqueue,
            stop_event=self.stop_event,
        )

    async def _execute_request(
        self,
        request: QueuedRequest,
    ) -> None:
        user = self.users[request.user_id]
        user.state = UserStateName.IN_FLIGHT
        queue_wait_ms = (time.perf_counter() - request.enqueue_ts) * 1000.0
        response = await self.request_client.send(messages=request.messages)

        result = RequestResult(
            request_id=request.request_id,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            success=bool(response["success"]),
            ttft_ms=float(response["ttft_ms"]),
            latency_ms=float(response["latency_ms"]),
            prompt_tokens=int(response["prompt_tokens"]),
            cached_tokens=int(response["cached_tokens"]),
            generated_tokens=int(response["generated_tokens"]),
            queue_wait_ms=queue_wait_ms,
            selection_reason=request.selection_reason,
            user_tier=request.user_tier,
            error=str(response["error"]) if response["error"] else None,
        )
        self.results.append(result)

        conversation = next(
            conversation
            for conversation in user.conversations
            if conversation.conversation_id == request.conversation_id
        )
        if result.success:
            conversation.advance(finished_ts=time.perf_counter())
            user.completed_requests += 1

        self.in_flight -= 1
        user.current_request_id = None

        if len(self.results) >= self.max_num_requests:
            self.stop_event.set()

        if self.stop_event.is_set() or not user.has_available_conversation():
            user.state = UserStateName.FINISHED
            return

        user.state = UserStateName.IDLE
        await self._start_user(user)

    async def run(self) -> list[RequestResult]:
        start_time = time.perf_counter()
        await asyncio.gather(*(self._start_user(user) for user in self.users.values()))

        while True:
            while (
                self.in_flight < self.max_parallel
                and not self.ready_queue.empty()
                and len(self.results) + self.in_flight < self.max_num_requests
            ):
                request = self.ready_queue.pop_weighted()
                if request is None:
                    break

                self.in_flight += 1
                task = asyncio.create_task(self._execute_request(request))
                self._running_tasks.add(task)
                task.add_done_callback(self._running_tasks.discard)

            all_done = all(
                user.state == UserStateName.FINISHED for user in self.users.values()
            )
            if (
                (self.stop_event.is_set() and self.in_flight == 0)
                or (all_done and self.in_flight == 0 and self.ready_queue.empty())
                or len(self.results) >= self.max_num_requests
            ):
                self.stop_event.set()
                break

            await asyncio.sleep(0.001)

        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)

        self.runtime_sec = time.perf_counter() - start_time
        return self.results
