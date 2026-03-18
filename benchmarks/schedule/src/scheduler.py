# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import random
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
        progress_interval_sec: float,
        prompt_validator,
        seed: int,
        rng,
    ) -> None:
        self.users = {user.user_id: user for user in users}
        self.ready_queue = ready_queue
        self.request_client = request_client
        self.max_parallel = max_parallel
        self.max_num_requests = max_num_requests
        self.request_rate_per_user = request_rate_per_user
        self.continue_prob = continue_prob
        self.progress_interval_sec = progress_interval_sec
        self.prompt_validator = prompt_validator
        self.seed = seed
        self.rng = rng
        self.stop_event = asyncio.Event()
        self.results: list[RequestResult] = []
        self.in_flight = 0
        self.skipped_overlong_conversations = 0
        self._lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task] = set()
        self._user_rngs = {
            user.user_id: random.Random(seed + user.user_id * 1009) for user in users
        }

    async def enqueue(self, request: QueuedRequest) -> None:
        async with self._lock:
            self.ready_queue.append(request)

    async def _start_user(self, user: UserState) -> None:
        await schedule_user_request(
            user=user,
            continue_prob=self.continue_prob,
            request_rate_per_user=self.request_rate_per_user,
            rng=self._user_rngs[user.user_id],
            prompt_validator=self.prompt_validator,
            enqueue_callback=self.enqueue,
            on_skip_conversation=self._record_skipped_conversation,
            stop_event=self.stop_event,
        )

    def _record_skipped_conversation(self) -> None:
        self.skipped_overlong_conversations += 1

    def _track_task(self, task: asyncio.Task) -> None:
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _launch_user(self, user: UserState) -> None:
        task = asyncio.create_task(self._start_user(user))
        self._track_task(task)

    async def _progress_reporter(self, start_time: float) -> None:
        while not self.stop_event.is_set():
            await asyncio.sleep(self.progress_interval_sec)
            elapsed = time.perf_counter() - start_time
            queued = len(self.ready_queue)
            completed = len(self.results)
            succeeded = sum(1 for result in self.results if result.success)
            print(
                "[进度] "
                f"已运行={elapsed:.1f}s "
                f"已完成={completed}/{self.max_num_requests} "
                f"成功={succeeded} "
                f"执行中={self.in_flight} "
                f"队列中={queued} "
                f"超长跳过={self.skipped_overlong_conversations}"
            )

    async def _execute_request(
        self,
        request: QueuedRequest,
    ) -> None:
        user = self.users[request.user_id]
        try:
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
        except Exception as exc:  # pragma: no cover - defensive scheduler guard
            self.results.append(
                RequestResult(
                    request_id=request.request_id,
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                    success=False,
                    ttft_ms=0.0,
                    latency_ms=0.0,
                    prompt_tokens=0,
                    cached_tokens=0,
                    generated_tokens=0,
                    queue_wait_ms=0.0,
                    selection_reason=request.selection_reason,
                    user_tier=request.user_tier,
                    error=str(exc),
                )
            )
        finally:
            self.in_flight -= 1
            user.current_request_id = None

            if len(self.results) >= self.max_num_requests:
                self.stop_event.set()

            if self.stop_event.is_set() or not user.has_available_conversation():
                user.state = UserStateName.FINISHED
                return

            user.state = UserStateName.IDLE
            self._launch_user(user)

    async def run(self) -> list[RequestResult]:
        start_time = time.perf_counter()
        if self.progress_interval_sec > 0:
            self._track_task(asyncio.create_task(self._progress_reporter(start_time)))

        for user in self.users.values():
            self._launch_user(user)

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
                self._track_task(task)

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

        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self.runtime_sec = time.perf_counter() - start_time
        return self.results
