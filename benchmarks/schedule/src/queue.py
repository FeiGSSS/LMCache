# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random

from .models import QueuedRequest


class WeightedReadyQueue:
    """In-memory weighted queue for ready requests."""

    def __init__(self, rng: random.Random):
        self._rng = rng
        self._requests: list[QueuedRequest] = []

    def __len__(self) -> int:
        return len(self._requests)

    def empty(self) -> bool:
        return not self._requests

    def append(self, request: QueuedRequest) -> None:
        self._requests.append(request)

    def pop_weighted(self) -> QueuedRequest | None:
        if not self._requests:
            return None

        if len(self._requests) == 1:
            return self._requests.pop()

        weights = [max(request.user_weight, 0.0) for request in self._requests]
        if sum(weights) <= 0:
            index = self._rng.randrange(len(self._requests))
            return self._requests.pop(index)

        index = self._rng.choices(range(len(self._requests)), weights=weights, k=1)[0]
        return self._requests.pop(index)
