# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from statistics import mean

from .models import MetricsReport, RequestResult, RunSummary


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0

    index = min(len(values) - 1, max(0, math.ceil(len(values) * ratio) - 1))
    return sorted(values)[index]


def _group_summary(items: list[RequestResult]) -> dict[str, float]:
    ttfts = [item.ttft_ms for item in items if item.success]
    if not ttfts:
        return {
            "count": float(len(items)),
            "mean_ttft_ms": 0.0,
            "p90_ttft_ms": 0.0,
        }

    return {
        "count": float(len(items)),
        "mean_ttft_ms": mean(ttfts),
        "p90_ttft_ms": _percentile(ttfts, 0.9),
    }


def _prompt_bucket(prompt_tokens: int) -> str:
    if prompt_tokens < 1024:
        return "<1k"
    if prompt_tokens < 4096:
        return "1k-4k"
    if prompt_tokens < 8192:
        return "4k-8k"
    return ">=8k"


def build_report(
    results: list[RequestResult],
    runtime_sec: float,
    skipped_overlong_conversations: int = 0,
) -> MetricsReport:
    successful = [result for result in results if result.success]
    ttfts = [result.ttft_ms for result in successful]
    prompt_tokens = [result.prompt_tokens for result in successful]
    cached_tokens = [result.cached_tokens for result in successful]

    summary = RunSummary(
        total_requests=len(results),
        succeeded_requests=len(successful),
        failed_requests=len(results) - len(successful),
        runtime_sec=runtime_sec,
        requests_per_sec=(len(successful) / runtime_sec) if runtime_sec > 0 else 0.0,
        mean_ttft_ms=mean(ttfts) if ttfts else 0.0,
        p50_ttft_ms=_percentile(ttfts, 0.5),
        p90_ttft_ms=_percentile(ttfts, 0.9),
        p99_ttft_ms=_percentile(ttfts, 0.99),
        mean_prompt_tokens=mean(prompt_tokens) if prompt_tokens else 0.0,
        mean_cached_tokens=mean(cached_tokens) if cached_tokens else 0.0,
        skipped_overlong_conversations=skipped_overlong_conversations,
    )

    by_tier: dict[str, list[RequestResult]] = {}
    by_reason: dict[str, list[RequestResult]] = {}
    by_bucket: dict[str, list[RequestResult]] = {}
    details: list[dict[str, object]] = []

    for result in results:
        by_tier.setdefault(result.user_tier, []).append(result)
        by_reason.setdefault(result.selection_reason, []).append(result)
        by_bucket.setdefault(_prompt_bucket(result.prompt_tokens), []).append(result)
        details.append(
            {
                "request_id": result.request_id,
                "user_id": result.user_id,
                "conversation_id": result.conversation_id,
                "user_tier": result.user_tier,
                "selection_reason": result.selection_reason,
                "success": result.success,
                "ttft_ms": result.ttft_ms,
                "latency_ms": result.latency_ms,
                "prompt_tokens": result.prompt_tokens,
                "cached_tokens": result.cached_tokens,
                "generated_tokens": result.generated_tokens,
                "queue_wait_ms": result.queue_wait_ms,
                "error": result.error,
            }
        )

    return MetricsReport(
        summary=summary,
        by_tier={key: _group_summary(value) for key, value in by_tier.items()},
        by_reason={key: _group_summary(value) for key, value in by_reason.items()},
        prompt_buckets={key: _group_summary(value) for key, value in by_bucket.items()},
        details=details,
    )
