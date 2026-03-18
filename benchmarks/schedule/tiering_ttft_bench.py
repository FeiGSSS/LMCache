#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import asdict

from src.config import build_parser, initialize_randomness
from src.dataset import assign_user_profiles, build_users, load_conversations
from src.metrics import build_report
from src.queue import WeightedReadyQueue
from src.request_client import RequestClient
from src.scheduler import BenchmarkScheduler


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rng = initialize_randomness(args.seed)

    conversations = load_conversations(
        input_file=args.input_file,
        min_user_turns=args.min_user_turns,
    )
    profiles = assign_user_profiles(
        num_users=args.num_users,
        vip_ratio=args.vip_ratio,
        active_ratio=args.active_ratio,
        vip_weight=args.vip_weight,
        active_weight=args.active_weight,
        normal_weight=args.normal_weight,
        rng=rng,
    )
    users = build_users(
        conversations=conversations,
        num_users=args.num_users,
        conversations_per_user=args.conversations_per_user,
        profiles=profiles,
        rng=rng,
    )

    scheduler = BenchmarkScheduler(
        users=users,
        ready_queue=WeightedReadyQueue(rng=rng),
        request_client=RequestClient(
            url=args.url,
            served_model_name=args.served_model_name,
            max_tokens=args.max_tokens,
            request_timeout_sec=args.request_timeout_sec,
            request_seed=args.seed,
        ),
        max_parallel=args.max_parallel,
        max_num_requests=args.max_num_requests,
        request_rate_per_user=args.request_rate_per_user,
        continue_prob=args.continue_prob,
        rng=rng,
    )

    import asyncio

    results = asyncio.run(scheduler.run())
    report = build_report(results=results, runtime_sec=scheduler.runtime_sec)

    print("-" * 80)
    print("Tiering TTFT Benchmark Summary")
    print("-" * 80)
    print(f"total_requests     : {report.summary.total_requests}")
    print(f"succeeded_requests : {report.summary.succeeded_requests}")
    print(f"failed_requests    : {report.summary.failed_requests}")
    print(f"runtime_sec        : {report.summary.runtime_sec:.3f}")
    print(f"requests_per_sec   : {report.summary.requests_per_sec:.3f}")
    print(f"mean_ttft_ms       : {report.summary.mean_ttft_ms:.3f}")
    print(f"p50_ttft_ms        : {report.summary.p50_ttft_ms:.3f}")
    print(f"p90_ttft_ms        : {report.summary.p90_ttft_ms:.3f}")
    print(f"p99_ttft_ms        : {report.summary.p99_ttft_ms:.3f}")
    print(f"mean_prompt_tokens : {report.summary.mean_prompt_tokens:.3f}")
    print(f"mean_cached_tokens : {report.summary.mean_cached_tokens:.3f}")

    if not args.print_summary_only:
        print("-" * 80)
        print("By Tier")
        for tier, stats in sorted(report.by_tier.items()):
            print(f"{tier}: {stats}")
        print("-" * 80)
        print("By Selection Reason")
        for reason, stats in sorted(report.by_reason.items()):
            print(f"{reason}: {stats}")
        print("-" * 80)
        print("By Prompt Bucket")
        for bucket, stats in sorted(report.prompt_buckets.items()):
            print(f"{bucket}: {stats}")

    if args.output_file is not None:
        with open(args.output_file, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "summary": asdict(report.summary),
                    "by_tier": report.by_tier,
                    "by_reason": report.by_reason,
                    "prompt_buckets": report.prompt_buckets,
                    "details": report.details,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()
