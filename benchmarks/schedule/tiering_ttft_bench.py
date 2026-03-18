#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import unicodedata
from dataclasses import asdict

from src.config import build_parser, initialize_randomness
from src.dataset import assign_user_profiles, build_users, load_conversations
from src.metrics import build_report
from src.queue import WeightedReadyQueue
from src.request_client import RequestClient
from src.scheduler import BenchmarkScheduler
from src.tokenizer_utils import PromptLengthValidator


def _print_section(title: str) -> None:
    print("-" * 80)
    print(title)


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _pad_display(text: str, width: int) -> str:
    padding = max(0, width - _display_width(text))
    return text + (" " * padding)


def _print_kv(label: str, value: object) -> None:
    print(f"{_pad_display(label, 24)} {value}")


def _print_group_table(title: str, rows: dict[str, dict[str, float]]) -> None:
    _print_section(title)
    group_header = _pad_display("分组", 14)
    print(f"{group_header} {'数量':>8} {'平均TTFT(ms)':>16} {'P90 TTFT(ms)':>16}")
    for key, stats in sorted(rows.items()):
        count = int(stats.get("count", 0.0))
        mean_ttft = stats.get("mean_ttft_ms", 0.0)
        p90_ttft = stats.get("p90_ttft_ms", 0.0)
        group_name = _pad_display(key, 14)
        print(f"{group_name} {count:>8} {mean_ttft:>14.3f} {p90_ttft:>14.3f}")


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
    prompt_validator = PromptLengthValidator(
        model_path=args.model,
        max_prompt_tokens=args.max_model_len - args.max_tokens,
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
        progress_interval_sec=args.progress_interval_sec,
        prompt_validator=prompt_validator,
        seed=args.seed,
        rng=rng,
    )

    import asyncio

    results = asyncio.run(scheduler.run())
    report = build_report(
        results=results,
        runtime_sec=scheduler.runtime_sec,
        skipped_overlong_conversations=scheduler.skipped_overlong_conversations,
    )

    _print_section("多层调度 TTFT Benchmark 汇总")
    _print_kv("总请求数", report.summary.total_requests)
    _print_kv("成功请求数", report.summary.succeeded_requests)
    _print_kv("失败请求数", report.summary.failed_requests)
    _print_kv("总运行时间(秒)", f"{report.summary.runtime_sec:.3f}")
    _print_kv("吞吐(req/s)", f"{report.summary.requests_per_sec:.3f}")
    _print_kv("平均TTFT(ms)", f"{report.summary.mean_ttft_ms:.3f}")
    _print_kv("P50 TTFT(ms)", f"{report.summary.p50_ttft_ms:.3f}")
    _print_kv("P90 TTFT(ms)", f"{report.summary.p90_ttft_ms:.3f}")
    _print_kv("P99 TTFT(ms)", f"{report.summary.p99_ttft_ms:.3f}")
    _print_kv("平均Prompt Tokens", f"{report.summary.mean_prompt_tokens:.3f}")
    _print_kv("平均缓存Tokens", f"{report.summary.mean_cached_tokens:.3f}")
    _print_kv("超长跳过会话数", report.summary.skipped_overlong_conversations)

    if not args.print_summary_only:
        _print_group_table("按用户档位统计", report.by_tier)
        _print_group_table("按会话选择方式统计", report.by_reason)
        _print_group_table("按Prompt长度分桶统计", report.prompt_buckets)

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
