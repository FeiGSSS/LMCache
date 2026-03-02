#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ShareGPT multi-turn concurrent TTFT benchmark (Phase-1).

Phase-1 constraints:
- Only TTFT is measured.
- The tested model must generate exactly 1 token.
- Assistant history is replayed from ShareGPT reference answers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import argparse
import asyncio
import csv
import heapq
import json
import math
import random
import sys
import time
import uuid


USER_ROLES = {"human", "user"}
ASSISTANT_ROLES = {"assistant", "gpt", "chatgpt", "bing", "bard"}
SYSTEM_ROLES = {"system"}

TURN_BUCKETS = ("turn_1", "turn_2_3", "turn_4_plus")
NEXT_TURN_PRIORITY = 0
NEW_SESSION_PRIORITY = 1


@dataclass
class ConversationTurn:
    user_text: str
    assistant_ref_text: str
    assistant_ref_tokens: int


@dataclass
class ConversationSession:
    session_id: int
    conversation_id: str
    turns: list[ConversationTurn]
    next_turn_idx: int
    finished: bool


class DispatchEventType(str, Enum):
    NEW_SESSION = "NEW_SESSION"
    NEXT_TURN = "NEXT_TURN"


@dataclass
class DispatchEvent:
    event_time: float
    event_type: DispatchEventType
    session_id: Optional[int]
    priority: int


@dataclass
class RequestRecord:
    run_id: str
    scenario: str
    seed: int
    session_id: int
    conversation_id: str
    turn_index: int
    ready_time: float
    dispatch_time: Optional[float]
    first_token_time: Optional[float]
    ttft_server_sec: Optional[float]
    ttft_effective_sec: Optional[float]
    client_queue_wait_sec: Optional[float]
    prompt_tokens_est: int
    status: str
    error: str


def estimate_tokens(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    words = len(text.split())
    if words <= 1:
        return max(1, int(len(text) / 2.5))
    return max(1, int(words * 1.3))


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(m["content"]) for m in messages)


def normalize_role(role: str) -> str:
    role = role.lower().strip()
    if role in USER_ROLES:
        return "user"
    if role in ASSISTANT_ROLES:
        return "assistant"
    if role in SYSTEM_ROLES:
        return "system"
    return "unknown"


def parse_sharegpt_turns(conversations: list[dict[str, Any]]) -> list[ConversationTurn]:
    turns: list[ConversationTurn] = []
    pending_user: Optional[str] = None

    for item in conversations:
        role = normalize_role(str(item.get("from", "")))
        value = str(item.get("value", ""))
        if role == "system":
            continue
        if role == "user":
            pending_user = value
            continue
        if role == "assistant" and pending_user is not None:
            turns.append(
                ConversationTurn(
                    user_text=pending_user,
                    assistant_ref_text=value,
                    assistant_ref_tokens=estimate_tokens(value),
                )
            )
            pending_user = None
    return turns


def load_sharegpt_sessions(sharegpt_path: str) -> list[tuple[str, list[ConversationTurn]]]:
    with open(sharegpt_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    sessions: list[tuple[str, list[ConversationTurn]]] = []
    for idx, row in enumerate(raw):
        conv_id = str(row.get("id", f"conv_{idx}"))
        conversations = row.get("conversations")
        if not isinstance(conversations, list):
            continue
        turns = parse_sharegpt_turns(conversations)
        if len(turns) < 2:
            continue
        sessions.append((conv_id, turns))
    return sessions


def build_messages_for_turn(
    session: ConversationSession, turn_idx: int, max_context_tokens: int
) -> tuple[list[dict[str, str]], int]:
    history_blocks: list[list[dict[str, str]]] = []
    for i in range(turn_idx):
        t = session.turns[i]
        history_blocks.append(
            [
                {"role": "user", "content": t.user_text},
                {"role": "assistant", "content": t.assistant_ref_text},
            ]
        )
    current_user = {"role": "user", "content": session.turns[turn_idx].user_text}

    def flatten(blocks: list[list[dict[str, str]]]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for block in blocks:
            out.extend(block)
        return out

    blocks = list(history_blocks)
    while True:
        messages = flatten(blocks)
        messages.append(current_user)
        prompt_tokens_est = estimate_message_tokens(messages)
        if prompt_tokens_est <= max_context_tokens or not blocks:
            return messages, prompt_tokens_est
        blocks.pop(0)


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    q = min(max(q, 0.0), 1.0)
    rank = (len(data) - 1) * q
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return data[low]
    frac = rank - low
    return data[low] + (data[high] - data[low]) * frac


def turn_bucket(turn_index: int) -> str:
    if turn_index == 1:
        return "turn_1"
    if turn_index in (2, 3):
        return "turn_2_3"
    return "turn_4_plus"


def summarize_records(records: list[RequestRecord], benchmark_time_sec: float) -> dict[str, Any]:
    total = len(records)
    success = [r for r in records if r.status == "success"]
    ttft_server = [r.ttft_server_sec for r in success if r.ttft_server_sec is not None]
    ttft_effective = [
        r.ttft_effective_sec for r in success if r.ttft_effective_sec is not None
    ]
    queue_wait = [
        r.client_queue_wait_sec
        for r in records
        if r.client_queue_wait_sec is not None
    ]

    by_turn_index: dict[str, dict[str, Any]] = {}
    for bucket in TURN_BUCKETS:
        bucket_rows = [r for r in records if turn_bucket(r.turn_index) == bucket]
        bucket_success = [r for r in bucket_rows if r.status == "success"]
        bucket_ttft = [
            r.ttft_server_sec
            for r in bucket_success
            if r.ttft_server_sec is not None
        ]
        by_turn_index[bucket] = {
            "count": len(bucket_rows),
            "success_rate": (
                len(bucket_success) / len(bucket_rows) if bucket_rows else 0.0
            ),
            "ttft_server_p50": percentile(bucket_ttft, 0.5),
            "ttft_server_p95": percentile(bucket_ttft, 0.95),
        }

    return {
        "total_requests": total,
        "success_rate": len(success) / total if total else 0.0,
        "throughput_rps": total / max(benchmark_time_sec, 1e-9),
        "ttft_server_p50": percentile(ttft_server, 0.5),
        "ttft_server_p90": percentile(ttft_server, 0.9),
        "ttft_server_p95": percentile(ttft_server, 0.95),
        "ttft_server_p99": percentile(ttft_server, 0.99),
        "ttft_effective_p50": percentile(ttft_effective, 0.5),
        "ttft_effective_p90": percentile(ttft_effective, 0.9),
        "ttft_effective_p95": percentile(ttft_effective, 0.95),
        "ttft_effective_p99": percentile(ttft_effective, 0.99),
        "client_queue_wait_p50": percentile(queue_wait, 0.5),
        "client_queue_wait_p95": percentile(queue_wait, 0.95),
        "by_turn_index": by_turn_index,
    }


def fmt_metric(value: Optional[float]) -> str:
    if value is None:
        return "na"
    return f"{value:.3f}"


def dump_records_csv(path: str, records: list[RequestRecord]) -> None:
    if not records:
        fields = [f.name for f in RequestRecord.__dataclass_fields__.values()]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
        return

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for row in records:
            writer.writerow(asdict(row))


def load_scenario_overrides(
    scenario_file: str, scenario_name: Optional[str]
) -> tuple[str, dict[str, Any]]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for --scenario-file. Install with `pip install pyyaml`."
        ) from exc

    with open(scenario_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    named_scenarios: dict[str, dict[str, Any]] = {}
    if isinstance(data, dict) and "scenarios" in data and isinstance(
        data["scenarios"], list
    ):
        for item in data["scenarios"]:
            if isinstance(item, dict) and "name" in item:
                item_copy = dict(item)
                name = str(item_copy.pop("name"))
                named_scenarios[name] = item_copy
    elif isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, dict):
                named_scenarios[str(key)] = dict(val)
    else:
        raise ValueError("Unsupported scenario file format.")

    if not named_scenarios:
        raise ValueError("No valid scenarios found in scenario file.")

    if scenario_name is None:
        if len(named_scenarios) != 1:
            raise ValueError(
                "--scenario-name is required when scenario file has multiple scenarios."
            )
        scenario_name = next(iter(named_scenarios.keys()))

    if scenario_name not in named_scenarios:
        raise ValueError(
            f"Scenario '{scenario_name}' not found. Available: {sorted(named_scenarios)}"
        )
    return scenario_name, named_scenarios[scenario_name]


def collect_explicit_cli_args(argv: list[str]) -> set[str]:
    explicit: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", 1)[0].replace("-", "_")
        explicit.add(key)
    return explicit


def apply_scenario(args: argparse.Namespace, explicit_args: set[str]) -> argparse.Namespace:
    if args.scenario_file is None:
        return args

    resolved_name, overrides = load_scenario_overrides(
        args.scenario_file, args.scenario_name
    )
    args.scenario_name = resolved_name
    for raw_key, value in overrides.items():
        key = raw_key.replace("-", "_")
        if hasattr(args, key) and key not in explicit_args:
            setattr(args, key, value)
    return args


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ShareGPT multi-turn concurrent TTFT benchmark (Phase-1)"
    )
    parser.add_argument("--sharegpt-path", type=str, required=True)
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--api-key", type=str, default="EMPTY")

    parser.add_argument("--num-users", type=int, default=1000)
    parser.add_argument("--max-inflight-requests", type=int, default=10)
    parser.add_argument("--new-user-rate", type=float, default=0.5)
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--think-base-sec", type=float, default=1.0)
    parser.add_argument("--read-tok-per-sec", type=float, default=5.0)
    parser.add_argument("--think-sigma", type=float, default=0.6)

    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=1)
    parser.add_argument("--request-timeout-sec", type=float, default=120.0)

    parser.add_argument(
        "--scenario-file",
        type=str,
        default=None,
        help="Path to scenario yaml/json file.",
    )
    parser.add_argument(
        "--scenario-name",
        type=str,
        default=None,
        help="Scenario name in scenario file.",
    )
    parser.add_argument("--output-csv", type=str, default="multiturn_ttft_requests.csv")
    parser.add_argument(
        "--summary-json", type=str, default="multiturn_ttft_summary.json"
    )
    parser.add_argument(
        "--progress-interval-sec",
        type=float,
        default=10.0,
        help="Live progress print interval in seconds; <=0 disables.",
    )
    parser.add_argument(
        "--progress-summary-json",
        type=str,
        default=None,
        help="Optional path to periodically dump live summary snapshots.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_users <= 0:
        raise ValueError("--num-users must be > 0")
    if args.max_inflight_requests <= 0:
        raise ValueError("--max-inflight-requests must be > 0")
    if args.new_user_rate <= 0:
        raise ValueError("--new-user-rate must be > 0")
    if args.duration_sec <= 0:
        raise ValueError("--duration-sec must be > 0")
    if args.request_timeout_sec <= 0:
        raise ValueError("--request-timeout-sec must be > 0")
    if args.progress_interval_sec < 0:
        raise ValueError("--progress-interval-sec must be >= 0")
    if args.max_context_tokens <= 0:
        raise ValueError("--max-context-tokens must be > 0")
    if args.think_base_sec < 0:
        raise ValueError("--think-base-sec must be >= 0")
    if args.read_tok_per_sec <= 0:
        raise ValueError("--read-tok-per-sec must be > 0")
    if args.think_sigma <= 0:
        raise ValueError("--think-sigma must be > 0")
    if args.max_output_tokens != 1:
        raise ValueError(
            "Phase-1 only supports --max-output-tokens=1. "
            "Real decode trajectory is Phase-2 TODO."
        )


async def run_benchmark(args: argparse.Namespace) -> tuple[list[RequestRecord], dict[str, Any]]:
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "openai package is required. Install with `pip install openai`."
        ) from exc

    all_sessions = load_sharegpt_sessions(args.sharegpt_path)
    if len(all_sessions) < args.num_users:
        raise ValueError(
            f"Not enough valid ShareGPT conversations: need {args.num_users}, got {len(all_sessions)}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(all_sessions)
    selected = all_sessions[: args.num_users]

    client = openai.AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.request_timeout_sec,
    )

    run_id = uuid.uuid4().hex
    scenario = args.scenario_name or "default"

    sessions_by_id: dict[int, ConversationSession] = {}
    next_session_to_start = 0
    next_session_id = 0

    semaphore = asyncio.Semaphore(args.max_inflight_requests)
    request_records: list[RequestRecord] = []
    pending_tasks: set[asyncio.Task[Any]] = set()

    inflight_current = 0
    inflight_peak = 0
    inflight_lock = asyncio.Lock()

    start_perf = time.perf_counter()
    end_time_limit = args.duration_sec
    progress_interval = (
        args.progress_interval_sec if args.progress_interval_sec > 0 else None
    )
    next_progress_time = progress_interval
    progress_json_path = (
        Path(args.progress_summary_json) if args.progress_summary_json else None
    )
    if progress_json_path is not None:
        progress_json_path.parent.mkdir(parents=True, exist_ok=True)

    event_heap: list[tuple[float, int, int, DispatchEvent]] = []
    event_seq = 0

    def now_rel() -> float:
        return time.perf_counter() - start_perf

    def schedule_event(event: DispatchEvent) -> None:
        nonlocal event_seq
        if event.event_time > end_time_limit:
            return
        heapq.heappush(
            event_heap, (event.event_time, event.priority, event_seq, event)
        )
        event_seq += 1

    def create_new_session() -> Optional[ConversationSession]:
        nonlocal next_session_to_start, next_session_id
        if next_session_to_start >= len(selected):
            return None
        conversation_id, turns = selected[next_session_to_start]
        next_session_to_start += 1
        s = ConversationSession(
            session_id=next_session_id,
            conversation_id=conversation_id,
            turns=turns,
            next_turn_idx=0,
            finished=False,
        )
        sessions_by_id[s.session_id] = s
        next_session_id += 1
        return s

    def progress_snapshot(elapsed_sec: float, phase: str) -> dict[str, Any]:
        total = len(request_records)
        success = [r for r in request_records if r.status == "success"]
        errors = total - len(success)
        ttft_server = [r.ttft_server_sec for r in success if r.ttft_server_sec is not None]
        queue_wait = [
            r.client_queue_wait_sec
            for r in request_records
            if r.client_queue_wait_sec is not None
        ]

        sessions_started = next_session_id
        sessions_finished = sum(1 for s in sessions_by_id.values() if s.finished)

        return {
            "phase": phase,
            "elapsed_sec": elapsed_sec,
            "duration_sec": end_time_limit,
            "records_total": total,
            "records_success": len(success),
            "records_error": errors,
            "success_rate": (len(success) / total) if total else 0.0,
            "throughput_rps": total / max(elapsed_sec, 1e-9),
            "inflight_current": inflight_current,
            "inflight_peak": inflight_peak,
            "max_inflight_limit": args.max_inflight_requests,
            "pending_tasks": len(pending_tasks),
            "pending_events": len(event_heap),
            "sessions_started": sessions_started,
            "sessions_finished": sessions_finished,
            "sessions_total_limit": args.num_users,
            "ttft_server_p50": percentile(ttft_server, 0.5),
            "ttft_server_p95": percentile(ttft_server, 0.95),
            "client_queue_wait_p50": percentile(queue_wait, 0.5),
            "client_queue_wait_p95": percentile(queue_wait, 0.95),
        }

    def emit_progress(elapsed_sec: float, phase: str) -> None:
        snap = progress_snapshot(elapsed_sec, phase)
        print(
            "[Progress]",
            f"t={snap['elapsed_sec']:.1f}/{snap['duration_sec']:.1f}s",
            f"phase={snap['phase']}",
            f"records={snap['records_total']}",
            f"succ={snap['records_success']}",
            f"err={snap['records_error']}",
            f"succ_rate={snap['success_rate']:.3f}",
            f"inflight={snap['inflight_current']}/{snap['max_inflight_limit']}",
            f"peak={snap['inflight_peak']}",
            f"pending_tasks={snap['pending_tasks']}",
            f"pending_events={snap['pending_events']}",
            f"sessions={snap['sessions_finished']}/{snap['sessions_started']}/{snap['sessions_total_limit']}",
            f"rps={snap['throughput_rps']:.2f}",
            f"ttft_p50={fmt_metric(snap['ttft_server_p50'])}",
            f"ttft_p95={fmt_metric(snap['ttft_server_p95'])}",
            f"qwait_p50={fmt_metric(snap['client_queue_wait_p50'])}",
            f"qwait_p95={fmt_metric(snap['client_queue_wait_p95'])}",
        )
        if progress_json_path is not None:
            with open(progress_json_path, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)

    async def execute_turn(session: ConversationSession, ready_time: float) -> None:
        nonlocal inflight_current, inflight_peak
        if session.finished:
            return

        turn_idx = session.next_turn_idx
        if turn_idx >= len(session.turns):
            session.finished = True
            return

        messages, prompt_tokens_est = build_messages_for_turn(
            session, turn_idx, args.max_context_tokens
        )
        turn_index = turn_idx + 1

        dispatch_time: Optional[float] = None
        first_token_time: Optional[float] = None
        ttft_server: Optional[float] = None
        ttft_effective: Optional[float] = None
        queue_wait: Optional[float] = None
        status = "success"
        error = ""

        submit_time = now_rel()
        await semaphore.acquire()
        try:
            dispatch_time = now_rel()
            queue_wait = max(0.0, dispatch_time - submit_time)

            async with inflight_lock:
                inflight_current += 1
                inflight_peak = max(inflight_peak, inflight_current)

            try:
                stream = await client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_output_tokens,
                    temperature=0.0,
                    stream=True,
                )
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta is not None and delta != "":
                        if first_token_time is None:
                            first_token_time = now_rel()
                if first_token_time is None:
                    first_token_time = now_rel()
                ttft_server = first_token_time - dispatch_time
                ttft_effective = first_token_time - ready_time
            except Exception as exc:  # noqa: BLE001
                status = "error"
                error = str(exc)
        finally:
            async with inflight_lock:
                inflight_current = max(0, inflight_current - 1)
            semaphore.release()

        request_records.append(
            RequestRecord(
                run_id=run_id,
                scenario=scenario,
                seed=args.seed,
                session_id=session.session_id,
                conversation_id=session.conversation_id,
                turn_index=turn_index,
                ready_time=ready_time,
                dispatch_time=dispatch_time,
                first_token_time=first_token_time,
                ttft_server_sec=ttft_server,
                ttft_effective_sec=ttft_effective,
                client_queue_wait_sec=queue_wait,
                prompt_tokens_est=prompt_tokens_est,
                status=status,
                error=error,
            )
        )

        if status != "success":
            session.finished = True
            return

        session.next_turn_idx += 1
        if session.next_turn_idx >= len(session.turns):
            session.finished = True
            return

        prev_turn = session.turns[session.next_turn_idx - 1]
        mean_think = args.think_base_sec + (
            prev_turn.assistant_ref_tokens / args.read_tok_per_sec
        )
        sigma = args.think_sigma
        mu = math.log(max(mean_think, 1e-6)) - 0.5 * sigma * sigma
        think_time = rng.lognormvariate(mu, sigma)
        next_ready_time = (first_token_time or now_rel()) + think_time
        schedule_event(
            DispatchEvent(
                event_time=next_ready_time,
                event_type=DispatchEventType.NEXT_TURN,
                session_id=session.session_id,
                priority=NEXT_TURN_PRIORITY,
            )
        )

    def launch_turn_task(session: ConversationSession, ready_time: float) -> None:
        task = asyncio.create_task(execute_turn(session, ready_time))
        pending_tasks.add(task)

        def _cleanup(t: asyncio.Task[Any]) -> None:
            pending_tasks.discard(t)

        task.add_done_callback(_cleanup)

    # Pre-generate NEW_SESSION events under Poisson arrival.
    arrival_time = 0.0
    for _ in range(args.num_users):
        if arrival_time > end_time_limit:
            break
        schedule_event(
            DispatchEvent(
                event_time=arrival_time,
                event_type=DispatchEventType.NEW_SESSION,
                session_id=None,
                priority=NEW_SESSION_PRIORITY,
            )
        )
        arrival_time += rng.expovariate(args.new_user_rate)

    # Event loop: dispatch due events until duration is reached, then drain in-flight.
    while True:
        now = now_rel()

        while event_heap and event_heap[0][0] <= now:
            _, _, _, event = heapq.heappop(event_heap)
            if event.event_time > end_time_limit:
                continue

            if event.event_type == DispatchEventType.NEW_SESSION:
                s = create_new_session()
                if s is None:
                    continue
                launch_turn_task(s, event.event_time)
            else:
                if event.session_id is None:
                    continue
                s = sessions_by_id.get(event.session_id)
                if s is None or s.finished:
                    continue
                launch_turn_task(s, event.event_time)

        no_more_events = not event_heap or event_heap[0][0] > end_time_limit
        if no_more_events and not pending_tasks:
            break

        if progress_interval is not None and next_progress_time is not None:
            while now >= next_progress_time:
                phase = "run" if now <= end_time_limit else "drain"
                emit_progress(now, phase)
                next_progress_time += progress_interval

        if event_heap:
            next_due = event_heap[0][0]
            sleep_for = max(0.0, min(0.05, next_due - now))
        else:
            sleep_for = 0.01
        await asyncio.sleep(sleep_for)

    benchmark_time = now_rel()
    emit_progress(benchmark_time, "done")
    summary = summarize_records(request_records, benchmark_time)
    summary["run_id"] = run_id
    summary["scenario"] = scenario
    summary["seed"] = args.seed
    summary["benchmark_time_sec"] = benchmark_time
    summary["max_inflight_observed"] = inflight_peak
    if progress_json_path is not None:
        summary["progress_summary_json"] = str(progress_json_path)
    return request_records, summary


def main() -> int:
    parser = create_parser()
    explicit_args = collect_explicit_cli_args(sys.argv[1:])
    args = parser.parse_args()
    args = apply_scenario(args, explicit_args)
    validate_args(args)

    records, summary = asyncio.run(run_benchmark(args))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dump_records_csv(str(output_csv), records)

    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        "Done:",
        f"records={len(records)}",
        f"success_rate={summary['success_rate']:.4f}",
        f"ttft_server_p95={summary['ttft_server_p95']}",
        f"max_inflight_observed={summary['max_inflight_observed']}",
    )
    print(f"Request CSV: {output_csv}")
    print(f"Summary JSON: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
