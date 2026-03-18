# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import random
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = None


def build_parser() -> argparse.ArgumentParser:
    schedule_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="TTFT benchmark for tiering-oriented multi-user workload"
    )
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument(
        "--served-model-name",
        type=str,
        required=True,
        help="Served model name exposed by vLLM",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL for the vLLM server",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(schedule_dir / "dataset" / "sharegpt_conv_top200_by_turns.json"),
        help="Converted ShareGPT input JSON file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional output JSON file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-users", type=int, default=128)
    parser.add_argument("--conversations-per-user", type=int, default=4)
    parser.add_argument("--min-user-turns", type=int, default=5)
    parser.add_argument("--max-parallel", type=int, default=32)
    parser.add_argument("--max-num-requests", type=int, default=1000)
    parser.add_argument("--request-rate-per-user", type=float, default=0.2)
    parser.add_argument("--continue-prob", type=float, default=0.8)
    parser.add_argument("--vip-ratio", type=float, default=0.1)
    parser.add_argument("--active-ratio", type=float, default=0.2)
    parser.add_argument("--normal-ratio", type=float, default=0.7)
    parser.add_argument("--vip-weight", type=float, default=8.0)
    parser.add_argument("--active-weight", type=float, default=3.0)
    parser.add_argument("--normal-weight", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=40960,
        help="Model context length used to guard prompt construction",
    )
    parser.add_argument("--request-timeout-sec", type=int, default=120)
    parser.add_argument(
        "--progress-interval-sec",
        type=float,
        default=2.0,
        help="Interval for progress logs during the benchmark",
    )
    parser.add_argument(
        "--print-summary-only",
        action="store_true",
        help="Print summary only without per-group details",
    )
    return parser


def initialize_randomness(seed: int) -> random.Random:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    return random.Random(seed)
