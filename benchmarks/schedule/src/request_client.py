# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request


class RequestClient:
    def __init__(
        self,
        url: str,
        served_model_name: str,
        max_tokens: int,
        request_timeout_sec: int,
        request_seed: int,
    ) -> None:
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.served_model_name = served_model_name
        self.max_tokens = max_tokens
        self.request_timeout_sec = request_timeout_sec
        self.request_seed = request_seed

    async def send(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, int | float | bool | str]:
        payload = {
            "model": self.served_model_name,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "seed": self.request_seed,
        }
        return await asyncio.to_thread(self._send_blocking, payload)

    def _send_blocking(
        self,
        payload: dict[str, object],
    ) -> dict[str, int | float | bool | str]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start_ts = time.perf_counter()
        ttft_ms: float | None = None
        latency_ms = 0.0
        prompt_tokens = 0
        cached_tokens = 0
        generated_tokens = 0

        try:
            with urllib.request.urlopen(
                request, timeout=self.request_timeout_sec
            ) as response:
                for raw_line in response:
                    line = raw_line.strip()
                    if not line:
                        continue

                    chunk = line.decode("utf-8")
                    if chunk.startswith("data: "):
                        chunk = chunk[6:]

                    if chunk == "[DONE]":
                        latency_ms = (time.perf_counter() - start_ts) * 1000.0
                        continue

                    data = json.loads(chunk)
                    choices = data.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {}) or {}
                        if delta.get("content") and ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start_ts) * 1000.0

                    usage = data.get("usage") or {}
                    if usage:
                        prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                        generated_tokens = int(
                            usage.get("completion_tokens", generated_tokens)
                        )
                        prompt_details = usage.get("prompt_tokens_details") or {}
                        cached_tokens = int(
                            prompt_details.get("cached_tokens", cached_tokens)
                        )

            if ttft_ms is None:
                ttft_ms = latency_ms

            return {
                "success": True,
                "error": "",
                "ttft_ms": ttft_ms,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "generated_tokens": generated_tokens or self.max_tokens,
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return {
                "success": False,
                "error": str(exc),
                "ttft_ms": 0.0,
                "latency_ms": 0.0,
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "generated_tokens": 0,
            }
        except Exception as exc:  # pragma: no cover - unexpected response shape
            return {
                "success": False,
                "error": str(exc),
                "ttft_ms": 0.0,
                "latency_ms": 0.0,
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "generated_tokens": 0,
            }
