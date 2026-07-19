#!/usr/bin/env python3
"""Replay the public grading trace with synthetic chat content.

The current public trace only exposes schedule + token shape:
`conv_id`, `turn_idx`, `timestamp_ms`, `think_ms`, `in_tokens_est`,
`out_tokens_max`.

To stress the serving path anyway, this script reconstructs synthetic chat
requests that:
1. preserve the original multi-conversation / multi-turn dependency graph,
2. preserve the turn-0 arrival schedule from `timestamp_ms`,
3. preserve approximate prompt lengths per turn,
4. preserve a shared prefix within each conversation.

This is not meant to reproduce model quality. It is a latency / serving stress
tool for the new public ruleset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-path", required=True)
    parser.add_argument("--model", required=True, help="Tokenizer / model path.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--max-convs", type=int, default=None)
    parser.add_argument(
        "--history-user-tokens",
        type=int,
        default=1,
        help="Synthetic user-token budget for prior turns in the same conversation.",
    )
    parser.add_argument(
        "--history-assistant-tokens",
        type=int,
        default=1,
        help="Synthetic assistant-token budget for prior turns in the same conversation.",
    )
    parser.add_argument(
        "--current-user-min-tokens",
        type=int,
        default=1,
        help="Minimum tokens reserved for the current user turn.",
    )
    return parser.parse_args()


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(len(values) * p)))
    return values[idx]


def stats_block(values: list[float | None]) -> dict[str, float | int] | dict[str, int]:
    xs = [float(value) for value in values if value is not None]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(st.mean(xs), 3),
        "p50": round(st.median(xs), 3),
        "p95": round(pct(xs, 0.95), 3),
        "max": round(max(xs), 3),
    }


class PublicTraceMaterializer:
    def __init__(
        self,
        model_path: str,
        history_user_tokens: int,
        history_assistant_tokens: int,
        current_user_min_tokens: int,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.history_user_tokens = history_user_tokens
        self.history_assistant_tokens = history_assistant_tokens
        self.current_user_min_tokens = current_user_min_tokens
        self.prefix_piece = self._pick_linear_piece(
            [" x", " a", " hi", " test", " foo", " bar", " cat", " dog"]
        )
        self.user_piece = self._pick_linear_piece(
            [" u", " ask", " user", " ping", " one", " alpha"]
        )
        self.assistant_piece = self._pick_linear_piece(
            [" a", " ok", " ans", " reply", " two", " beta"]
        )
        self._prompt_len_cache: dict[tuple[int, int, int], int] = {}

    def _pick_linear_piece(self, candidates: list[str]) -> str:
        for piece in candidates:
            one = len(self.tokenizer.encode(piece, add_special_tokens=False))
            many = len(self.tokenizer.encode(piece * 8, add_special_tokens=False))
            if one == 1 and many == 8:
                return piece
        raise RuntimeError("Could not find a stable 1-token filler piece for this tokenizer.")

    @staticmethod
    def _fill(piece: str, n_tokens: int) -> str:
        if n_tokens <= 0:
            return ""
        return piece * n_tokens

    def _messages(
        self,
        base_prefix_tokens: int,
        turn_idx: int,
        current_user_tokens: int,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self._fill(self.prefix_piece, base_prefix_tokens),
            }
        ]
        for _ in range(turn_idx):
            messages.append(
                {
                    "role": "user",
                    "content": self._fill(self.user_piece, self.history_user_tokens),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": self._fill(
                        self.assistant_piece, self.history_assistant_tokens
                    ),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": self._fill(self.user_piece, current_user_tokens),
            }
        )
        return messages

    def prompt_len(
        self,
        base_prefix_tokens: int,
        turn_idx: int,
        current_user_tokens: int,
    ) -> int:
        key = (base_prefix_tokens, turn_idx, current_user_tokens)
        cached = self._prompt_len_cache.get(key)
        if cached is not None:
            return cached
        rendered = self.tokenizer.apply_chat_template(
            self._messages(base_prefix_tokens, turn_idx, current_user_tokens),
            tokenize=True,
            add_generation_prompt=True,
        )
        input_ids = None
        if hasattr(rendered, "get"):
            input_ids = rendered.get("input_ids")
        elif hasattr(rendered, "input_ids"):
            input_ids = rendered.input_ids
        if input_ids is not None:
            prompt_len = len(input_ids)
        else:
            prompt_len = len(rendered)
        self._prompt_len_cache[key] = prompt_len
        return prompt_len

    def _is_base_feasible(self, base_prefix_tokens: int, targets: list[int]) -> bool:
        for turn_idx, target in enumerate(targets):
            min_len = self.prompt_len(
                base_prefix_tokens, turn_idx, self.current_user_min_tokens
            )
            if min_len > target:
                return False
        return True

    def choose_base_prefix_tokens(self, targets: list[int]) -> int:
        lo, hi = 0, max(targets)
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._is_base_feasible(mid, targets):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def choose_current_user_tokens(
        self,
        base_prefix_tokens: int,
        turn_idx: int,
        target_prompt_tokens: int,
    ) -> tuple[int, int]:
        lo = self.current_user_min_tokens
        hi = max(target_prompt_tokens, lo)
        best_tokens = lo
        best_len = self.prompt_len(base_prefix_tokens, turn_idx, lo)
        while lo <= hi:
            mid = (lo + hi) // 2
            cur_len = self.prompt_len(base_prefix_tokens, turn_idx, mid)
            if cur_len <= target_prompt_tokens:
                best_tokens = mid
                best_len = cur_len
                lo = mid + 1
            else:
                hi = mid - 1
        return best_tokens, best_len

    def materialize(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        convs: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            convs[int(row["conv_id"])].append(row)
        materialized: list[dict[str, Any]] = []
        for conv_id, conv_rows in sorted(convs.items()):
            conv_rows = sorted(conv_rows, key=lambda row: int(row["turn_idx"]))
            targets = [int(row["in_tokens_est"]) for row in conv_rows]
            base_prefix_tokens = self.choose_base_prefix_tokens(targets)
            for row in conv_rows:
                turn_idx = int(row["turn_idx"])
                current_user_tokens, actual_prompt_tokens = self.choose_current_user_tokens(
                    base_prefix_tokens, turn_idx, int(row["in_tokens_est"])
                )
                materialized.append(
                    {
                        **row,
                        "messages": self._messages(
                            base_prefix_tokens, turn_idx, current_user_tokens
                        ),
                        "target_prompt_tokens": int(row["in_tokens_est"]),
                        "actual_prompt_tokens": actual_prompt_tokens,
                        "prompt_gap_tokens": actual_prompt_tokens
                        - int(row["in_tokens_est"]),
                        "base_prefix_tokens": base_prefix_tokens,
                        "current_user_tokens": current_user_tokens,
                    }
                )
        materialized.sort(key=lambda row: (int(row["conv_id"]), int(row["turn_idx"])))
        return materialized


async def run_turn(
    client: httpx.AsyncClient,
    base_url: str,
    served_model_name: str,
    req: dict[str, Any],
    t0: float,
    conv_last_done: dict[int, float],
    timeout_s: float,
) -> dict[str, Any]:
    conv_id = int(req["conv_id"])
    turn_idx = int(req["turn_idx"])
    if turn_idx == 0:
        target = t0 + int(req["timestamp_ms"]) / 1000.0
    else:
        target = conv_last_done[conv_id] + int(req["think_ms"]) / 1000.0
    now = time.monotonic()
    if target > now:
        await asyncio.sleep(target - now)

    body = {
        "model": served_model_name,
        "messages": req["messages"],
        "max_tokens": int(req["out_tokens_max"]),
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    send_time = time.monotonic()
    server_request_id = None
    ttft = None
    chunk_times: list[float] = []
    usage: dict[str, Any] | None = None
    status_code = None
    error_text = None
    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=body,
            timeout=timeout_s,
        ) as resp:
            status_code = resp.status_code
            if resp.status_code != 200:
                error_text = await resp.aread()
                raise RuntimeError(
                    f"status={resp.status_code} body={error_text.decode(errors='replace')[:500]}"
                )
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                if line.strip() == "data: [DONE]":
                    break
                try:
                    payload = json.loads(line[len("data:") :].strip())
                except json.JSONDecodeError:
                    payload = None
                now = time.monotonic()
                if server_request_id is None and isinstance(payload, dict):
                    server_request_id = payload.get("id")
                if isinstance(payload, dict) and payload.get("usage") is not None:
                    usage = payload["usage"]
                chunk_times.append(now)
                if ttft is None:
                    ttft = now - send_time
    except Exception as exc:  # noqa: BLE001
        conv_last_done[conv_id] = time.monotonic()
        return {
            "conv_id": conv_id,
            "turn_idx": turn_idx,
            "send_t_rel_s": round(send_time - t0, 3),
            "scheduled_t_rel_s": round(target - t0, 3),
            "status_code": status_code,
            "error": str(exc),
            "target_prompt_tokens": req["target_prompt_tokens"],
            "actual_prompt_tokens": req["actual_prompt_tokens"],
            "prompt_gap_tokens": req["prompt_gap_tokens"],
            "max_tokens": req["out_tokens_max"],
            "server_request_id": server_request_id,
        }

    done_time = time.monotonic()
    conv_last_done[conv_id] = done_time
    mean_chunk_gap_ms = None
    if len(chunk_times) >= 2:
        mean_chunk_gap_ms = (
            (chunk_times[-1] - chunk_times[0]) * 1000 / (len(chunk_times) - 1)
        )
    return {
        "conv_id": conv_id,
        "turn_idx": turn_idx,
        "send_t_rel_s": round(send_time - t0, 3),
        "scheduled_t_rel_s": round(target - t0, 3),
        "ttft_ms": round(ttft * 1000, 3) if ttft is not None else None,
        "total_latency_ms": round((done_time - send_time) * 1000, 3),
        "client_mean_chunk_gap_ms": round(mean_chunk_gap_ms, 3)
        if mean_chunk_gap_ms is not None
        else None,
        "n_stream_events": len(chunk_times),
        "status_code": status_code,
        "target_prompt_tokens": req["target_prompt_tokens"],
        "actual_prompt_tokens": req["actual_prompt_tokens"],
        "prompt_gap_tokens": req["prompt_gap_tokens"],
        "max_tokens": req["out_tokens_max"],
        "server_request_id": server_request_id,
        "usage": usage,
    }


async def replay(
    rows: list[dict[str, Any]],
    base_url: str,
    served_model_name: str,
    timeout_s: float,
) -> list[dict[str, Any]]:
    conv_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        conv_rows[int(row["conv_id"])].append(row)
    for conv_id in conv_rows:
        conv_rows[conv_id].sort(key=lambda row: int(row["turn_idx"]))

    limits = httpx.Limits(max_connections=256, max_keepalive_connections=256)
    conv_last_done: dict[int, float] = {}
    t0 = time.monotonic()

    async with httpx.AsyncClient(limits=limits) as client:
        async def run_conv(conv_id: int, seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for req in seq:
                out.append(
                    await run_turn(
                        client,
                        base_url,
                        served_model_name,
                        req,
                        t0,
                        conv_last_done,
                        timeout_s,
                    )
                )
            return out

        nested = await asyncio.gather(
            *[run_conv(conv_id, seq) for conv_id, seq in sorted(conv_rows.items())]
        )
    results = [item for seq in nested for item in seq]
    results.sort(key=lambda row: (int(row["conv_id"]), int(row["turn_idx"])))
    return results


def build_summary(materialized: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in results if row.get("ttft_ms") is not None]
    prompt_gaps = [row["prompt_gap_tokens"] for row in materialized]
    return {
        "n_total": len(results),
        "n_failed": len(results) - len(ok),
        "ttft_ms": stats_block([row.get("ttft_ms") for row in ok]),
        "total_latency_ms": stats_block([row.get("total_latency_ms") for row in ok]),
        "client_mean_chunk_gap_ms": stats_block(
            [row.get("client_mean_chunk_gap_ms") for row in ok]
        ),
        "prompt_gap_tokens": stats_block(prompt_gaps),
        "failed_examples": [row for row in results if row.get("error")][:10],
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in Path(args.trace_path).read_text().splitlines() if line.strip()]
    if args.max_convs is not None:
        allowed = sorted({int(row["conv_id"]) for row in rows})[: args.max_convs]
        rows = [row for row in rows if int(row["conv_id"]) in set(allowed)]

    materializer = PublicTraceMaterializer(
        args.model,
        history_user_tokens=args.history_user_tokens,
        history_assistant_tokens=args.history_assistant_tokens,
        current_user_min_tokens=args.current_user_min_tokens,
    )
    materialized = materializer.materialize(rows)
    (out_dir / "materialized_trace.json").write_text(
        json.dumps(materialized, ensure_ascii=False, indent=2)
    )

    results = asyncio.run(
        replay(materialized, args.base_url, args.served_model_name, args.timeout_s)
    )
    summary = build_summary(materialized, results)

    (out_dir / "requests.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
