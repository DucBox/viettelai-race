#!/usr/bin/env python3
"""Replay trace-round1.jsonl with real relative timestamps against a local
vLLM server, measuring per-request TTFT. Groups results by round (message
count) to see if TTFT drops for rounds 2-6 (prefix cache hit) vs round 1
(cold start). No trace content used for warmup -- this IS the trace, run
directly as the actual measurement.
"""
import asyncio
import json
import time
import statistics as stats
import httpx

TRACE_PATH = "/root/trace-round1.jsonl"
BASE_URL = "http://localhost:8000/v1/chat/completions"


async def fire_request(client, req, t0, results):
    # wait until its scheduled arrival time relative to t0
    target = t0 + req["timestamp_ms"] / 1000.0
    now = time.monotonic()
    if target > now:
        await asyncio.sleep(target - now)

    body = dict(req["body"])
    body["stream"] = True
    send_time = time.monotonic()
    ttft = None
    try:
        async with client.stream("POST", BASE_URL, json=body, timeout=120.0) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                if line.strip() == "data: [DONE]":
                    break
                if ttft is None:
                    ttft = time.monotonic() - send_time
    except Exception as e:
        results.append({"request_id": req["request_id"], "error": str(e)})
        return

    results.append({
        "request_id": req["request_id"],
        "num_messages": len(req["body"]["messages"]),
        "ttft_ms": ttft * 1000 if ttft is not None else None,
    })


async def main():
    rows = []
    with open(TRACE_PATH) as f:
        for line in f:
            rows.append(json.loads(line))

    results = []
    async with httpx.AsyncClient() as client:
        t0 = time.monotonic()
        tasks = [fire_request(client, r, t0, results) for r in rows]
        await asyncio.gather(*tasks)

    results.sort(key=lambda r: r["request_id"])
    with open("/root/replay_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # group by round (num_messages)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if r.get("ttft_ms") is not None:
            groups[r["num_messages"]].append(r["ttft_ms"])

    print(f"{'msgs':>5} {'n':>4} {'mean_ttft_ms':>14} {'median_ttft_ms':>16} {'min':>8} {'max':>8}")
    for k in sorted(groups):
        v = groups[k]
        print(f"{k:5d} {len(v):4d} {stats.mean(v):14.1f} {stats.median(v):16.1f} {min(v):8.1f} {max(v):8.1f}")

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\n{len(errors)} ERRORS:")
        for e in errors[:5]:
            print(" ", e)


if __name__ == "__main__":
    asyncio.run(main())
