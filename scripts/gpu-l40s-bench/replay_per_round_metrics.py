#!/usr/bin/env python3
"""Replay trace, snapshotting /metrics right after each round's 20 requests
ALL fully complete (queue_time/prefill_time are recorded at full-completion
time in vLLM, not at TTFT -- confirmed empirically: an earlier version of
this script that snapshotted at TTFT produced round1 n=0, round2 n=9, round3
n=31, proving these sums only update at request completion). Snapshotting at
completion aligns the snapshot instant with when the metric actually updates.
"""
import asyncio
import json
import time
import httpx

TRACE_PATH = "/root/trace-round1.jsonl"
BASE_URL = "http://localhost:8000/v1/chat/completions"
METRICS_URL = "http://localhost:8000/metrics"

round_of_nmsg = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5}
round_done_count = [0] * 6
round_snapshots = [None] * 6
snapshot_lock = asyncio.Lock()


async def snapshot_metrics(client):
    r = await client.get(METRICS_URL)
    return r.text


async def fire_request(client, req, t0, results):
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

    # request fully done here (stream exhausted) -- this is when vLLM
    # finalizes queue_time/prefill_time/e2e metrics for this request
    nmsg = len(req["body"]["messages"])
    rnd = round_of_nmsg[nmsg]
    async with snapshot_lock:
        round_done_count[rnd] += 1
        if round_done_count[rnd] == 20 and round_snapshots[rnd] is None:
            round_snapshots[rnd] = await snapshot_metrics(client)

    results.append({
        "request_id": req["request_id"],
        "num_messages": nmsg,
        "ttft_ms": ttft * 1000 if ttft is not None else None,
    })


def parse_metrics(text):
    keys = [
        "vllm:request_queue_time_seconds_sum",
        "vllm:request_queue_time_seconds_count",
        "vllm:request_prefill_time_seconds_sum",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
        "vllm:request_inference_time_seconds_sum",
    ]
    out = {}
    for line in text.splitlines():
        for k in keys:
            if line.startswith(k + "{"):
                val = float(line.rsplit(" ", 1)[-1])
                out[k] = val
    return out


async def main():
    rows = []
    with open(TRACE_PATH) as f:
        for line in f:
            rows.append(json.loads(line))

    results = []
    async with httpx.AsyncClient() as client:
        baseline = parse_metrics(await snapshot_metrics(client))
        t0 = time.monotonic()
        tasks = [fire_request(client, r, t0, results) for r in rows]
        await asyncio.gather(*tasks)

    with open("/root/replay_per_round_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"{'round':>6} {'queue_ms':>10} {'prefill_ms':>11} {'ttft_ms':>9} {'n':>4}")
    prev = baseline
    for i in range(6):
        if round_snapshots[i] is None:
            print(f"{i+1:>6}  (khong du 20/20 hoan tat -- snapshot thieu)")
            continue
        cur = parse_metrics(round_snapshots[i])
        dq = cur["vllm:request_queue_time_seconds_sum"] - prev["vllm:request_queue_time_seconds_sum"]
        dn = cur["vllm:request_queue_time_seconds_count"] - prev["vllm:request_queue_time_seconds_count"]
        dp = cur["vllm:request_prefill_time_seconds_sum"] - prev["vllm:request_prefill_time_seconds_sum"]
        dt = cur["vllm:time_to_first_token_seconds_sum"] - prev["vllm:time_to_first_token_seconds_sum"]
        dtn = cur["vllm:time_to_first_token_seconds_count"] - prev["vllm:time_to_first_token_seconds_count"]
        n = dn if dn > 0 else 1
        print(f"{i+1:>6} {dq/n*1000:>10.1f} {dp/n*1000:>11.1f} {dt/dtn*1000 if dtn else 0:>9.1f} {int(dn):>4}")
        prev = cur


if __name__ == "__main__":
    asyncio.run(main())
