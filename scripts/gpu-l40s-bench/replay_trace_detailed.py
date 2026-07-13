#!/usr/bin/env python3
"""Replay trace-round1.jsonl and record, for EVERY request individually
(not just round-level aggregates):
  - request_id, round, num_messages
  - wall-clock send time and TTFT-arrival time (so it can be joined against
    the concurrent system sampler timeline below)
  - ttft_ms, total_latency_ms, output token count

While the replay runs, a background task samples host state (loadavg,
cgroup cpu.stat, nvidia-smi util/clocks/temp/power) every SAMPLE_INTERVAL_S
seconds -- continuously, not just before/after -- so each request's TTFT can
be correlated against what the GPU/CPU were doing at that exact moment,
instead of an averaged before/after delta that can hide short-lived spikes.

Output: two JSON files in /root:
  - replay_detailed_requests.json   (per-request rows)
  - replay_detailed_samples.json    (system timeline, one row per interval)

Also prints a per-request table (not just per-round mean/median) sorted by
ttft_ms descending, so the worst outliers are immediately visible together
with the nearest system sample.
"""
import asyncio
import json
import subprocess
import time
import httpx

TRACE_PATH = "/root/trace-round1.jsonl"
BASE_URL = "http://localhost:8000/v1/chat/completions"
SAMPLE_INTERVAL_S = 0.5

round_of_nmsg = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5}
NUM_USERS = 20


def read_cpu_stat():
    d = {}
    with open("/sys/fs/cgroup/cpu.stat") as f:
        for line in f:
            k, v = line.split()
            d[k] = int(v)
    return d


def read_loadavg():
    with open("/proc/loadavg") as f:
        parts = f.read().split()
    return {"load1": float(parts[0]), "load5": float(parts[1]), "load15": float(parts[2])}


def read_nvidia_smi():
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=utilization.gpu,clocks.sm,temperature.gpu,power.draw,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    util, clk, temp, power, mem = [x.strip() for x in out.split(",")]
    return {
        "gpu_util_pct": float(util),
        "sm_clock_mhz": float(clk),
        "gpu_temp_c": float(temp),
        "power_draw_w": float(power),
        "mem_used_mib": float(mem),
    }


async def sampler_task(t0, samples, stop_event):
    prev_cpu = read_cpu_stat()
    prev_t = time.monotonic()
    while not stop_event.is_set():
        await asyncio.sleep(SAMPLE_INTERVAL_S)
        now = time.monotonic()
        cur_cpu = read_cpu_stat()
        dt_usec = (now - prev_t) * 1e6
        row = {
            "t_rel_s": round(now - t0, 3),
            **read_loadavg(),
            **read_nvidia_smi(),
            "cpu_throttled_pct_of_window": round(
                (cur_cpu["throttled_usec"] - prev_cpu["throttled_usec"]) / dt_usec * 100, 2
            ) if dt_usec > 0 else 0.0,
            "cpu_usage_pct_of_window": round(
                (cur_cpu["usage_usec"] - prev_cpu["usage_usec"]) / dt_usec * 100, 2
            ) if dt_usec > 0 else 0.0,
        }
        samples.append(row)
        prev_cpu, prev_t = cur_cpu, now


async def fire_request(client, req, t0, results):
    target = t0 + req["timestamp_ms"] / 1000.0
    now = time.monotonic()
    if target > now:
        await asyncio.sleep(target - now)

    body = dict(req["body"])
    body["stream"] = True
    send_time = time.monotonic()
    ttft = None
    ttft_wall_rel = None
    server_request_id = None
    n_chunks = 0
    chunk_timestamps = []
    try:
        async with client.stream("POST", BASE_URL, json=body, timeout=120.0) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                if line.strip() == "data: [DONE]":
                    break
                payload = None
                try:
                    payload = json.loads(line[len("data:"):].strip())
                except json.JSONDecodeError:
                    payload = None
                if server_request_id is None and isinstance(payload, dict):
                    server_request_id = payload.get("id")
                n_chunks += 1
                chunk_timestamps.append(time.monotonic())
                if ttft is None:
                    ttft = time.monotonic() - send_time
                    ttft_wall_rel = time.monotonic() - t0
    except Exception as e:
        results.append({
            "request_id": req["request_id"],
            "send_t_rel_s": round(send_time - t0, 3),
            "error": str(e),
        })
        return
    total_latency = time.monotonic() - send_time

    nmsg = len(req["body"]["messages"])
    round_idx = round_of_nmsg.get(nmsg)
    client_mean_chunk_gap_ms = None
    if len(chunk_timestamps) >= 2:
        client_mean_chunk_gap_ms = (
            (chunk_timestamps[-1] - chunk_timestamps[0]) * 1000 / (len(chunk_timestamps) - 1)
        )
    results.append({
        "request_id": req["request_id"],
        "user_id": req["request_id"] % NUM_USERS,
        "num_messages": nmsg,
        "round": round_idx,
        "turn_index": round_idx,
        "server_request_id": server_request_id,
        "send_t_rel_s": round(send_time - t0, 3),
        "ttft_arrival_t_rel_s": round(ttft_wall_rel, 3) if ttft_wall_rel else None,
        "ttft_ms": ttft * 1000 if ttft is not None else None,
        "total_latency_ms": total_latency * 1000,
        "output_chunks": n_chunks,
        "client_mean_chunk_gap_ms": client_mean_chunk_gap_ms,
    })


def nearest_sample(samples, t_rel):
    if not samples or t_rel is None:
        return None
    return min(samples, key=lambda s: abs(s["t_rel_s"] - t_rel))


async def main():
    rows = []
    with open(TRACE_PATH) as f:
        for line in f:
            rows.append(json.loads(line))

    results = []
    samples = []
    stop_event = asyncio.Event()

    async with httpx.AsyncClient() as client:
        t0 = time.monotonic()
        sampler = asyncio.create_task(sampler_task(t0, samples, stop_event))
        tasks = [fire_request(client, r, t0, results) for r in rows]
        await asyncio.gather(*tasks)
        stop_event.set()
        await sampler

    results.sort(key=lambda r: r["request_id"])
    with open("/root/replay_detailed_requests.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("/root/replay_detailed_samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    ok = [r for r in results if r.get("ttft_ms") is not None]
    errors = [r for r in results if "error" in r]

    print(f"Tong so request: {len(results)}  OK: {len(ok)}  ERROR: {len(errors)}")
    print(f"So sample he thong da thu: {len(samples)} (moi {SAMPLE_INTERVAL_S}s)\n")

    print("=== Toan bo request, sap xep theo TTFT giam dan, kem trang thai he thong ===")
    print(f"{'req_id':>7} {'round':>5} {'ttft_ms':>9} {'total_ms':>9} "
          f"{'gpu_util%':>9} {'sm_mhz':>7} {'cpu_throttle%':>13} {'load1':>6}")
    for r in sorted(ok, key=lambda r: -r["ttft_ms"]):
        s = nearest_sample(samples, r.get("ttft_arrival_t_rel_s"))
        if s:
            print(f"{r['request_id']:>7} {str(r['round']):>5} {r['ttft_ms']:>9.1f} "
                  f"{r['total_latency_ms']:>9.1f} {s['gpu_util_pct']:>9.1f} "
                  f"{s['sm_clock_mhz']:>7.0f} {s['cpu_throttled_pct_of_window']:>13.1f} "
                  f"{s['load1']:>6.2f}")
        else:
            print(f"{r['request_id']:>7} {str(r['round']):>5} {r['ttft_ms']:>9.1f} "
                  f"{r['total_latency_ms']:>9.1f} {'--':>9} {'--':>7} {'--':>13} {'--':>6}")

    if errors:
        print(f"\n{len(errors)} ERRORS:")
        for e in errors[:10]:
            print(" ", e)


if __name__ == "__main__":
    asyncio.run(main())
