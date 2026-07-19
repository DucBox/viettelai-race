#!/usr/bin/env python3
"""Run A/B benchmarks for env-gated V1 backport flags against a real trace."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import httpx

REQSTAT_RE = re.compile(
    r"REQSTAT request_id=(?P<server_request_id>\S+) "
    r"queued_time=(?P<queued_time_s>[0-9.]+) "
    r"prefill_time=(?P<prefill_time_s>[0-9.]+) "
    r"decode_time=(?P<decode_time_s>[0-9.]+) "
    r"inference_time=(?P<inference_time_s>[0-9.]+) "
    r"e2e_latency=(?P<e2e_latency_s>[0-9.]+) "
    r"mean_tpot=(?P<mean_tpot_s>[0-9.]+) "
    r"num_prompt_tokens=(?P<num_prompt_tokens>\d+) "
    r"num_generation_tokens=(?P<num_generation_tokens>\d+) "
    r"num_cached_tokens=(?P<num_cached_tokens>\d+) "
    r"finish_reason=(?P<finish_reason>\S+)"
)

F_TTFT, C_TTFT = 100.0, 1500.0
F_TPOT, C_TPOT = 20.0, 45.0
GAMMA, W = 2.0, 0.5
SAMPLE_INTERVAL_S = 0.5
CPU_STAT_CANDIDATES = (
    Path("/sys/fs/cgroup/cpu.stat"),
    Path("/sys/fs/cgroup/cpu/cpu.stat"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--trace-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--quantization", default="fp8")
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-breakdown", action="store_true")
    parser.add_argument("--server-ready-timeout", type=int, default=900)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="name or name:ENV1=1;ENV2=1",
    )
    return parser.parse_args()


def parse_configs(config_args: list[str]) -> list[tuple[str, dict[str, str]]]:
    configs: list[tuple[str, dict[str, str]]] = []
    for raw in config_args:
        name, _, env_blob = raw.partition(":")
        env_map: dict[str, str] = {}
        if env_blob:
            for item in env_blob.split(";"):
                item = item.strip()
                if not item:
                    continue
                key, sep, value = item.partition("=")
                if not sep:
                    raise ValueError(f"Invalid env item in config {raw!r}: {item!r}")
                env_map[key] = value
        configs.append((name, env_map))
    return configs


def patch_reqstat_logger(python_bin: str) -> None:
    patch_code = r"""
import vllm.v1.metrics.loggers as _lg
path = _lg.__file__
with open(path) as f:
    content = f.read()
if "REQSTAT request_id=%s" in content:
    print("REQSTAT patch already present")
    raise SystemExit(0)
old = "        for finished_request in iteration_stats.finished_requests:\n"
assert content.count(old) == 1, content.count(old)
new = old + (
    '            logger.info(\n'
    '                "REQSTAT request_id=%s queued_time=%.4f prefill_time=%.4f "\n'
    '                "decode_time=%.4f inference_time=%.4f e2e_latency=%.4f "\n'
    '                "mean_tpot=%.4f num_prompt_tokens=%d num_generation_tokens=%d "\n'
    '                "num_cached_tokens=%d finish_reason=%s",\n'
    '                finished_request.request_id,\n'
    '                finished_request.queued_time,\n'
    '                finished_request.prefill_time,\n'
    '                finished_request.decode_time,\n'
    '                finished_request.inference_time,\n'
    '                finished_request.e2e_latency,\n'
    '                finished_request.mean_time_per_output_token,\n'
    '                finished_request.num_prompt_tokens,\n'
    '                finished_request.num_generation_tokens,\n'
    '                finished_request.num_cached_tokens,\n'
    '                finished_request.finish_reason,\n'
    '            )\n'
)
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
print("REQSTAT patch applied")
"""
    subprocess.run([python_bin, "-c", patch_code], check=True)


def wait_for_server(base_url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=5.0)
            if resp.status_code == 200:
                return
            last_error = f"status={resp.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"Server did not become ready: {last_error}")


def launch_server(args: argparse.Namespace, env_map: dict[str, str], run_dir: Path) -> subprocess.Popen[str]:
    log_path = run_dir / "server.log"
    env = os.environ.copy()
    env.update(env_map)
    if args.debug_breakdown:
        env["VLLM_V1_BACKPORT_DEBUG"] = "1"
        env["VLLM_V1_BACKPORT_DEBUG_DUMP_PATH"] = str(run_dir / "backport_debug.json")
    cmd = [
        args.python_bin,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--quantization",
        args.quantization,
        "--kv-cache-dtype",
        args.kv_cache_dtype,
    ]
    cmd.append(
        "--enable-prefix-caching" if args.enable_prefix_caching else "--no-enable-prefix-caching"
    )
    log_file = log_path.open("w")
    return subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)


def read_cpu_stat() -> dict[str, int]:
    cpu_stat_path = next((path for path in CPU_STAT_CANDIDATES if path.exists()), None)
    if cpu_stat_path is None:
        return {"usage_usec": 0, "throttled_usec": 0}
    out: dict[str, int] = {}
    with cpu_stat_path.open() as f:
        for line in f:
            key, value = line.split()
            out[key] = int(value)
    out.setdefault("usage_usec", 0)
    out.setdefault("throttled_usec", 0)
    return out


def read_loadavg() -> dict[str, float]:
    with open("/proc/loadavg") as f:
        parts = f.read().split()
    return {"load1": float(parts[0]), "load5": float(parts[1]), "load15": float(parts[2])}


def read_nvidia_smi() -> dict[str, float]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,clocks.sm,temperature.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    ).stdout.strip()
    util, clk, temp, power, mem = [x.strip() for x in out.split(",")]
    return {
        "gpu_util_pct": float(util),
        "sm_clock_mhz": float(clk),
        "gpu_temp_c": float(temp),
        "power_draw_w": float(power),
        "mem_used_mib": float(mem),
    }


async def sampler_task(t0: float, samples: list[dict[str, float]], stop_event: asyncio.Event) -> None:
    prev_cpu = read_cpu_stat()
    prev_t = time.monotonic()
    while not stop_event.is_set():
        await asyncio.sleep(SAMPLE_INTERVAL_S)
        now = time.monotonic()
        cur_cpu = read_cpu_stat()
        dt_usec = (now - prev_t) * 1e6
        samples.append(
            {
                "t_rel_s": round(now - t0, 3),
                **read_loadavg(),
                **read_nvidia_smi(),
                "cpu_throttled_pct_of_window": round(
                    (cur_cpu["throttled_usec"] - prev_cpu["throttled_usec"]) / dt_usec * 100,
                    2,
                )
                if dt_usec > 0
                else 0.0,
                "cpu_usage_pct_of_window": round(
                    (cur_cpu["usage_usec"] - prev_cpu["usage_usec"]) / dt_usec * 100,
                    2,
                )
                if dt_usec > 0
                else 0.0,
            }
        )
        prev_cpu, prev_t = cur_cpu, now


async def fire_request(
    client: httpx.AsyncClient,
    req: dict,
    base_url: str,
    served_model_name: str,
    t0: float,
    results: list[dict],
) -> None:
    target = t0 + req["timestamp_ms"] / 1000.0
    now = time.monotonic()
    if target > now:
        await asyncio.sleep(target - now)

    body = dict(req["body"])
    body["stream"] = True
    body["model"] = served_model_name
    send_time = time.monotonic()
    ttft = None
    ttft_wall_rel = None
    server_request_id = None
    n_chunks = 0
    chunk_timestamps: list[float] = []
    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=body,
            timeout=180.0,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                if line.strip() == "data: [DONE]":
                    break
                try:
                    payload = json.loads(line[len("data:") :].strip())
                except json.JSONDecodeError:
                    payload = None
                if server_request_id is None and isinstance(payload, dict):
                    server_request_id = payload.get("id")
                n_chunks += 1
                chunk_timestamps.append(time.monotonic())
                if ttft is None:
                    ttft = time.monotonic() - send_time
                    ttft_wall_rel = time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001
        results.append(
            {
                "request_id": req["request_id"],
                "send_t_rel_s": round(send_time - t0, 3),
                "error": str(exc),
            }
        )
        return

    total_latency = time.monotonic() - send_time
    client_mean_chunk_gap_ms = None
    if len(chunk_timestamps) >= 2:
        client_mean_chunk_gap_ms = (
            (chunk_timestamps[-1] - chunk_timestamps[0]) * 1000 / (len(chunk_timestamps) - 1)
        )
    results.append(
        {
            "request_id": req["request_id"],
            "server_request_id": server_request_id,
            "send_t_rel_s": round(send_time - t0, 3),
            "ttft_arrival_t_rel_s": round(ttft_wall_rel, 3) if ttft_wall_rel else None,
            "ttft_ms": ttft * 1000 if ttft is not None else None,
            "total_latency_ms": total_latency * 1000,
            "output_chunks": n_chunks,
            "client_mean_chunk_gap_ms": client_mean_chunk_gap_ms,
        }
    )


async def replay_trace(
    trace_path: Path, base_url: str, served_model_name: str
) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    results: list[dict] = []
    samples: list[dict[str, float]] = []
    stop_event = asyncio.Event()

    async with httpx.AsyncClient() as client:
        t0 = time.monotonic()
        sampler = asyncio.create_task(sampler_task(t0, samples, stop_event))
        tasks = [
            fire_request(client, row, base_url, served_model_name, t0, results)
            for row in rows
        ]
        await asyncio.gather(*tasks)
        stop_event.set()
        await sampler
    results.sort(key=lambda row: row["request_id"])
    return results, samples


def parse_reqstats(log_path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in log_path.read_text().splitlines():
        match = REQSTAT_RE.search(line)
        if not match:
            continue
        data = match.groupdict()
        server_request_id = data.pop("server_request_id")
        rows[server_request_id] = {
            "server_request_id": server_request_id,
            "server_queue_ms": float(data["queued_time_s"]) * 1000,
            "server_prefill_ms": float(data["prefill_time_s"]) * 1000,
            "server_decode_ms": float(data["decode_time_s"]) * 1000,
            "server_inference_ms": float(data["inference_time_s"]) * 1000,
            "server_e2e_ms": float(data["e2e_latency_s"]) * 1000,
            "server_mean_tpot_ms": float(data["mean_tpot_s"]) * 1000,
            "num_prompt_tokens": int(data["num_prompt_tokens"]),
            "num_generation_tokens": int(data["num_generation_tokens"]),
            "num_cached_tokens": int(data["num_cached_tokens"]),
            "finish_reason": data["finish_reason"],
        }
    return rows


def merge_results(requests: list[dict], reqstats: dict[str, dict]) -> list[dict]:
    merged: list[dict] = []
    for row in requests:
        merged_row = dict(row)
        server_request_id = row.get("server_request_id")
        if server_request_id and server_request_id in reqstats:
            merged_row.update(reqstats[server_request_id])
        merged.append(merged_row)
    return merged


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, int(p * len(values)))
    return values[idx]


def stats_block(values: list[float | None]) -> dict[str, float | int | None]:
    xs = [value for value in values if value is not None]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(st.mean(xs), 3),
        "p50": round(st.median(xs), 3),
        "p95": round(pct(xs, 0.95), 3),
        "p99": round(pct(xs, 0.99), 3),
        "max": round(max(xs), 3),
    }


def comp(value: float | None, floor: float, ceil: float) -> float:
    if value is None:
        return 0.0
    score = (ceil - value) / (ceil - floor)
    score = max(0.0, min(1.0, score))
    return score**GAMMA


def build_system_summary(samples: list[dict]) -> dict[str, float | int | None]:
    if not samples:
        return {}

    def col(name: str) -> list[float]:
        return [float(sample[name]) for sample in samples if name in sample]

    util = col("gpu_util_pct")
    clk = col("sm_clock_mhz")
    power = col("power_draw_w")
    throttle = col("cpu_throttled_pct_of_window")
    cpu_usage = col("cpu_usage_pct_of_window")
    return {
        "n_samples": len(samples),
        "gpu_util_mean": round(st.mean(util), 3) if util else None,
        "sm_clock_mean": round(st.mean(clk), 3) if clk else None,
        "power_mean": round(st.mean(power), 3) if power else None,
        "cpu_throttle_max": round(max(throttle), 3) if throttle else None,
        "cpu_usage_mean": round(st.mean(cpu_usage), 3) if cpu_usage else None,
    }


def build_summary(name: str, env_map: dict[str, str], merged: list[dict], samples: list[dict], run_dir: Path) -> dict:
    ok = [row for row in merged if row.get("ttft_ms") is not None]
    ttft_scores = [comp(row.get("ttft_ms"), F_TTFT, C_TTFT) for row in ok]
    client_tpot_scores = [
        comp(row.get("client_mean_chunk_gap_ms"), F_TPOT, C_TPOT) for row in ok
    ]
    server_tpot_scores = [
        comp(row.get("server_mean_tpot_ms"), F_TPOT, C_TPOT) for row in ok
    ]
    score_client = (
        sum(
            W * comp(row.get("ttft_ms"), F_TTFT, C_TTFT)
            + (1 - W) * comp(row.get("client_mean_chunk_gap_ms"), F_TPOT, C_TPOT)
            for row in merged
        )
        / len(merged)
        if merged
        else 0.0
    )
    score_server = (
        sum(
            W * comp(row.get("ttft_ms"), F_TTFT, C_TTFT)
            + (1 - W) * comp(row.get("server_mean_tpot_ms"), F_TPOT, C_TPOT)
            for row in merged
        )
        / len(merged)
        if merged
        else 0.0
    )
    debug_stats = {}
    debug_path = run_dir / "backport_debug.json"
    if debug_path.exists():
        debug_stats = json.loads(debug_path.read_text())
    return {
        "name": name,
        "env": env_map,
        "n_total": len(merged),
        "n_failed": len(merged) - len(ok),
        "score_client_x100": round(score_client * 100, 3),
        "score_server_x100": round(score_server * 100, 3),
        "mean_s_ttft": round(st.mean(ttft_scores), 6) if ttft_scores else None,
        "mean_s_tpot_client": round(st.mean(client_tpot_scores), 6)
        if client_tpot_scores
        else None,
        "mean_s_tpot_server": round(st.mean(server_tpot_scores), 6)
        if server_tpot_scores
        else None,
        "ttft_ms": stats_block([row.get("ttft_ms") for row in ok]),
        "tpot_ms_server": stats_block([row.get("server_mean_tpot_ms") for row in ok]),
        "tpot_ms_client": stats_block([row.get("client_mean_chunk_gap_ms") for row in ok]),
        "queue_ms": stats_block([row.get("server_queue_ms") for row in ok]),
        "prefill_ms": stats_block([row.get("server_prefill_ms") for row in ok]),
        "decode_ms": stats_block([row.get("server_decode_ms") for row in ok]),
        "system": build_system_summary(samples),
        "backport_debug": debug_stats,
    }


def write_csv(path: Path, summaries: list[dict]) -> None:
    lines = [
        "name,n_total,n_failed,score_client_x100,score_server_x100,ttft_mean,ttft_p50,ttft_p95,tpot_server_mean,tpot_server_p50,tpot_server_p95,queue_mean,prefill_mean,decode_mean"
    ]
    for summary in summaries:
        ttft = summary["ttft_ms"]
        tpot = summary["tpot_ms_server"]
        queue = summary["queue_ms"]
        prefill = summary["prefill_ms"]
        decode = summary["decode_ms"]
        lines.append(
            ",".join(
                [
                    summary["name"],
                    str(summary["n_total"]),
                    str(summary["n_failed"]),
                    str(summary["score_client_x100"]),
                    str(summary["score_server_x100"]),
                    str(ttft.get("mean")),
                    str(ttft.get("p50")),
                    str(ttft.get("p95")),
                    str(tpot.get("mean")),
                    str(tpot.get("p50")),
                    str(tpot.get("p95")),
                    str(queue.get("mean")),
                    str(prefill.get("mean")),
                    str(decode.get("mean")),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    configs = parse_configs(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = Path(args.trace_path)
    base_url = f"http://{args.host}:{args.port}"

    patch_reqstat_logger(args.python_bin)

    summaries: list[dict] = []
    for name, env_map in configs:
        run_dir = out_dir / name
        if run_dir.exists():
            for child in run_dir.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink()
        else:
            run_dir.mkdir(parents=True)

        print(f"\n=== RUN {name} ===", flush=True)
        print(f"env={env_map}", flush=True)
        proc = launch_server(args, env_map, run_dir)
        try:
            wait_for_server(base_url, args.server_ready_timeout)
            requests, samples = asyncio.run(
                replay_trace(trace_path, base_url, args.served_model_name)
            )
        finally:
            stop_server(proc)

        (run_dir / "requests.json").write_text(json.dumps(requests, indent=2))
        (run_dir / "samples.json").write_text(json.dumps(samples, indent=2))
        reqstats = parse_reqstats(run_dir / "server.log")
        merged = merge_results(requests, reqstats)
        (run_dir / "merged.json").write_text(json.dumps(merged, indent=2))
        summary = build_summary(name, env_map, merged, samples, run_dir)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "name": name,
                    "score_client_x100": summary["score_client_x100"],
                    "n_failed": summary["n_failed"],
                    "ttft_p50": summary["ttft_ms"].get("p50"),
                    "ttft_p95": summary["ttft_ms"].get("p95"),
                    "tpot_p50": summary["tpot_ms_server"].get("p50"),
                    "tpot_p95": summary["tpot_ms_server"].get("p95"),
                },
                indent=2,
            ),
            flush=True,
        )

    write_csv(out_dir / "summary.csv", summaries)
    print(f"\nWrote {out_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
