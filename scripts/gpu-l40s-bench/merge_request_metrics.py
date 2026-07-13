#!/usr/bin/env python3
"""Join client-side per-request replay metrics with server-side REQSTAT logs.

Inputs:
  - /root/replay_detailed_requests.json
  - /root/vllm_serve*.log (path passed via --log)

Output:
  - /root/replay_request_metrics_full.json
"""
import argparse
import json
import re
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", default="/root/replay_detailed_requests.json")
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", default="/root/replay_request_metrics_full.json")
    return parser.parse_args()


def parse_reqstats(log_path: Path):
    rows = {}
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


def main():
    args = parse_args()
    requests = json.loads(Path(args.requests).read_text())
    reqstats = parse_reqstats(Path(args.log))

    merged = []
    missing_server = 0
    for row in requests:
        server_request_id = row.get("server_request_id")
        merged_row = dict(row)
        if server_request_id and server_request_id in reqstats:
            merged_row.update(reqstats[server_request_id])
        else:
            missing_server += 1
        merged.append(merged_row)

    Path(args.out).write_text(json.dumps(merged, indent=2))

    ok = [r for r in merged if "server_queue_ms" in r and r.get("ttft_ms") is not None]
    print(
        f"Merged {len(merged)} requests, joined server stats for {len(ok)} "
        f"requests, missing_server_stats={missing_server}"
    )
    if ok:
        print(
            f"{'req_id':>7} {'user':>4} {'turn':>4} {'queue':>9} {'prefill':>9} "
            f"{'ttft':>9} {'tpot':>9} {'total':>9}"
        )
        for row in sorted(ok, key=lambda r: (r["request_id"])):
            print(
                f"{row['request_id']:>7} {row.get('user_id', -1):>4} "
                f"{row.get('turn_index', -1):>4} "
                f"{row['server_queue_ms']:>9.1f} {row['server_prefill_ms']:>9.1f} "
                f"{row['ttft_ms']:>9.1f} {row['server_mean_tpot_ms']:>9.1f} "
                f"{row['total_latency_ms']:>9.1f}"
            )


if __name__ == "__main__":
    main()
