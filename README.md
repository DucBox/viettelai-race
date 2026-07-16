# Viettel AI Race 2026 — Track 3 · Baseline / Flow-Verify Kit

Serve **LiquidAI/LFM2.5-1.2B-Instruct** on vLLM, fire test traffic, and view every
metric that matters — **before** touching the production server or any optimization.

> ℹ️ **Ruleset changed 2026-07-16.** The competition model is now
> **LFM2.5-1.2B-Instruct** (was Qwen3.5-2B). LFM2.5 is a hybrid *short-range
> convolution + GQA attention* model (16 layers = 10 conv + 6 attention, 1.17B,
> text-only). Everything about the old Qwen setup (docs under `docs/vllm-v0.22.1*`,
> `docs/vllm-v0.24.0*`, `scripts/gpu-l40s-bench/`) is kept only as **reference**.

> ⚠️ **Runs on a Linux + NVIDIA GPU box only.** This kit does not work on macOS /
> Apple Silicon: vLLM needs CUDA. Use a dev GPU for this step (a cloud L4 / A10 /
> L40S works). Everything here is pre-written so you just copy the folder to the
> GPU box and run.

> 📦 **This repo does not download models by default.** Weights are assumed to be
> on disk at `serve/models/lfm2.5-1.2b/` — `scripts/01_check_model.sh` only
> *verifies* the directory is complete and reports what's missing. Grab them with
> `hf download LiquidAI/LFM2.5-1.2B-Instruct --local-dir serve/models/lfm2.5-1.2b`.
> See `serve/models/README.md` for the expected layout.

## The goal of this step

Prove the whole loop works and that you can *see* the numbers:

```
 vLLM serve  ──►  /v1/chat/completions  ──►  AIPerf drives load  ──►  reports
      │                                                                  │
      └────────────► /metrics (KV cache, hit rate, queue) ◄─────────────┘
                     (AIPerf auto-scrapes this every 333 ms)
```

**Key insight:** AIPerf collects *both* sides in one tool — client-side latency
(TTFT / ITL / throughput) **and** vLLM's own server metrics (`kv_cache_usage_perc`,
`num_requests_running/waiting`, prompt-token source mix, generation throughput).
So AIPerf alone gives you the whole display — no Prometheus/Grafana needed.

## Layout

```
serve/
  docker-compose.yml   vLLM server (single service, fully offline — HF_HUB_OFFLINE=1)
  .env.example         copy to .env and edit
  models/
    README.md          expected model directory layout
    lfm2.5-1.2b/        <- put the weights here yourself (gitignored, not fetched by this repo)
scripts/
  00_list_gpus.sh      optional: list host GPUs if the default card (0) is busy
  01_check_model.sh    verify serve/models/<name>/ is complete; reports missing files, doesn't fetch
  serve_up.sh          one command: check model → serve (native or docker, auto-detected) → wait for healthy
  02_smoke_test.sh     one chat request → confirm inference works
  03_watch_metrics.sh  live tail of the metrics that matter (curl /metrics)
  04_inspect_arch.py   architecture facts straight from config.json
  05_inspect_weights.py  per-tensor ground truth from the safetensors header
  06_module_tree.py / 06_module_tree_docker.sh   real nn.Module tree (via Docker py3.11)
  07_per_request_report.py   per-user / per-turn table from AIPerf's raw output + full CSV
bench/
  install_aiperf.sh    installs AIPerf from ../aiperf (vendored clone) or PyPI as fallback
  run_aiperf_baseline.sh   drive load + print report (smoke | trace | replay)
  convert_trace_to_aiperf.py   convert the real trace to AIPerf's mooncake_trace format
data/
  trace-round1.jsonl         real competition trace (organizer-provided, gitignored)
  trace-round1.aiperf.jsonl  converted replay input the bench actually fires (committed)
aiperf/
  vendored clone of github.com/ai-dynamo/aiperf — source + docs reference (gitignored)
docs/
  VIETTEL AI RACE.pdf              the problem statement
  qwen35-architecture.html        (REFERENCE, old Qwen model) architecture deep-dive
  trace-round1-data-description.md  schema + structure of the trace files
```

## Running it

One runbook, two contexts — only the `.env` values differ:

| | Small dev GPU (RTX 3060 / L4 / A10) | Internal multi-GPU server |
|---|---|---|
| `IMAGE` | default (Docker Hub `vllm/vllm-openai:latest`, must support `lfm2`) | your internal registry path |
| `GPU_ID` | default `0` (only card) | default `0` — override only if card 0 is busy (`./scripts/00_list_gpus.sh`) |
| `MAX_MODEL_LEN` | `8192` (trace ~4k in + 200 out) | same, unless the card has more headroom |

### 0. Prereqs
- NVIDIA driver + `nvidia-container-toolkit` (check: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`)
- `docker` + `docker compose`
- Model weights already placed at `serve/models/lfm2.5-1.2b/` (see `serve/models/README.md` for the one-line `hf download`)

### 1. Configure
```bash
cd serve && cp .env.example .env
# edit .env: set IMAGE if using your internal registry, GPU_ID if card 0 is busy
cd ..
```

### 2. Serve
```bash
./scripts/serve_up.sh
```
One command: verifies `serve/models/lfm2.5-1.2b/` is complete (fails fast with an
exact missing-file list if not), brings the container up pinned to `GPU_ID`, and
polls `/health` until vLLM is actually ready (first boot includes model load +
CUDA graph compile — can take a few minutes).

### 3. Smoke test (does inference actually work?)
```bash
./scripts/02_smoke_test.sh
```
Expect a real answer to the prefix-caching question. If you get one, the model is
loaded, tokenizer is fine, streaming path is fine.

### 4. See the metrics vLLM exposes
```bash
./scripts/03_watch_metrics.sh           # refreshing view of /metrics, in its own terminal
```

### 5. Drive load + view the full report (AIPerf)
```bash
./bench/install_aiperf.sh
source bench/.venv/bin/activate

./bench/run_aiperf_baseline.sh              # smoke (default): concurrency 4, 20 reqs
MODE=trace    ./bench/run_aiperf_baseline.sh   # competition-*shaped* synthetic: 20 conc, ~15k in / 200 out
MODE=replay   ./bench/run_aiperf_baseline.sh   # THE REAL TRACE: replays data/trace-round1.jsonl
```
`MODE=replay` is the faithful benchmark: it replays the actual 120 competition
requests (full conversation history baked into each) on the trace's own fixed
timestamp schedule — no synthetic content. It auto-converts the trace via
`bench/convert_trace_to_aiperf.py` on first run.

Keep `./scripts/03_watch_metrics.sh` running in another terminal while this runs —
you'll see `num_requests_running` jump and `kv_cache_usage_perc` climb. AIPerf
prints a TTFT / ITL / throughput table and writes everything (including
`server_metrics_export.*`) under `./artifacts/`.

### 6. Per-request breakdown (per user / per turn)

```bash
./bench/.venv/bin/python scripts/07_per_request_report.py
```

The AIPerf console table only shows aggregates. This reads AIPerf's raw
`profile_export.jsonl` + `server_metrics_export.jsonl` and prints one row per
request — for `MODE=replay` it labels each as **(user, turn)** (user = req%20,
turn = req//20), with arrival / start / end / queue timestamps, TTFT / TPOT /
in / out / latency, per-request **prefix-cache hit** (`cached_tokens` from vLLM
usage), and the nearest KV-cache / prefix scrape. It also dumps a full
`per_request_report.csv` with every column.

## What to look at (mapped to the scoring)

| You want to see        | Where                                                            |
|------------------------|-----------------------------------------------------------------|
| **TTFT**               | AIPerf "Time to First Token"                                    |
| **TPOT / ITL**         | AIPerf "Inter Token Latency"                                    |
| **Throughput**         | AIPerf "Request/Output Token Throughput"                        |
| **KV cache usage**     | `vllm:kv_cache_usage_perc` (watch script / AIPerf server metrics)|
| **Prefix cache reuse** | `vllm:*prefix_cache*` counters + `prompt_tokens_by_source` mix   |
| **Request flow/burst** | `vllm:num_requests_running` / `num_requests_waiting`            |
| **Memory pressure**    | `vllm:num_preemptions` (>0 = eviction happening = bad)          |

## Config vs. the BTC baseline

| Setting              | This test kit           | BTC scoring env              |
|----------------------|-------------------------|-----------------------------|
| image                | `vllm/vllm-openai:latest` (must support `lfm2`) | your submitted image on MIG H200 |
| model source         | local dir, offline      | weights baked into image     |
| `max-model-len`      | 8192 (trace ~4k in + 200 out) | set to fit workload    |
| `gpu-memory-util`    | 0.90                    | 0.95 (on 18 GiB MIG slice)  |
| `enable-prefix-caching` | on                   | on                          |

Keep this local kit deliberately modest — its job is to confirm the pipeline and
let you read metrics, not to hit competition numbers. Real tuning happens next, on
the production GPU.

## Next step (not this kit)
The real-trace replay is now in the kit (`MODE=replay` above). What's left:
build the internal ERS + GPQA scorers and start the optimization axes
(prefix-cache verification on the hybrid arch → FP8 → chunked prefill → CPU).
