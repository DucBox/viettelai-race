# Viettel AI Race 2026 — Track 3 · Baseline / Flow-Verify Kit

Serve **Qwen/Qwen3.5-2B** on vLLM, fire test traffic, and view every metric that
matters — **before** touching the production server or any optimization.

> ⚠️ **Runs on a Linux + NVIDIA GPU box only.** This kit does not work on macOS /
> Apple Silicon: vLLM needs CUDA, and Qwen3.5-2B is a hybrid *Gated DeltaNet +
> attention* model whose kernels are CUDA-only. Use a small dev GPU for this step
> (the doc mentions an RTX 3060; a cloud L4 / A10 also works). Everything here is
> pre-written so you just copy the folder to the GPU box and run.

> 📦 **This repo does not download models.** Weights are assumed to already be
> on disk at `serve/models/qwen3.5-2b/` — `scripts/01_check_model.sh` only
> *verifies* the directory is complete and reports exactly what's missing. See
> `serve/models/README.md` for the expected layout.

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
    qwen3.5-2b/         <- put the weights here yourself (gitignored, not fetched by this repo)
scripts/
  00_list_gpus.sh      optional: list host GPUs if the default card (0) is busy
  01_check_model.sh    verify serve/models/<name>/ is complete; reports missing files, doesn't fetch
  serve_up.sh          one command: check model → docker compose up → wait for healthy
  02_smoke_test.sh     one chat request → confirm inference works
  03_watch_metrics.sh  live tail of the metrics that matter (curl /metrics)
  04_inspect_arch.py   architecture facts straight from config.json
  05_inspect_weights.py  per-tensor ground truth from the safetensors header
  06_module_tree.py / 06_module_tree_docker.sh   real nn.Module tree (via Docker py3.11)
bench/
  install_aiperf.sh    installs AIPerf from ../aiperf (vendored clone) or PyPI as fallback
  run_aiperf_baseline.sh   drive load + print report (smoke | trace mode)
aiperf/
  vendored clone of github.com/ai-dynamo/aiperf — source + docs reference (gitignored)
docs/
  VIETTEL AI RACE.pdf         the problem statement
  qwen35-architecture.html    bilingual architecture deep-dive
```

## Deploying on the internal server (multi-GPU, registry image, model already present)

The common path once the model and this repo are already on the box:

```bash
git clone <this-repo-url> && cd viettelai-race
cd serve && cp .env.example .env
```
Edit `serve/.env`:
```bash
IMAGE=registry.internal.example.com/team/vllm-openai:v0.22.1   # your internal image
GPU_ID=0                                                         # defaults to card 0
```
`GPU_ID` pins the container to exactly **one** of the host's cards via
`device_ids` in `docker-compose.yml` — the other 3 are left alone. Only touch
it if card 0 happens to be busy (check with `./scripts/00_list_gpus.sh`).

```bash
cd ..
./scripts/serve_up.sh
```
This verifies `serve/models/qwen3.5-2b/` is complete, brings the container up,
and polls `/health` until vLLM is actually ready (first boot includes model
load + CUDA graph compile, can take a few minutes).

Then collect and view metrics with AIPerf — no request timing / 20-user trace
yet, just confirm the serve→metrics loop works end-to-end:
```bash
./bench/install_aiperf.sh
source bench/.venv/bin/activate
./bench/run_aiperf_baseline.sh        # MODE=smoke (default): light synthetic traffic
```
Watch `./scripts/03_watch_metrics.sh` in another terminal at the same time for
the server-side view (KV cache usage, request queue). AIPerf's own console
table plus `./artifacts/**/server_metrics_export.*` give the client-side view.

## Runbook (on a small dev GPU box)

### 0. Prereqs
- NVIDIA driver + `nvidia-container-toolkit` (so Docker can see the GPU:
  `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`)
- `docker` + `docker compose`, Python 3.10+
- The model weights already placed at `serve/models/qwen3.5-2b/` (see below)

### 1. Configure
```bash
cd serve
cp .env.example .env
# edit .env — on a small GPU keep MAX_MODEL_LEN=32768, GPU_MEM_UTIL≈0.90
cd ..
```

### 2. Verify the model is complete (no download happens here)
```bash
./scripts/01_check_model.sh
```
Exits with a clear list of missing files if the directory is incomplete — get the
weights onto `serve/models/qwen3.5-2b/` however you normally would (rsync/scp from
wherever they're cached) and re-run. See `serve/models/README.md` for the exact
file list this model needs.

### 3. Serve
```bash
cd serve
docker compose up -d vllm               # just the server
docker compose logs -f vllm             # watch it load; wait for "Application startup complete"
```
First boot loads the model + compiles CUDA graphs — give it a few minutes.
Health: `curl -fsS localhost:8000/health` and `curl -fsS localhost:8000/v1/models`.

### 4. Smoke test (does inference actually work?)
```bash
./scripts/02_smoke_test.sh
```
Expect a real answer to the prefix-caching question. If you get one, the model is
loaded, tokenizer is fine, streaming path is fine.

### 5. See the metrics vLLM exposes
```bash
./scripts/03_watch_metrics.sh           # refreshing view of /metrics in terminal 2
# or open the raw feed once:
curl -s localhost:8000/metrics | grep -E '^vllm:(kv_cache_usage_perc|num_requests_(running|waiting)|.*prefix_cache)'
```

### 6. Drive load + view the full report (AIPerf)
```bash
./bench/install_aiperf.sh
source bench/.venv/bin/activate

./bench/run_aiperf_baseline.sh          # quick smoke: concurrency 4, 20 reqs
MODE=trace ./bench/run_aiperf_baseline.sh   # competition-shaped: 20 sessions, ~15k in / 200 out
```
Watch `./scripts/03_watch_metrics.sh` in another terminal while this runs — you'll
see `num_requests_running` jump to ~20 on the burst and `kv_cache_usage_perc` climb.
AIPerf prints a TTFT / ITL / throughput table and writes everything (including
`server_metrics_export.*`) under `./artifacts/`.

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

| Setting              | This test kit           | BTC scoring baseline        |
|----------------------|-------------------------|-----------------------------|
| image                | `vllm/vllm-openai:v0.22.1` | `vllm/vllm-openai:v0.22.1` |
| model source         | local dir, offline      | weights baked into image     |
| `max-model-len`      | 32768 (fits small GPU)  | 262144                      |
| `gpu-memory-util`    | 0.90                    | 0.95                        |
| `enable-prefix-caching` | on                   | on                          |

Keep this local kit deliberately modest — its job is to confirm the pipeline and
let you read metrics, not to hit competition numbers. Real tuning happens next, on
the production GPU.

## Next step (not this kit)
Replay the actual `trace-round1.jsonl` by timestamp with AIPerf's fixed schedule:
```bash
aiperf profile --model qwen3.5-2b --url localhost:8000 --endpoint-type chat --streaming \
  --tokenizer serve/models/qwen3.5-2b \
  --input-file trace-round1.jsonl --custom-dataset-type mooncake_trace --fixed-schedule
```
Then build the internal ERS + GPQA scorers and start the optimization axes
(prefix-cache verification on the hybrid arch → FP8 → chunked prefill → CPU).
