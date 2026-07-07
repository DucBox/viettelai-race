# vLLM tuning + nguyen ly goc — Track 3 (Qwen3.5-2B)

Kim chi nam toi uu serving. Rang buoc cung: **chi duoc serve bang vLLM**, khong doi
engine. Moi thu duoi day deu xoay quanh cac co cua vLLM.

> Ghi nho khung canh: **single GPU, prefill-bound** (prefill:decode ~95:1), diem
> mat nhieu nhat o **round-1 cold start** (20 request × ~13k token don trong 475ms).
> Cua diem: TTFT F=100/C=1500ms (γ=2, rong), TPOT F=20/C=45ms (hep). Decode hien
> gan sat tran diem → uu tien tuyet doi la **prefill/TTFT**.

## 0. Su that phan cung ve model (doc tu config.json)

| | Gia tri | He qua |
|---|---|---|
| Layer | 24, nhip `L L L F` (`full_attention_interval=4`) | **6 full-attn + 18 GDN (linear)** |
| KV cache phinh o | **chi 6 layer full-attn** | footprint KV nho hon Transformer thuan |
| GDN state | phang ~19 MB/seq, moi do dai | 18 layer khong ton KV theo context |
| `head_dim` | **256** | ⚠️ canh bao cho FP8 KV (xem §2) |
| Attention heads | 8 query / **2 KV** (GQA) | KV moi layer da rat nho |
| `hidden_size` | 2048 | |
| Vision tower | 0.33B, **khong dung** cho text | dead weight trong VRAM |
| MTP head | 0.06B, draft speculative | co san (uu tien thap) |
| Params | 2.27B, BF16, Dense (no MoE) | 1.37B GEMM `*_proj` la phan FP8-able |

---

## 1. Nguyen ly goc: VRAM chua gi → vi sao evict → 3 don bay

### 1.1 Ngan sach VRAM (vi du 18GB × `gpu-mem-util` 0.90 = 16.2 GB)

```
① Model weights   +   ② Framework/runtime   +   ③ KV CACHE POOL
  ~4.5 GB (BF16)        ~1–2 GB                    = phan con lai (~10 GB)
  co dinh               gan co dinh                 CO GIAN — chien truong
```

- **① Weights**: 2.27B × 2 byte (BF16) ≈ **4.5 GB**. FP8 → ~2.3 GB (giai phong ~2.2GB).
  Gom ca vision 0.33B (~0.66GB) nam chet.
- **② Framework** (khong phai context prompt): CUDA context, activation buffer,
  CUDA graph capture, workspace kernel. Gan co dinh.
- **③ KV pool = ngan sach − ① − ②**. Chi phan nay co gian. Giu **KV blocks (6
  full-attn) + GDN state (18 layer)** duoi dang block co dinh.

### 1.2 Toan KV cu the (quan trong cho ket luan)

Moi token, moi full-attn layer: K+V = 2 × (num_kv_heads × head_dim) = 2 × (2×256)
= 1024 phan tu. BF16 = 2048 byte/token/layer × 6 layer = **~12 KB/token** (FP8: ~6KB).

- 1 session 25k token: 25.000 × 12KB ≈ **300 MB** KV + 19MB GDN state ≈ 320 MB.
- 20 session: **~6.4 GB**. Pool ~10GB → **du cho ca 20 session, KHONG bi memory-bound**
  o muc 0.90 tren 18GB.

> **Ket luan then chot:** vi GQA (chi 2 KV head) + chi 6 layer full-attn, KV o day
> **von da rat nho**. Bo nho gan nhu **khong phai** rang buoc that su cua bai nay →
> cac don bay ve bo nho (KV quant, tang util) **it tac dung hon** cac don bay ve
> **toc do prefill**. Day la dieu can nho khi xep uu tien.

### 1.3 Vi sao phai evict

Pool ③ chot cung luc boot. Trong pool co 2 loai cu dan: **active** (request dang
chay — khong duoc dua) va **cached prefix** (KV cua request da xong, giu de tai
dung — **evictable LRU**). Khi block day → evict cached prefix cu → turn sau dang
le hit thi mien → **re-prefill → TTFT vot**. Neu active cung thieu → **preemption**
(`vllm:num_preemptions>0`) → phat nang.

### 1.4 Ba don bay goc (chi co 3, moi giai phap la mot nhanh con)

```
A. KHONG TINH LAI  →  tai dung KV (prefix caching)         [thang round 2–6]
B. TINH NHANH HON  →  cung viec, it thoi gian (FP8, kernel, batch, CUDA graph)  [thang round 1]
C. GIU DUOC CACHE  →  quan bo nho de A con tac dung          [nen tang]
```

⚠️ **Round 1 khong co gi de cache → don bay A = 0 → chi B cuu duoc.** Ma round 1 la
noi mat diem nhat. Nen **B (toc do prefill tho) la thu quyet dinh thu hang**, khong
phai cache. "Ai cache tot hon" chi thang tu round 2 tro di.

---

## 2. KV cache quantization (`--kv-cache-dtype`)

KV cache cung chi la so — luu BF16 (2 byte) hay **FP8 (1 byte)** → **halves bo nho
moi token cached**, luu duoc nhieu token hon → it evict, giu prefix lau hon.

**Cach dung (vLLM):**
- `--kv-cache-dtype fp8` (mac dinh `fp8_e4m3`; co `fp8_e5m2`). `auto` = giu nguyen dtype model.
- `--kv-cache-dtype-skip-layers <...>` : giu mot so layer o BF16 (vd sliding-window). Model nay **khong co sliding window** nen it dung.
- `--calculate-kv-scales` : tu uoc luong scale luc warmup. Mac dinh scale=1.0 (uncalibrated). Muon chuan hon → calibrate bang **llm-compressor** (per-tensor / per-head scale).

**Loi ich:** giam 50% bo nho KV; decode ITL con ~54% BF16; ho tro context dai hon.
Accuracy tut rat it (thuong 1–2 diem, long-context recover 94–98%).

### ⚠️ Nhung tai sao o BAI NAY, KV FP8 co the KHONG dang / thậm chí HAI

1. **`head_dim = 256`** — blog vLLM FP8-KV canh bao ro: model `head_dim=256` khi
   **prefill quan trong** thi FP8 KV lam **tang overhead TTFT** (two-level
   accumulation). Ma TTFT chinh la metric mat diem cua minh → **rui ro nguoc**.
2. **KV von da nho** (§1.2): tiet kiem 3GB tren pool 10GB khong phai nut that →
   loi ich bo nho khong dang ke o day.
3. **Break-even ~7k token**: context minh 13–27k thi qua nguong, nhung loi ich
   chinh la o decode/bo nho — ca hai deu **khong** phai cho minh dang can.

→ **Khuyen nghi:** van **A/B test** `--kv-cache-dtype fp8` (do TTFT truoc/sau bang
script 07 + 08), nhung **dung ky vong** day la don bay lon; nhieu kha nang **net
tieu cuc cho TTFT**. Uu tien **FP8 *weight*** (§4) truoc — no vua nhanh prefill vua
giai phong VRAM that su.

> Luu y phan biet: **FP8 weight** (`--quantization`, quantize trong so) khac **FP8
> KV cache** (`--kv-cache-dtype`, quantize cache). Hai thu doc lap, co the bat rieng.

Model GDN con co cache rieng: `--mamba-cache-dtype` / `--mamba-ssm-cache-dtype`
lang le quantize state 18 layer GDN — nhung state chi ~19MB/seq nen loi ich bo nho
cang nho; chu y độ chinh xac.

---

## 3. Checklist diem nghen (chi vLLM)

- [ ] **① Cold-start round 1** — 20×~13k token don, prefill thuan, khong cache cuu. Mat diem nhat.
- [ ] **② Prefix caching hybrid CO RUI RO** — vLLM prefix caching cho GDN/Mamba-hybrid la **experimental + co bug ghi nhan** (block-size ep = 528 token de align Mamba; chi cache block hoan chinh → duoi prefix mat cache). Phai verify, dung tin mac dinh.
- [ ] **③ Prefill compute-bound** — 1.37B GEMM BF16, quyet dinh TTFT.
- [ ] **④ Eviction/ngan sach VRAM** — thuc te **khong phai** nut that o 18GB (KV nho), nhung van phai giu `num_preemptions=0` va khong evict giua round.
- [ ] **⑤ Nghen CPU (3 core/8GB)** — tokenize 20 prompt dai + overhead HTTP/streaming co the cong tram ms vao TTFT vo hinh.
- [ ] **⑥ TPOT dưới burst** — cua hep 20–45ms; batch prefill to → decode-step phinh → TPOT vot.
- [ ] **⑦ Accuracy gate (nhan)** — FP8 phai giu GPQA ≥ ~0.30, khong mat trang.
- [ ] **⑧ Startup < 15 phut** — CUDA graph compile + warmup khong duoc rot healthcheck.

---

## 4. Checklist giai phap (chi vLLM) — anh xa toi co

> Uu tien theo don bay. Vi prefill-bound, nhom B (tinh nhanh) va A (cache) an diem nhat.

### 🟠 B — Tinh prefill nhanh hon (thang round 1)
- [ ] **FP8 weight quant** *(nghen ①③)* — `--quantization` + checkpoint FP8 (tao bang **llm-compressor**, ignore `lm_head`+layer nhay). Tang toc GEMM `*_proj` **va** giai phong ~2.2GB. ⚠️ verify GPQA giu margin. Bake vao image (cam pull runtime).
- [ ] **Chunked prefill + sweep** *(nghen ①⑥)* — `--enable-chunked-prefill`, `--max-num-batched-tokens` (to → TTFT tot, canh TPOT), `--long-prefill-token-threshold`, `--max-num-partial-prefills` / `--max-long-partial-prefills`.
- [ ] **Bo vision tower** *(nghen ④)* — `--language-model-only` (neu co) de bo xu ly multimodal / `--skip-mm-profiling`; hoac `--limit-mm-per-prompt image=0`. Tra VRAM ve KV, bot compute.
- [ ] **GDN prefill backend** — `--gdn-prefill-backend flashinfer|triton` chon kernel nhanh cho 18 layer GDN (rieng cho arch nay).
- [ ] **CUDA graphs** *(nghen ⑥)* — **KHONG** `--enforce-eager` (giu CUDA graph); tinh `--cudagraph-capture-sizes` / `--max-cudagraph-capture-size` phu batch 20–32.
- [ ] **Attention backend** — `--attention-backend` (FlashAttention/FlashInfer), `--enable-flashinfer-autotune`.

### 🔵 A — Tai dung cache (thang round 2–6)
- [ ] **Bat + verify prefix caching** *(nghen ②)* — `--enable-prefix-caching`; do that su an bang `07_per_request_report.py` (`cache_rd/hit%`). Thu `--prefix-caching-hash-algo`.
- [ ] **Ha max-model-len** *(nghen ④)* — `--max-model-len 32768` (tu 262144) → engine cap KV sat nhu cau.
- [ ] **Block-size / hybrid manager** *(nghen ②)* — `--block-size`, `--mamba-block-size`; can nhac `--disable-hybrid-kv-cache-manager` de so sanh (bug align-mode). `--kv-sharing-fast-prefill` neu co.

### 🟢 C — Giu cache / quan bo nho
- [ ] **`--gpu-memory-utilization 0.95`** — pool to hon (nhung §1.2: khong phai nut that o day).
- [ ] **FP8 weight** (trung nhom B) → khoi ① nho → pool to → don kep.
- [ ] **`--kv-cache-dtype fp8`** — A/B test, **nhung xem canh bao §2** (head_dim=256 hai TTFT).
- [ ] **Theo doi `--kv-cache-metrics`** + `num_preemptions=0`.

### ⚙️ Serving/CPU (re, de bo sot)
- [ ] `--disable-log-stats` + tat request logging → giam tai 3 core.
- [ ] Dam bao **streaming that** (server buffer output → TTFT do duoc te oan).
- [ ] **Warmup luc boot** bang request tu tao (**KHONG dung noi dung trace** — luat cam pre-compute) de compile CUDA graph + nong allocator truoc healthcheck.

### 🟢 Decode/TPOT (uu tien thap — dang on)
- [ ] Speculative decoding qua **MTP head**: `--speculative-config` / `--spec-method` / `--spec-tokens`. CHI khi do TPOT thuc >20ms (dat Floor roi thi khong them diem). Rui ro.

---

## 5. THAM CHIEU CO vLLM DAY DU (tu source `arg_utils.py`)

> ⚠️ Tap co thay doi theo version. **Verify bang `vllm serve --help` tren dung image
> cua ban** (baseline BTC: `vllm/vllm-openai:v0.22.1`). Danh dau: 🎯 = dang de y cho bai nay.

### (1) Model & loading
`--model` · `--tokenizer` · `--tokenizer-mode` · `--trust-remote-code` · `--dtype` 🎯(bf16/auto) · `--seed` · `--max-model-len` 🎯 · `--served-model-name` 🎯 · `--load-format` 🎯 · `--download-dir` · `--revision` · `--hf-config-path` · `--config-format` · `--model-impl` · `--override-attention-dtype` · `--override-generation-config` · `--generation-config` · `--safetensors-load-strategy` / `--safetensors-prefetch-num-threads` / `--safetensors-prefetch-block-size` 🎯(toc do load → startup) · `--ignore-patterns` · `--model-weights` · `--hf-overrides` · `--skip-tokenizer-init`

### (2) Parallelism (single GPU → hau het = 1)
`--tensor-parallel-size/-tp` 🎯(=1) · `--pipeline-parallel-size/-pp` · `--data-parallel-size/-dp` · `--decode-context-parallel-size/-dcp` · `--prefill-context-parallel-size/-pcp` · `--enable-expert-parallel/-ep` (MoE — n/a) · `--distributed-executor-backend` · `--max-parallel-loading-workers` 🎯(startup) · `--worker-cls` · `--numa-bind` · `--device-ids` · (nhieu co DP/EP multi-node khac — khong lien quan single GPU)

### (3) Memory & KV cache
`--gpu-memory-utilization` 🎯 · `--kv-cache-dtype` 🎯 · `--kv-cache-dtype-skip-layers` 🎯 · `--calculate-kv-scales` 🎯 · `--block-size` 🎯 · `--kv-cache-memory-bytes` · `--num-gpu-blocks-override` 🎯 · `--cpu-offload-gb` · `--cpu-offload-params` · `--kv-offloading-size` / `--kv-offloading-backend` · `--kv-sharing-fast-prefill` 🎯 · `--swap-space` (best_of>1; co the =0)

### (4) Prefix caching
`--enable-prefix-caching` 🎯 (`--no-enable-prefix-caching` de tat) · `--prefix-caching-hash-algo` 🎯

### (5) Scheduling / batching / chunked prefill
`--max-num-batched-tokens` 🎯 · `--max-num-seqs` 🎯 · `--enable-chunked-prefill` 🎯 · `--long-prefill-token-threshold` 🎯 · `--max-num-partial-prefills` 🎯 · `--max-long-partial-prefills` 🎯 · `--scheduling-policy` 🎯(fcfs/priority) · `--scheduler-reserve-full-isl` · `--watermark` · `--prefill-schedule-interval` · `--disable-hybrid-kv-cache-manager` 🎯(so sanh cho hybrid) · `--async-scheduling` · `--stream-interval` 🎯 · `--scheduler-cls`

### (6) Quantization (weight)
`--quantization/-q` 🎯 · `--quantization-config` 🎯 · `--allow-deprecated-quantization`

### (7) CUDA graphs & compilation
`--enforce-eager` 🎯(TAT no de giu CUDA graph) · `--cudagraph-capture-sizes` 🎯 · `--max-cudagraph-capture-size` 🎯 · `--compilation-config/-cc` 🎯 · `--optimization-level` · `--performance-mode`

### (8) Attention & kernels
`--attention-backend` 🎯 · `--attention-config/-ac` · `--enable-flashinfer-autotune` 🎯 · `--linear-backend` 🎯(GDN linear) · `--ir-op-priority` · `--kernel-config`

### (9) Mamba / GDN (rieng arch nay)
`--mamba-backend` 🎯(triton/…) · `--gdn-prefill-backend` 🎯(flashinfer/triton) · `--mamba-cache-dtype` 🎯 · `--mamba-ssm-cache-dtype` 🎯 · `--mamba-block-size` 🎯 · `--mamba-cache-mode` 🎯 · `--enable-mamba-cache-stochastic-rounding`

### (10) Speculative decoding (MTP)
`--speculative-config/-sc` 🎯 · `--spec-method` 🎯 · `--spec-model` · `--spec-tokens` 🎯

### (11) Multimodal (bo vision)
`--language-model-only` 🎯(bo xu ly multimodal) · `--limit-mm-per-prompt` 🎯 · `--skip-mm-profiling` 🎯 · `--mm-encoder-*` (nhieu co encoder) · `--media-io-kwargs` · `--mm-processor-kwargs`

### (12) Logging / API / observability
`--disable-log-stats` 🎯 · `--kv-cache-metrics` 🎯 / `--kv-cache-metrics-sample` · `--cudagraph-metrics` · `--otlp-traces-endpoint` · `--collect-detailed-traces` · `--enable-logging-iteration-details` · `--show-hidden-metrics-for-version` · `--enable-mfu-metrics`

### (13) Misc
`--max-logprobs` 🎯 · `--disable-cascade-attn` 🎯 · `--disable-sliding-window` (n/a) · `--enable-sleep-mode` · `--enable-cumem-allocator` · `--shutdown-timeout` · `--tokens-only` · `--enable-prompt-embeds` · `--logits-processors` · `--kv-transfer-config` / `--kv-events-config` (disaggregated — n/a single GPU) · `--additional-config`

### (14) LoRA / structured outputs / diffusion — khong lien quan bai nay
`--enable-lora` … · `--reasoning-parser` · `--structured-outputs-config` · `--diffusion-config` …

---

## 6. Cau hinh khoi diem de nghi (roi sweep)

```bash
vllm serve <model> \
  --served-model-name qwen3.5-2b \
  --dtype bfloat16 \
  --max-model-len 32768 \              # tu 262144 → cap KV sat nhu cau
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \            # verify bang script 07 (cache_rd/hit%)
  --enable-chunked-prefill \
  --max-num-batched-tokens 16384 \     # sweep 8192↔32768 → toi uu TTFT, canh TPOT
  --max-num-seqs 32 \                  # ≥ 20 concurrent + headroom
  --disable-log-stats
  # --language-model-only              # neu co: bo vision tower
  # --quantization <fp8-method>        # DON BAY LON NHAT (checkpoint FP8 bake san) — verify GPQA
  # --kv-cache-dtype fp8               # A/B ONLY — canh bao head_dim=256 hai TTFT (§2)
  # --gdn-prefill-backend flashinfer   # neu co, cho 18 layer GDN
```

Ky luat do: **moi lan doi 1 bien**, ghi lai bang `config ↔ ERS (script 08) ↔ GPQA`,
so voi baseline. Uu tien: **verify prefix caching → FP8 weight → sweep chunked
prefill → bo vision/CPU hygiene → (KV quant, spec decode) chi khi do co loi that**.

**Sources:** [vLLM engine args (arg_utils.py)](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py) · [Quantized KV cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/) · [FP8 KV-cache state (2026-04)](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-22-fp8-kvcache.md) · [Optimization & tuning](https://docs.vllm.ai/en/stable/configuration/optimization/) · [Qwen3.5 prefix-cache block-size bug #40696](https://github.com/vllm-project/vllm/issues/40696) · [Prefix caching for hybrid #26201](https://github.com/vllm-project/vllm/issues/26201) · [LLM Compressor FP8](https://developers.redhat.com/articles/2025/10/07/llm-compressor-080-extended-support-qwen3)
