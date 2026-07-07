# vLLM tuning + nguyên lý gốc — Track 3 (Qwen3.5-2B)

Kim chỉ nam tối ưu serving. Ràng buộc cứng: **chỉ được serve bằng vLLM**, không đổi
engine. Mọi thứ dưới đây đều xoay quanh các cờ của vLLM.

> Ghi nhớ khung cảnh: **single GPU, prefill-bound** (prefill:decode ~95:1), điểm
> mất nhiều nhất ở **round-1 cold start** (20 request × ~13k token dồn trong 475ms).
> Cửa điểm: TTFT F=100/C=1500ms (γ=2, rộng), TPOT F=20/C=45ms (hẹp). Decode hiện
> gần sát trần điểm → ưu tiên tuyệt đối là **prefill/TTFT**.

## 0. Sự thật phần cứng về model (đọc từ config.json)

| | Giá trị | Hệ quả |
|---|---|---|
| Layer | 24, nhịp `L L L F` (`full_attention_interval=4`) | **6 full-attn + 18 GDN (linear)** |
| KV cache phình ở | **chỉ 6 layer full-attn** | footprint KV nhỏ hơn Transformer thuần |
| GDN state | phẳng ~19 MB/seq, mọi độ dài | 18 layer không tốn KV theo context |
| `head_dim` | **256** | ⚠️ cảnh báo cho FP8 KV (xem §2) |
| Attention heads | 8 query / **2 KV** (GQA) | KV mỗi layer đã rất nhỏ |
| `hidden_size` | 2048 | |
| Vision tower | 0.33B, **không dùng** cho text | dead weight trong VRAM |
| MTP head | 0.06B, draft speculative | có sẵn (ưu tiên thấp) |
| Params | 2.27B, BF16, Dense (no MoE) | 1.37B GEMM `*_proj` là phần FP8-able |

---

## 1. Nguyên lý gốc: VRAM chứa gì → vì sao evict → 3 đòn bẩy

### 1.1 Ngân sách VRAM (ví dụ 18GB × `gpu-mem-util` 0.90 = 16.2 GB)

```
① Model weights   +   ② Framework/runtime   +   ③ KV CACHE POOL
  ~4.5 GB (BF16)        ~1–2 GB                    = phần còn lại (~10 GB)
  cố định               gần cố định                 CO GIÃN — chiến trường
```

- **① Weights**: 2.27B × 2 byte (BF16) ≈ **4.5 GB**. FP8 → ~2.3 GB (giải phóng ~2.2GB).
  Gồm cả vision 0.33B (~0.66GB) nằm chết.
- **② Framework** (không phải context prompt): CUDA context, activation buffer,
  CUDA graph capture, workspace kernel. Gần cố định.
- **③ KV pool = ngân sách − ① − ②**. Chỉ phần này co giãn. Giữ **KV blocks (6
  full-attn) + GDN state (18 layer)** dưới dạng block cố định.

### 1.2 Toán KV cụ thể (quan trọng cho kết luận)

Mỗi token, mỗi full-attn layer: K+V = 2 × (num_kv_heads × head_dim) = 2 × (2×256)
= 1024 phần tử. BF16 = 2048 byte/token/layer × 6 layer = **~12 KB/token** (FP8: ~6KB).

- 1 session 25k token: 25.000 × 12KB ≈ **300 MB** KV + 19MB GDN state ≈ 320 MB.
- 20 session: **~6.4 GB**. Pool ~10GB → **đủ cho cả 20 session, KHÔNG bị memory-bound**
  ở mức 0.90 trên 18GB.

> **Kết luận then chốt:** vì GQA (chỉ 2 KV head) + chỉ 6 layer full-attn, KV ở đây
> **vốn đã rất nhỏ**. Bộ nhớ gần như **không phải** ràng buộc thật sự của bài này →
> các đòn bẩy về bộ nhớ (KV quant, tăng util) **ít tác dụng hơn** các đòn bẩy về
> **tốc độ prefill**. Đây là điều cần nhớ khi xếp ưu tiên.

### 1.3 Vì sao phải evict

Pool ③ chốt cứng lúc boot. Trong pool có 2 loại cư dân: **active** (request đang
chạy — không được đưa) và **cached prefix** (KV của request đã xong, giữ để tái
dùng — **evictable LRU**). Khi block đầy → evict cached prefix cũ → turn sau đáng
lẽ hit thì miss → **re-prefill → TTFT vọt**. Nếu active cũng thiếu → **preemption**
(`vllm:num_preemptions>0`) → phạt nặng.

### 1.4 Ba đòn bẩy gốc (chỉ có 3, mỗi giải pháp là một nhánh con)

```
A. KHÔNG TÍNH LẠI  →  tái dùng KV (prefix caching)         [thắng round 2–6]
B. TÍNH NHANH HƠN  →  cùng việc, ít thời gian (FP8, kernel, batch, CUDA graph)  [thắng round 1]
C. GIỮ ĐƯỢC CACHE  →  quản bộ nhớ để A còn tác dụng          [nền tảng]
```

⚠️ **Round 1 không có gì để cache → đòn bẩy A = 0 → chỉ B cứu được.** Mà round 1 là
nơi mất điểm nhất. Nên **B (tốc độ prefill thô) là thứ quyết định thứ hạng**, không
phải cache. "Ai cache tốt hơn" chỉ thắng từ round 2 trở đi.

---

## 2. KV cache quantization (`--kv-cache-dtype`)

KV cache cũng chỉ là số — lưu BF16 (2 byte) hay **FP8 (1 byte)** → **halves bộ nhớ
mỗi token cached**, lưu được nhiều token hơn → ít evict, giữ prefix lâu hơn.

**Cách dùng (vLLM):**
- `--kv-cache-dtype fp8` (mặc định `fp8_e4m3`; có `fp8_e5m2`). `auto` = giữ nguyên dtype model.
- `--kv-cache-dtype-skip-layers <...>` : giữ một số layer ở BF16 (vd sliding-window). Model này **không có sliding window** nên ít dùng.
- `--calculate-kv-scales` : tự ước lượng scale lúc warmup. Mặc định scale=1.0 (uncalibrated). Muốn chuẩn hơn → calibrate bằng **llm-compressor** (per-tensor / per-head scale).

**Lợi ích:** giảm 50% bộ nhớ KV; decode ITL còn ~54% BF16; hỗ trợ context dài hơn.
Accuracy tụt rất ít (thường 1–2 điểm, long-context recover 94–98%).

### ⚠️ Nhưng tại sao ở BÀI NÀY, KV FP8 có thể KHÔNG đáng / thậm chí HẠI

1. **`head_dim = 256`** — blog vLLM FP8-KV cảnh báo rõ: model `head_dim=256` khi
   **prefill quan trọng** thì FP8 KV làm **tăng overhead TTFT** (two-level
   accumulation). Mà TTFT chính là metric mất điểm của mình → **rủi ro ngược**.
2. **KV vốn đã nhỏ** (§1.2): tiết kiệm 3GB trên pool 10GB không phải nút thắt →
   lợi ích bộ nhớ không đáng kể ở đây.
3. **Break-even ~7k token**: context mình 13–27k thì qua ngưỡng, nhưng lợi ích
   chính là ở decode/bộ nhớ — cả hai đều **không** phải cho mình đang cần.

→ **Khuyến nghị:** vẫn **A/B test** `--kv-cache-dtype fp8` (đo TTFT trước/sau bằng
script 07 + 08), nhưng **đừng kỳ vọng** đây là đòn bẩy lớn; nhiều khả năng **net
tiêu cực cho TTFT**. Ưu tiên **FP8 *weight*** (§4) trước — nó vừa nhanh prefill vừa
giải phóng VRAM thật sự.

> Lưu ý phân biệt: **FP8 weight** (`--quantization`, quantize trọng số) khác **FP8
> KV cache** (`--kv-cache-dtype`, quantize cache). Hai thứ độc lập, có thể bật riêng.

Model GDN còn có cache riêng: `--mamba-cache-dtype` / `--mamba-ssm-cache-dtype`
lặng lẽ quantize state 18 layer GDN — nhưng state chỉ ~19MB/seq nên lợi ích bộ nhớ
càng nhỏ; chú ý độ chính xác.

---

## 3. Checklist điểm nghẽn (chỉ vLLM)

- [ ] **① Cold-start round 1** — 20×~13k token dồn, prefill thuần, không cache cứu. Mất điểm nhất.
- [ ] **② Prefix caching hybrid CÓ RỦI RO** — vLLM prefix caching cho GDN/Mamba-hybrid là **experimental + có bug ghi nhận** (block-size ép ~528–544 token để align Mamba; chỉ cache block hoàn chỉnh → đuôi prefix mất cache). Phải verify, đừng tin mặc định.
- [ ] **③ Prefill compute-bound** — 1.37B GEMM BF16, quyết định TTFT.
- [ ] **④ Eviction/ngân sách VRAM** — thực tế **không phải** nút thắt ở 18GB (KV nhỏ), nhưng vẫn phải giữ `num_preemptions=0` và không evict giữa round.
- [ ] **⑤ Nghẽn CPU (3 core/8GB)** — tokenize 20 prompt dài + overhead HTTP/streaming có thể cộng trăm ms vào TTFT vô hình.
- [ ] **⑥ TPOT dưới burst** — cửa hẹp 20–45ms; batch prefill to → decode-step phình → TPOT vọt.
- [ ] **⑦ Accuracy gate (nhân)** — FP8 phải giữ GPQA ≥ ~0.30, không mất trắng.
- [ ] **⑧ Startup < 15 phút** — CUDA graph compile + warmup không được rớt healthcheck.

---

## 4. Checklist giải pháp (chỉ vLLM) — ánh xạ tới cờ

> Ưu tiên theo đòn bẩy. Vì prefill-bound, nhóm B (tính nhanh) và A (cache) ăn điểm nhất.

### 🟠 B — Tính prefill nhanh hơn (thắng round 1)
- [ ] **FP8 weight quant** *(nghẽn ①③)* — `--quantization` + checkpoint FP8 (tạo bằng **llm-compressor**, ignore `lm_head`+layer nhạy). Tăng tốc GEMM `*_proj` **và** giải phóng ~2.2GB. ⚠️ verify GPQA giữ margin. Bake vào image (cấm pull runtime).
- [ ] **Chunked prefill + sweep** *(nghẽn ①⑥)* — `--enable-chunked-prefill`, `--max-num-batched-tokens` (to → TTFT tốt, canh TPOT), `--long-prefill-token-threshold`, `--max-num-partial-prefills` / `--max-long-partial-prefills`.
- [ ] **Bỏ vision tower** *(nghẽn ④)* — `--language-model-only` (nếu có) để bỏ xử lý multimodal / `--skip-mm-profiling`; hoặc `--limit-mm-per-prompt image=0`. Trả VRAM về KV, bớt compute.
- [ ] **GDN prefill backend** — `--gdn-prefill-backend flashinfer|triton` chọn kernel nhanh cho 18 layer GDN (riêng cho arch này).
- [ ] **CUDA graphs** *(nghẽn ⑥)* — **KHÔNG** `--enforce-eager` (giữ CUDA graph); tính `--cudagraph-capture-sizes` / `--max-cudagraph-capture-size` phủ batch 20–32.
- [ ] **Attention backend** — `--attention-backend` (FlashAttention/FlashInfer), `--enable-flashinfer-autotune`.

### 🔵 A — Tái dùng cache (thắng round 2–6)
- [ ] **Bật + verify prefix caching** *(nghẽn ②)* — `--enable-prefix-caching`; đo thật sự ăn bằng `07_per_request_report.py` (`cache_rd/hit%`). Thử `--prefix-caching-hash-algo`.
- [ ] **Hạ max-model-len** *(nghẽn ④)* — `--max-model-len 32768` (từ 262144) → engine cấp KV sát nhu cầu.
- [ ] **Block-size / hybrid manager** *(nghẽn ②)* — `--block-size`, `--mamba-block-size`; cân nhắc `--disable-hybrid-kv-cache-manager` để so sánh (bug align-mode). `--kv-sharing-fast-prefill` nếu có.

### 🟢 C — Giữ cache / quản bộ nhớ
- [ ] **`--gpu-memory-utilization 0.95`** — pool to hơn (nhưng §1.2: không phải nút thắt ở đây).
- [ ] **FP8 weight** (trùng nhóm B) → khối ① nhỏ → pool to → đòn kép.
- [ ] **`--kv-cache-dtype fp8`** — A/B test, **nhưng xem cảnh báo §2** (head_dim=256 hại TTFT).
- [ ] **Theo dõi `--kv-cache-metrics`** + `num_preemptions=0`.

### ⚙️ Serving/CPU (rẻ, dễ bỏ sót)
- [ ] `--disable-log-stats` + tắt request logging → giảm tải 3 core.
- [ ] Đảm bảo **streaming thật** (server buffer output → TTFT đo được tệ oan).
- [ ] **Warmup lúc boot** bằng request tự tạo (**KHÔNG dùng nội dung trace** — luật cấm pre-compute) để compile CUDA graph + nóng allocator trước healthcheck.

### 🟢 Decode/TPOT (ưu tiên thấp — đang ổn)
- [ ] Speculative decoding qua **MTP head**: `--speculative-config` / `--spec-method` / `--spec-tokens`. CHỈ khi đo TPOT thực >20ms (đạt Floor rồi thì không thêm điểm). Rủi ro.

---

## 4b. FP8 weight — kế hoạch chi tiết theo tensor (ignore-list)

Đây là phần việc thật sự của FP8 (tốc độ FP8 đến từ hardware tensor core + kernel
của engine, KHÔNG phải từ việc tự quantize): **chọn đúng tensor nào quantize, tensor
nào giữ**. Nguyên tắc: chỉ FP8 các **ma
trận GEMM lớn** (`*_proj`) — nơi tập trung FLOPs + bộ nhớ; **giữ nguyên** các tensor
nhỏ/nhạy (norm, bias, gate, conv, tham số SSM, embedding/lm_head). Quantize mấy cái
nhỏ gần như không tiết kiệm gì mà dễ làm **tụt accuracy → rơi khỏi cổng GPQA**.

Kế hoạch dưới đây đọc từ header safetensors thật (xem `docs/qwen35-architecture.html`,
mục Phụ lục tensor).

### GDN layer (×18)
| Tensor | Shape | Dtype | Kế hoạch |
|---|---|---|---|
| `linear_attn.in_proj_qkv` | [6144, 2048] | BF16 | **FP8** |
| `linear_attn.in_proj_z` | [2048, 2048] | BF16 | **FP8** |
| `linear_attn.out_proj` | [2048, 2048] | BF16 | **FP8** |
| `mlp.{gate,up}_proj` | [6144, 2048] | BF16 | **FP8** |
| `mlp.down_proj` | [2048, 6144] | BF16 | **FP8** |
| `linear_attn.in_proj_a / _b` | [16, 2048] | BF16 | keep (nhỏ, cổng SSM) |
| `linear_attn.conv1d.weight` | [6144, 1, 4] | BF16 | keep (short conv) |
| `linear_attn.A_log` | [16] | F32 | keep fp32 (động lực hồi quy) |
| `linear_attn.dt_bias` | [16] | BF16 | keep |
| `linear_attn.norm.weight` | [128] | F32 | keep fp32 |
| `input / post_attention_layernorm` | [2048] | BF16 | keep |

### Full-attention layer (×6)
| Tensor | Shape | Dtype | Kế hoạch |
|---|---|---|---|
| `self_attn.q_proj` | [4096, 2048] | BF16 | **FP8** |
| `self_attn.{k,v}_proj` | [512, 2048] | BF16 | **FP8** |
| `self_attn.o_proj` | [2048, 2048] | BF16 | **FP8** |
| `self_attn.{q,k}_norm` | [256] | BF16 | keep (QK-norm nhạy) |
| `mlp.{gate,up,down}_proj` | [6144, 2048] | BF16 | **FP8** |

### Dùng chung
| Tensor | Shape | Note |
|---|---|---|
| `embed_tokens / lm_head` | [248320, 2048] | tied · **giữ BF16** (0.51B) — quantize logits hại accuracy nhiều |
| `mtp.*` | 15 tensors | draft head speculative (0.06B) · keep |
| `model.visual.*` | Qwen3_5VisionModel | 0.33B · không dùng cho text · keep hoặc bỏ nạp |

### Vì sao "keep" các tensor nhỏ
- **Norm (`*norm*`), layernorm, `q/k_norm`**: chỉ [128]–[2048], là hệ số scale rất
  nhạy về số học; FP8 chúng ≈ 0 lợi ích bộ nhớ nhưng dễ lệch phân phối.
- **`A_log`, `dt_bias`, `conv1d`, `in_proj_a/_b`**: tham số điều khiển động lực hồi
  quy của GDN/SSM (state evolution). Rất nhạy — sai một chút là trôi cả chuỗi. Kích
  thước tí hon nên giữ nguyên là "free".
- **`embed_tokens/lm_head` (tied)**: đây là chiếu ra logits vocab 248k; quantize làm
  hỏng phân phối output → tụt accuracy. Giữ BF16 (0.51B, ~1GB — đáng).

### Recipe llm-compressor (ignore-list cụ thể)
Quantize mọi `Linear` **trừ** danh sách ignore (các tensor không-Linear như norm /
A_log / conv1d tự động bị bỏ qua, nhưng liệt kê cho chắc):

```python
# oneshot FP8 W8A8 (channel-wise), data-free
from llmcompressor.modifiers.quantization import QuantizationModifier
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",        # W8A8 FP8, per-channel weight, dynamic act
    ignore=[
        "lm_head",               # tied embedding → giữ BF16
        "re:.*in_proj_a",        # cổng SSM nhỏ, nhạy
        "re:.*in_proj_b",
        "re:.*visual.*",         # vision tower (không dùng)
        "re:.*mtp.*",            # draft head
        # norms/A_log/dt_bias/conv1d không phải Linear → tự bỏ qua
    ],
)
```

### Serve trong vLLM
- Checkpoint FP8 (compressed-tensors) → vLLM tự nhận qua metadata; hoặc ép
  `--quantization compressed-tensors`.
- **Bake checkpoint FP8 thẳng vào image** (luật cấm pull runtime).
- **Verify GPQA** sau khi quantize, giữ margin ≥5 điểm so với ngưỡng 0.30. Nếu tụt,
  mở rộng ignore-list (vd giữ thêm `down_proj` hoặc layer đầu/cuối).

**Tác động ước lượng:** ~1.37B GEMM `*_proj` xuống FP8 → giải phóng ~1.3–2.2GB VRAM
**và** tăng tốc GEMM prefill (đòn bẩy B — thắng round-1). `lm_head` 0.51B + vision
0.33B vẫn BF16 (hoặc bỏ vision).

---

## 5. THAM CHIẾU CỜ vLLM ĐẦY ĐỦ (từ source `arg_utils.py`)

> ⚠️ Tập cờ thay đổi theo version. **Verify bằng `vllm serve --help` trên đúng image
> của bạn** (baseline BTC: `vllm/vllm-openai:v0.22.1`). Đánh dấu: 🎯 = đang để ý cho bài này.

### (1) Model & loading
`--model` · `--tokenizer` · `--tokenizer-mode` · `--trust-remote-code` · `--dtype` 🎯(bf16/auto) · `--seed` · `--max-model-len` 🎯 · `--served-model-name` 🎯 · `--load-format` 🎯 · `--download-dir` · `--revision` · `--hf-config-path` · `--config-format` · `--model-impl` · `--override-attention-dtype` · `--override-generation-config` · `--generation-config` · `--safetensors-load-strategy` / `--safetensors-prefetch-num-threads` / `--safetensors-prefetch-block-size` 🎯(tốc độ load → startup) · `--ignore-patterns` · `--model-weights` · `--hf-overrides` · `--skip-tokenizer-init`

### (2) Parallelism (single GPU → hầu hết = 1)
`--tensor-parallel-size/-tp` 🎯(=1) · `--pipeline-parallel-size/-pp` · `--data-parallel-size/-dp` · `--decode-context-parallel-size/-dcp` · `--prefill-context-parallel-size/-pcp` · `--enable-expert-parallel/-ep` (MoE — n/a) · `--distributed-executor-backend` · `--max-parallel-loading-workers` 🎯(startup) · `--worker-cls` · `--numa-bind` · `--device-ids` · (nhiều cờ DP/EP multi-node khác — không liên quan single GPU)

### (3) Memory & KV cache
`--gpu-memory-utilization` 🎯 · `--kv-cache-dtype` 🎯 · `--kv-cache-dtype-skip-layers` 🎯 · `--calculate-kv-scales` 🎯 · `--block-size` 🎯 · `--kv-cache-memory-bytes` · `--num-gpu-blocks-override` 🎯 · `--cpu-offload-gb` · `--cpu-offload-params` · `--kv-offloading-size` / `--kv-offloading-backend` · `--kv-sharing-fast-prefill` 🎯 · `--swap-space` (best_of>1; có thể =0)

### (4) Prefix caching
`--enable-prefix-caching` 🎯 (`--no-enable-prefix-caching` để tắt) · `--prefix-caching-hash-algo` 🎯

### (5) Scheduling / batching / chunked prefill
`--max-num-batched-tokens` 🎯 · `--max-num-seqs` 🎯 · `--enable-chunked-prefill` 🎯 · `--long-prefill-token-threshold` 🎯 · `--max-num-partial-prefills` 🎯 · `--max-long-partial-prefills` 🎯 · `--scheduling-policy` 🎯(fcfs/priority) · `--scheduler-reserve-full-isl` · `--watermark` · `--prefill-schedule-interval` · `--disable-hybrid-kv-cache-manager` 🎯(so sánh cho hybrid) · `--async-scheduling` · `--stream-interval` 🎯 · `--scheduler-cls`

### (6) Quantization (weight)
`--quantization/-q` 🎯 · `--quantization-config` 🎯 · `--allow-deprecated-quantization`

### (7) CUDA graphs & compilation
`--enforce-eager` 🎯(TẮT nó để giữ CUDA graph) · `--cudagraph-capture-sizes` 🎯 · `--max-cudagraph-capture-size` 🎯 · `--compilation-config/-cc` 🎯 · `--optimization-level` · `--performance-mode`

### (8) Attention & kernels
`--attention-backend` 🎯 · `--attention-config/-ac` · `--enable-flashinfer-autotune` 🎯 · `--linear-backend` 🎯(GDN linear) · `--ir-op-priority` · `--kernel-config`

### (9) Mamba / GDN (riêng arch này)
`--mamba-backend` 🎯(triton/…) · `--gdn-prefill-backend` 🎯(flashinfer/triton) · `--mamba-cache-dtype` 🎯 · `--mamba-ssm-cache-dtype` 🎯 · `--mamba-block-size` 🎯 · `--mamba-cache-mode` 🎯 · `--enable-mamba-cache-stochastic-rounding`

### (10) Speculative decoding (MTP)
`--speculative-config/-sc` 🎯 · `--spec-method` 🎯 · `--spec-model` · `--spec-tokens` 🎯

### (11) Multimodal (bỏ vision)
`--language-model-only` 🎯(bỏ xử lý multimodal) · `--limit-mm-per-prompt` 🎯 · `--skip-mm-profiling` 🎯 · `--mm-encoder-*` (nhiều cờ encoder) · `--media-io-kwargs` · `--mm-processor-kwargs`

### (12) Logging / API / observability
`--disable-log-stats` 🎯 · `--kv-cache-metrics` 🎯 / `--kv-cache-metrics-sample` · `--cudagraph-metrics` · `--otlp-traces-endpoint` · `--collect-detailed-traces` · `--enable-logging-iteration-details` · `--show-hidden-metrics-for-version` · `--enable-mfu-metrics`

### (13) Misc
`--max-logprobs` 🎯 · `--disable-cascade-attn` 🎯 · `--disable-sliding-window` (n/a) · `--enable-sleep-mode` · `--enable-cumem-allocator` · `--shutdown-timeout` · `--tokens-only` · `--enable-prompt-embeds` · `--logits-processors` · `--kv-transfer-config` / `--kv-events-config` (disaggregated — n/a single GPU) · `--additional-config`

### (14) LoRA / structured outputs / diffusion — không liên quan bài này
`--enable-lora` … · `--reasoning-parser` · `--structured-outputs-config` · `--diffusion-config` …

---

## 6. Cấu hình khởi điểm đề nghị (rồi sweep)

```bash
vllm serve <model> \
  --served-model-name qwen3.5-2b \
  --dtype bfloat16 \
  --max-model-len 32768 \              # từ 262144 → cấp KV sát nhu cầu
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \            # verify bằng script 07 (cache_rd/hit%)
  --enable-chunked-prefill \
  --max-num-batched-tokens 16384 \     # sweep 8192↔32768 → tối ưu TTFT, canh TPOT
  --max-num-seqs 32 \                  # ≥ 20 concurrent + headroom
  --disable-log-stats
  # --language-model-only              # nếu có: bỏ vision tower
  # --quantization <fp8-method>        # ĐÒN BẨY LỚN NHẤT (checkpoint FP8 bake sẵn) — verify GPQA
  # --kv-cache-dtype fp8               # A/B ONLY — cảnh báo head_dim=256 hại TTFT (§2)
  # --gdn-prefill-backend flashinfer   # nếu có, cho 18 layer GDN
```

Kỷ luật đo: **mỗi lần đổi 1 biến**, ghi lại bằng `config ↔ ERS (script 08) ↔ GPQA`,
so với baseline. Ưu tiên: **verify prefix caching → FP8 weight → sweep chunked
prefill → bỏ vision/CPU hygiene → (KV quant, spec decode) chỉ khi đo có lợi thật**.

**Sources:** [vLLM engine args (arg_utils.py)](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py) · [Quantized KV cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/) · [FP8 KV-cache state (2026-04)](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-22-fp8-kvcache.md) · [Optimization & tuning](https://docs.vllm.ai/en/stable/configuration/optimization/) · [Qwen3.5 prefix-cache block-size bug #40696](https://github.com/vllm-project/vllm/issues/40696) · [Prefix caching for hybrid #26201](https://github.com/vllm-project/vllm/issues/26201) · [LLM Compressor FP8](https://developers.redhat.com/articles/2025/10/07/llm-compressor-080-extended-support-qwen3)
