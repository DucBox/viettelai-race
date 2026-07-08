# Tham chiếu cờ vLLM — giải thích từng cờ (kỹ thuật + ứng dụng)

Từ điển tra cứu **mọi cờ `vllm serve`** (verify theo `--help` thật của image
`vllm/vllm-openai:v0.22.1`). Mỗi cờ có **2 phần**:
- **Kỹ thuật:** nó làm gì bên trong.
- **Ứng dụng:** đổi nó ảnh hưởng gì cho **bài Track 3** (single GPU, prefill-bound,
  điểm = TTFT + TPOT, decode đang ổn).

Đánh dấu: 🎯 = đáng chỉnh cho bài này · ⚪ = có thể liên quan · ⬛ = không áp dụng.
Chiến lược tổng thể xem `docs/vllm-optimization-guide.md`; file này chỉ tra cờ.

---

## 1. CacheConfig — Bộ nhớ & KV cache (nhóm quan trọng nhất)

**🎯 `--gpu-memory-utilization FLOAT`** (default 0.90)
- *Kỹ thuật:* tỉ lệ VRAM vLLM được phép dùng. Sau khi trừ weights + activation/framework, phần còn lại = **KV cache pool**.
- *Ứng dụng:* cao hơn → pool KV to hơn → chứa nhiều token cache hơn, ít evict. Với bài này KV vốn nhỏ nên không phải nút thắt; nhưng để mimic đúng ngân sách 18GB MIG phải **scale theo VRAM card** (xem `.env.example`).

**🎯 `--kv-cache-dtype {auto,fp8,fp8_e4m3,fp8_e5m2,fp8_per_token_head,int8_per_token_head,nvfp4,...}`** (default auto)
- *Kỹ thuật:* dtype lưu KV cache. `auto` = theo model (BF16). `fp8` = `fp8_e4m3`, 1 byte/phần tử → **halves bộ nhớ KV/token**.
- *Ứng dụng:* ⚠️ với model `head_dim=256` này, FP8 KV **tăng overhead TTFT** (two-level accumulation) → có thể **hại** đúng metric mình cần. KV lại đã nhỏ → lợi ích bộ nhớ thấp. A/B test, đừng mặc định bật.

**🎯 `--calculate-kv-scales`** (default off)
- *Kỹ thuật:* tự ước lượng scale FP8 cho KV lúc warmup (thay vì scale=1.0).
- *Ứng dụng:* chỉ có ý nghĩa khi đã bật `--kv-cache-dtype fp8`; giúp giữ accuracy KV-FP8 tốt hơn.

**🎯 `--kv-cache-dtype-skip-layers LAYERS...`**
- *Kỹ thuật:* danh sách layer giữ KV ở dtype gốc (không FP8) — vd layer sliding-window.
- *Ứng dụng:* model này không có sliding-window nên ít dùng; có thể giữ vài layer full-attn nhạy ở BF16 nếu FP8-KV làm tụt chất lượng.

**🎯 `--block-size INT`** (CUDA ≤ 32 cho attention thường; hybrid tự ép ~544)
- *Kỹ thuật:* số token mỗi KV block (đơn vị cấp phát KV). Prefix caching hoạt động **theo block**.
- *Ứng dụng:* liên quan bug prefix-cache hybrid (chỉ cache block hoàn chỉnh). Đổi để tinh chỉnh granularity cache; thường để mặc định.

**🎯 `--kv-cache-memory-bytes INT`**
- *Kỹ thuật:* ép cỡ KV pool = số byte cố định (thay vì suy từ util).
- *Ứng dụng:* set KV pool chính xác để tái lập điều kiện thi, không phụ thuộc VRAM card.

**⚪ `--num-gpu-blocks-override INT`**
- *Kỹ thuật:* ép số KV block, bỏ qua profiling tự động.
- *Ứng dụng:* dùng khi muốn cố định dung lượng KV để so sánh công bằng giữa các lần đo.

**🎯 `--enable-prefix-caching` / `--no-enable-prefix-caching`**
- *Kỹ thuật:* bật tái dùng KV của prefix chung giữa các request.
- *Ứng dụng:* **điều kiện tiên quyết** để turn 2–6 nhanh (tái dùng lịch sử). Luôn bật; verify thực sự ăn bằng `hit%` (script 07).

**⚪ `--prefix-caching-hash-algo {sha256,sha256_cbor,xxhash,xxhash_cbor}`**
- *Kỹ thuật:* thuật toán băm để nhận diện prefix trùng. `xxhash` nhanh hơn, `sha256` an toàn hơn.
- *Ứng dụng:* `xxhash` giảm chút overhead CPU khi băm prefix dài — nhỏ nhưng free với 3-core.

**⚪ `--kv-sharing-fast-prefill`**
- *Kỹ thuật:* tối ưu chia sẻ KV để prefill nhanh hơn (khi các request share KV).
- *Ứng dụng:* đáng thử cho workload prefix chung to như trace này (đo TTFT trước/sau).

**⬛ `--swap-space GiB`** (default 4)
- *Kỹ thuật:* RAM CPU dùng làm swap cho KV khi `best_of>1`.
- *Ứng dụng:* trace dùng `best_of=1` → có thể để 0 (bớt dùng 8GB RAM ít ỏi).

---

## 2. SchedulerConfig — Lập lịch, batching, chunked prefill

**🎯 `--max-num-batched-tokens INT`**
- *Kỹ thuật:* trần token xử lý mỗi iteration (gộp prefill + decode).
- *Ứng dụng:* **cần sweep**. To hơn → gom nhiều prefill/iteration → **TTFT tốt hơn** (cứu round-1 burst); nhưng decode-step phình → **dọa TPOT**. Bài prefill-bound → nghiêng lớn (8k–32k), canh cửa TPOT 45ms.

**🎯 `--max-num-seqs INT`**
- *Kỹ thuật:* số sequence tối đa chạy đồng thời trong 1 batch.
- *Ứng dụng:* trace có 20 user song song → đặt ≥ 20 (32 cho margin). Thấp quá → request bị xếp hàng → TTFT tăng.

**🎯 `--enable-chunked-prefill` / `--no-...`**
- *Kỹ thuật:* cắt prefill dài thành chunk, xen với decode trong cùng batch.
- *Ứng dụng:* làm mượt TTFT khi prefill dài (13k+) tranh GPU với decode; giảm p95 TTFT. Nên bật, rồi tinh chỉnh qua `max-num-batched-tokens`.

**🎯 `--long-prefill-token-threshold INT`**
- *Kỹ thuật:* ngưỡng token để coi 1 prefill là "dài" (được chunk/ưu tiên khác).
- *Ứng dụng:* trace toàn prefill dài → điều chỉnh ngưỡng đổi cách scheduler đối xử với chúng.

**🎯 `--max-num-partial-prefills INT`** · **🎯 `--max-long-partial-prefills INT`**
- *Kỹ thuật:* số prefill (dài) được cắt-dở-dang chạy đồng thời.
- *Ứng dụng:* điều tiết bao nhiêu prefill dài chen cùng lúc → cân bằng TTFT (round-1 burst) vs TPOT (decode bị chen).

**🎯 `--scheduling-policy {fcfs,priority}`** (default fcfs)
- *Kỹ thuật:* thứ tự phục vụ. `fcfs` = đến trước phục vụ trước; `priority` = theo priority field.
- *Ứng dụng:* trace có timestamp cố định → `fcfs` khớp tự nhiên. `priority` chỉ hữu ích nếu muốn ưu tiên request nào đó (thường không).

**🎯 `--disable-hybrid-kv-cache-manager`**
- *Kỹ thuật:* tắt trình quản lý KV lai (quản chung KV attention + state Mamba). Quay về cơ chế cũ.
- *Ứng dụng:* lever để **A/B khi nghi ngờ prefix-cache hybrid có bug** — so hit%/TTFT có/không hybrid manager.

**⚪ `--async-scheduling`**
- *Kỹ thuật:* lập lịch bất đồng bộ (overlap scheduling với compute).
- *Ứng dụng:* giảm bong bóng CPU-GPU → nhẹ TTFT/throughput. Log của bạn đã bật sẵn.

**⚪ `--scheduler-reserve-full-isl`**
- *Kỹ thuật:* đặt trước đủ block cho toàn bộ ISL của request ngay khi nhận.
- *Ứng dụng:* giảm nguy cơ preempt giữa chừng cho prompt dài, đổi lại tốn KV hơn.

**⚪ `--stream-interval INT`**
- *Kỹ thuật:* mỗi bao nhiêu token thì flush 1 chunk SSE ra client.
- *Ứng dụng:* =1 → token ra ngay (TTFT/streaming mượt); lớn hơn → gom, giảm overhead nhưng tệ cảm giác realtime. Giữ nhỏ để TTFT đo đúng.

**⚪ `--watermark FLOAT`** · **⚪ `--prefill-schedule-interval`** · **⚪ `--scheduler-cls`**
- *Kỹ thuật:* ngưỡng nước cho scheduler / nhịp lên lịch prefill / class scheduler tùy biến.
- *Ứng dụng:* tinh chỉnh nâng cao; thường để mặc định.

---

## 3. CompilationConfig + preset — CUDA graph & tối ưu cấp cao

**🎯 `--enforce-eager` / `--no-enforce-eager`** (default off = dùng CUDA graph)
- *Kỹ thuật:* `--enforce-eager` TẮT CUDA graph, chạy eager từng op.
- *Ứng dụng:* **GIỮ TẮT** (tức để CUDA graph bật) — CUDA graph giảm overhead mỗi bước decode → bảo vệ TPOT. Chỉ bật eager khi debug.

**🎯 `--cudagraph-capture-sizes N...`** · **🎯 `--max-cudagraph-capture-size N`**
- *Kỹ thuật:* các batch-size được "chụp" thành CUDA graph (graph chỉ dùng lại cho đúng size đã chụp).
- *Ứng dụng:* đảm bảo phủ các batch-size thực tế (20–40 concurrent) → decode chạy trên graph đã chụp thay vì eager. Log của bạn chụp `[1,2,4,...,64]`.

**🎯 `--performance-mode {balanced,interactivity,throughput}`**
- *Kỹ thuật:* preset 1 cờ chỉnh loạt tham số theo hướng tối ưu (latency-thấp vs throughput-cao).
- *Ứng dụng:* thử `throughput` — bài prefill-bound có thể hưởng lợi; đo ERS trước/sau. Nhanh gọn để dò hướng.

**⚪ `--optimization-level INT`** · **⚪ `--compilation-config JSON`**
- *Kỹ thuật:* mức tối ưu compile / config chi tiết (mode, inductor passes, fusion...).
- *Ứng dụng:* nâng cao — chỉnh fusion/inductor cho arch lai. Để sau khi các lever to đã cạn.

---

## 4. Quantization (weight) — đòn bẩy prefill lớn nhất

**🎯 `--quantization / -q METHOD`** (vd `compressed-tensors`, `fp8`, ...)
- *Kỹ thuật:* phương pháp quantize **trọng số**. Với checkpoint FP8 (compressed-tensors) vLLM tự nhận; hoặc ép method.
- *Ứng dụng:* **đòn bẩy B lớn nhất** — FP8 W8A8 tăng tốc GEMM `*_proj` (compute-bound prefill) **và** giảm ~2GB weights. Cần checkpoint FP8 bake sẵn (xem §4b optimization-guide). Verify GPQA.

**⚪ `--quantization-config JSON`** · **⬛ `--allow-deprecated-quantization`**
- *Kỹ thuật:* config quantize chi tiết / cho phép method cũ.
- *Ứng dụng:* dùng khi cần override cấu hình quant; hiếm khi cần.

---

## 5. MambaConfig / GDN — riêng kiến trúc lai này

**🎯 `--mamba-cache-mode {align,all,none}`** (default `align` khi bật prefix caching)
- *Kỹ thuật:* chiến lược snapshot **state hồi quy** của 18 layer GDN để prefix-cache. `align` = tối ưu (Marconi-style, experimental); `all` = lưu full state mọi block (cơ bản); `none` = tắt cache Mamba.
- *Ứng dụng:* **lever A/B hàng đầu** — `align` mặc định đang experimental + có perf issue → thử `all` xem `hit%`/TTFT ổn hơn không. `none` để cô lập xem cache Mamba có đang giúp/hại.

**🎯 `--gdn-prefill-backend {flashinfer,triton,cutedsl}`** (default triton theo log)
- *Kỹ thuật:* kernel tính prefill cho layer Gated DeltaNet.
- *Ứng dụng:* 18/24 layer là GDN → kernel này ảnh hưởng lớn tới TTFT. Thử `flashinfer`/`cutedsl` vs `triton`, đo TTFT.

**⚪ `--mamba-cache-dtype {auto,bfloat16,float16,float32}`** · **⚪ `--mamba-ssm-cache-dtype`**
- *Kỹ thuật:* dtype lưu state GDN / state SSM.
- *Ứng dụng:* hạ precision state → tiết kiệm chút bộ nhớ (state chỉ ~19MB/seq nên lợi ích nhỏ); coi chừng độ chính xác động lực hồi quy.

**⚪ `--mamba-block-size INT`**
- *Kỹ thuật:* granularity block cho cache state Mamba.
- *Ứng dụng:* liên quan bug block-size ~544; chỉnh để tinh chỉnh cache GDN.

**⬛ `--mamba-backend` · `--enable-mamba-cache-stochastic-rounding` · `--mamba-cache-philox-rounds`**
- *Kỹ thuật:* backend Mamba / làm tròn ngẫu nhiên state (giảm sai số tích luỹ).
- *Ứng dụng:* nâng cao, hiếm chỉnh; stochastic rounding chủ yếu cho training/độ ổn định dài.

---

## 6. AttentionConfig / KernelConfig — kernel

**🎯 `--attention-backend BACKEND`** (FLASH_ATTN / FLASHINFER / TRITON_ATTN / FLEX_ATTENTION)
- *Kỹ thuật:* kernel cho 6 layer full-attention. Log của bạn dùng `FLASH_ATTN` (FA2).
- *Ứng dụng:* FlashInfer đôi khi nhanh hơn cho prefill dài; A/B đo TTFT.

**🎯 `--enable-flashinfer-autotune`**
- *Kỹ thuật:* tự dò cấu hình kernel FlashInfer tối ưu lúc warmup.
- *Ứng dụng:* có thể nhặt thêm ít tốc độ prefill; đổi lại warmup lâu hơn (canh cửa 15 phút).

**⚪ `--linear-backend {triton,cutlass,marlin,machete,...}`**
- *Kỹ thuật:* kernel cho các lớp Linear (GEMM). Quan trọng khi quantize (marlin/machete cho INT4, cutlass cho FP8).
- *Ứng dụng:* sau khi FP8, chọn backend GEMM khớp để lấy đúng tốc độ. `auto` thường ổn.

**⬛ `--moe-backend ...`** — MoE, model này Dense → N/A. **⬛ `--ir-op-priority` · `--kernel-config`** — nâng cao.

---

## 7. ModelConfig — model & sinh

**🎯 `--max-model-len INT`** (config gốc 262144)
- *Kỹ thuật:* độ dài chuỗi tối đa engine cấp phát/hỗ trợ.
- *Ứng dụng:* hạ về ~32768 (đủ 27k prompt + 200 out) → engine không phải dự trù cho 262k → tính KV/concurrency sát nhu cầu. Log "Max concurrency 3.83x" là do để 262144.

**🎯 `--dtype {auto,bfloat16,float16,float32,half}`** (default auto → bf16)
- *Kỹ thuật:* dtype tính toán của model (khi không quantize).
- *Ứng dụng:* giữ `bfloat16` (gốc BF16). FP8 làm qua `--quantization`, không phải cờ này.

**🎯 `--enforce-eager`** — (xem §3, thuộc cả ModelConfig).

**🎯 `--served-model-name NAME`**
- *Kỹ thuật:* tên model client gọi qua API.
- *Ứng dụng:* giữ ổn định `qwen3.5-2b` để AIPerf/BTC target đúng.

**⚪ `--seed INT`** · **⚪ `--max-logprobs INT`** · **⚪ `--logprobs-mode ...`**
- *Kỹ thuật:* seed sinh / số logprobs trả về / chế độ tính logprob.
- *Ứng dụng:* trace set seed=42 per-request (qua body). `max-logprobs` chỉ cần nếu chấm accuracy bằng logprob.

**⚪ `--disable-cascade-attn`**
- *Kỹ thuật:* tắt cascade attention (tối ưu attention khi nhiều request share prefix dài).
- *Ứng dụng:* trace share prefix to → cascade attn CÓ THỂ giúp; nếu nghi nó gây lỗi/chậm thì tắt để so. Đáng A/B.

**⚪ `--hf-overrides JSON`** · **⚪ `--override-attention-dtype`** · **⚪ `--override-generation-config`**
- *Kỹ thuật:* ghi đè config HF / dtype attention / config sinh.
- *Ứng dụng:* nâng cao — vd ép rope scaling, đổi dtype attention cục bộ. Cẩn thận (đổi hành vi model).

**⬛ `--trust-remote-code`** — model đã hỗ trợ native, thường không cần. **⬛ `--disable-sliding-window`** — model không có sliding window. **⬛ `--enable-prompt-embeds` · `--use-fp64-gumbel` · `--logits-processors`** — không dùng.

---

## 8. LoadConfig — nạp weights (ảnh hưởng STARTUP)

**🎯 `--load-format FORMAT`** (auto/safetensors/...)
- *Kỹ thuật:* định dạng đọc weights.
- *Ứng dụng:* giữ auto (safetensors). Ảnh hưởng thời gian nạp → cửa startup 15 phút.

**🎯 `--safetensors-load-strategy` · `--safetensors-prefetch-num-threads` · `--safetensors-prefetch-block-size`**
- *Kỹ thuật:* chiến lược/độ song song prefetch khi đọc safetensors (đặc biệt trên network FS như CEPH).
- *Ứng dụng:* log của bạn báo "Auto-prefetch disabled (CEPH)" → set `--safetensors-load-strategy=prefetch` + tăng threads có thể **rút ngắn startup** (đang tải 4.24GB). Đáng thử để chắc lọt cửa 15 phút.

**⬛ `--download-dir` · `--revision` · `--ignore-patterns` · `--model-loader-extra-config` · `--pt-load-map-location`** — ít liên quan (offline, local path).

---

## 9. MultiModalConfig — bỏ vision tower

**🎯 `--language-model-only` / `--no-...`**
- *Kỹ thuật:* chỉ nạp/chạy phần language model, bỏ qua đường multimodal (vision).
- *Ứng dụng:* **win rẻ** — bỏ vision tower 0.33B (trả VRAM + bỏ 14s multimodal warmup log của bạn). Bài text thuần nên an toàn.

**🎯 `--skip-mm-profiling`**
- *Kỹ thuật:* bỏ bước profiling bộ nhớ cho multimodal lúc startup.
- *Ứng dụng:* rút ngắn startup nếu không dùng ảnh.

**⚪ `--limit-mm-per-prompt JSON`** (vd `image=0`)
- *Kỹ thuật:* giới hạn số item multimodal mỗi prompt.
- *Ứng dụng:* set `image=0` để chặn hẳn đường ảnh nếu `--language-model-only` chưa đủ.

**⬛ `--mm-encoder-* · --media-io-kwargs · --mm-processor-* · --video-*`** — chi tiết encoder ảnh/video, N/A cho text.

---

## 10. Speculative decoding (MTP) — ưu tiên thấp (decode đang ổn)

**⚪ `--spec-method {qwen3_5_mtp,mtp,eagle,ngram,...}`**
- *Kỹ thuật:* phương pháp sinh nháp nhiều token/bước rồi verify. **`qwen3_5_mtp`** dùng MTP head có sẵn của đúng model.
- *Ứng dụng:* tăng tốc DECODE (TPOT). Nhưng TPOT đã ~17ms < Floor 20ms → **đạt trần rồi, không thêm điểm** → ưu tiên thấp + rủi ro. Chỉ cân nhắc nếu TPOT thực tế >20ms.

**⚪ `--spec-tokens INT`** · **⚪ `--speculative-config JSON`** · **⬛ `--spec-model`**
- *Kỹ thuật:* số token nháp mỗi bước / config / model nháp riêng.
- *Ứng dụng:* nếu thử spec, tune số token nháp; `qwen3_5_mtp` không cần spec-model riêng.

---

## 11. ParallelConfig — single GPU nên hầu hết = 1

**🎯 `--tensor-parallel-size INT`** (default 1)
- *Kỹ thuật:* chia model qua nhiều GPU theo tensor.
- *Ứng dụng:* **=1** (1 MIG slice). Không tăng được vì chỉ có 1 GPU.

**⬛ `--pipeline-parallel-size` · `--data-parallel-size` · `--decode/prefill-context-parallel-size` · `--enable-expert-parallel` · toàn bộ cờ `--data-parallel-*`, `--eplb-*`, `--all2all-backend`, `--nnodes/--node-rank`, `--distributed-executor-backend`**
- *Kỹ thuật:* các dạng song song đa GPU/đa node, expert parallel (MoE).
- *Ứng dụng:* **N/A** — 1 GPU, model Dense, 1 node.

**⚪ `--max-parallel-loading-workers INT`** · **⚪ `--numa-bind`**
- *Kỹ thuật:* số worker nạp weights song song / ghim tiến trình vào NUMA node.
- *Ứng dụng:* có thể rút ngắn startup / giảm nhiễu CPU (3 core) một chút.

---

## 12. OffloadConfig — offload sang CPU (cẩn thận với 8GB RAM)

**⬛ `--cpu-offload-gb GiB` · `--cpu-offload-params` · `--kv-offloading-size/-backend` · `--offload-*`**
- *Kỹ thuật:* đẩy weights/KV sang RAM CPU để tiết kiệm VRAM (đổi lấy độ trễ PCIe).
- *Ứng dụng:* bài **không thiếu VRAM** (KV nhỏ) và **chỉ 8GB RAM + 3 core** → offload sẽ **làm chậm** (thêm trễ + tải CPU). **Tránh.**

---

## 13. Frontend (API server) — logging, usage, mạng

**🎯 `--enable-prompt-tokens-details` / `--no-...`**
- *Kỹ thuật:* cho vLLM nhét `prompt_tokens_details` (gồm `cached_tokens`) vào `usage` của response.
- *Ứng dụng:* **bật để có `cache_rd`/`hit%` per-request** (script 07). Không bật thì chỉ thấy `usage_prompt_tokens`, cột hit% trống.

**⚪ `--enable-force-include-usage`**
- *Kỹ thuật:* ép server luôn trả block `usage` (kể cả client không xin include_usage).
- *Ứng dụng:* thay thế `--use-server-token-count` phía AIPerf — bật cái này thì `usage`/`cache_rd` hiện kể cả bench không xin.

**🎯 `--disable-log-stats` · `--enable-log-requests`/`--no-...` · `--disable-uvicorn-access-log` · `--max-log-len`**
- *Kỹ thuật:* tắt log thống kê / log từng request / log access uvicorn / cắt độ dài log.
- *Ứng dụng:* **giảm tải 3-core CPU** → gián tiếp cứu TTFT. Tắt hết logging không cần cho lúc chấm. (`--enable-log-requests` mặc định off — cứ để off.)

**⚪ `--host` · `--port` · `--api-key` · `--uvicorn-log-level` · `--root-path`**
- *Kỹ thuật:* địa chỉ/cổng/khóa API/mức log/prefix path.
- *Ứng dụng:* hạ tầng — đặt cho khớp cách BTC gọi; không ảnh hưởng hiệu năng.

**⬛ `--chat-template*` · `--tool-call-parser` · `--reasoning-parser` · `--enable-auto-tool-choice` · `--middleware` · `--ssl-*` · `--h11-*`**
- *Kỹ thuật:* template chat, parse tool-calling/reasoning, middleware, SSL...
- *Ứng dụng:* trace là chat thường, không tool/reasoning → để mặc định. (Chat template đã auto-detect 'openai' theo log.)

---

## 14. ObservabilityConfig — đo đạc

**🎯 `--kv-cache-metrics` / `--kv-cache-metrics-sample FLOAT`**
- *Kỹ thuật:* bật thu metric KV cache chi tiết (lấy mẫu theo tỉ lệ).
- *Ứng dụng:* thêm dữ liệu chẩn đoán cache; hữu ích khi soi vì sao hit% thấp.

**⚪ `--collect-detailed-traces {all,model,worker}` · `--otlp-traces-endpoint` · `--enable-mfu-metrics` · `--show-hidden-metrics-for-version`**
- *Kỹ thuật:* trace chi tiết theo module / xuất OTLP / đo MFU (model FLOPs utilization) / lộ metric ẩn.
- *Ứng dụng:* nâng cao để phân tích sâu (MFU cho biết đang tận dụng bao nhiêu % compute). Bật khi cần đào; tắt lúc chấm cho nhẹ.

---

## 15. Các nhóm KHÔNG áp dụng (liệt kê cho đủ)

- **LoRA** (`--enable-lora`, `--max-loras`, `--max-lora-rank`, `--lora-*`, `--default-mm-loras`...): không dùng adapter → **N/A**.
- **StructuredOutputs / Reasoning** (`--structured-outputs-config`, `--reasoning-config`): trace không ép JSON/schema → **N/A**.
- **KV/EC transfer & disaggregated** (`--kv-transfer-config`, `--kv-events-config`, `--ec-transfer-config`, `--weight-transfer-config`): cho prefill/decode tách máy → **N/A** (1 GPU).
- **Diffusion** (`--diffusion-config`): model ngôn ngữ → **N/A**.
- **Misc** (`--enable-sleep-mode`, `--enable-cumem-allocator`, `--shutdown-timeout`, `--tokens-only`, `--fail-on-environ-validation`): tiện ích vận hành, không ảnh hưởng điểm.

---

## Nhắc lại ưu tiên (chi tiết ở `vllm-optimization-guide.md`)

1. Verify **prefix caching** ăn (`--enable-prefix-caching` + đo `hit%`) → thử **`--mamba-cache-mode all`**, **`--gdn-prefill-backend`**.
2. **FP8 weight** (`--quantization`) — đòn bẩy prefill lớn nhất.
3. Sweep **`--max-num-batched-tokens`** (+ chunked prefill) cho TTFT round-1.
4. **`--language-model-only`** (bỏ vision) + hygiene CPU (`--disable-log-stats`) + startup (`--safetensors-load-strategy=prefetch`).
5. `--performance-mode throughput` để dò nhanh. KV-quant / spec-decode chỉ khi đo có lợi thật.

Cách truyền cờ: bỏ vào `EXTRA_VLLM_ARGS` trong `serve/.env` (xem `.env.example`).
