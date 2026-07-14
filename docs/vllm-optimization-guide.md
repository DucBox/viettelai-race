# vLLM tuning + nguyên lý gốc — Track 3 (Qwen3.5-2B)

Kim chỉ nam tối ưu serving. Ràng buộc cứng: **chỉ được serve bằng vLLM**, không đổi
engine. Mọi thứ dưới đây đều xoay quanh các cờ của vLLM.

> Cập nhật sau trace vLLM 0.24.0 + kết quả v26-v31: model vẫn có rất nhiều việc
> prefill, nhưng **recipe thắng điểm hiện tại không còn là "tối ưu TTFT bằng mọi
> giá"**. Với hàm điểm này, v26/v31 thắng vì giữ **TPOT/TBT = 16ms** bằng
> `--max-num-seqs=10`, chấp nhận TTFT p50 ~9.3s. Phần dưới có vài đoạn lịch sử từ
> giai đoạn v0.22/v12-v23; khi xung đột, ưu tiên section v26/v31 ngay sau đây.

## -1. V26/V31: breakdown TTFT/TPOT -> tối ưu từng component

Mình không coi là "nắm hết vLLM tuyệt đối" theo nghĩa mọi nhánh engine trên mọi
model, nhưng với bài này thì đã trace đủ các đường nóng cần quyết định:

- V1 scheduler/admission: `v1/core/sched/scheduler.py`.
- KV/cache manager + watermark/prefix hit: `v1/core/kv_cache_manager.py`.
- CLI -> config: `engine/arg_utils.py`, `config/cache.py`, `config/scheduler.py`.
- Qwen3.5 hybrid guard: `model_executor/models/qwen3_5.py`.
- Mamba/GDN alignment: `config/vllm.py`.
- GDN prefill backend: `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`.

### -1.1 TTFT gồm những gì

TTFT client không phải một cục "prefill". Với AIPerf, nên tách như sau:

```
TTFT_client
= client_queue
+ http_ingress
+ request_parse/chat_template/tokenizer
+ engine_enqueue
+ server_queue/admission
+ prefix_cache_hash_lookup
+ prefill_scheduling_wait
+ prefill_compute_chunks
+ KV/GDN_state_write
+ first_decode_or_prefill_output_step
+ detokenize/SSE_flush_first_token
```

Trong đó `client_queue` của AIPerf là thời gian request nằm ở phía load generator
trước khi tới server. Phần **server queue** nằm bên trong TTFT, không hiện riêng
nếu chỉ nhìn CSV tổng.

| Component | vLLM path nóng | Cờ tác động | Kết luận cho v26/v31 |
|---|---|---|---|
| Client queue | ngoài vLLM | không tune bằng compose server | Chỉ dùng để đọc kết quả; đừng nhầm với queue nội bộ scheduler. |
| HTTP ingress + JSON parse | OpenAI API server | `--no-enable-log-requests`, `--disable-log-stats`, `--stream-interval` | v31 đã đo: logging hygiene gần như neutral, TTFT p50 9290->9284, p95 11670->11666. |
| Tokenizer/chat template | tokenizer pool + request preprocess | `--tokenizer-pool-size`, `--tokenizer-pool-type`, chat-template args | Với prompt dài, tokenize có thật nhưng nhỏ hơn prefill. Không nên tăng tokenizer pool nếu CPU contention/IPC tăng. |
| Engine enqueue | async engine/core | `--async-scheduling` | v17 giảm TBT 1ms nhưng xấu TTFT trên v0.22. Với v31/seqs=10 có thể A/B, chưa phải mặc định. |
| Server queue/admission | scheduler waiting/running loop | `--max-num-seqs`, `--max-num-batched-tokens`, `--watermark`, `--prefill-schedule-interval` | `max-num-seqs=10` là đòn điểm chính: queue/TTFT xấu hơn, nhưng TBT thắng lớn. Không nâng lên 20/128 cho score hiện tại. |
| Prefix lookup | KV cache manager + hybrid connector | `--enable-prefix-caching`, `--prefix-caching-hash-algo`, Mamba cache mode | Giữ `--enable-prefix-caching`. Không set `--mamba-cache-mode=all`: Qwen3.5 v0.24 hard-reject. |
| Prefill chunk admission | scheduler token budget + Mamba alignment | `--max-num-batched-tokens`, `--enable-chunked-prefill`, `--long-prefill-token-threshold`, partial-prefill flags | V26 dùng `2174` để vượt ngưỡng 2 block Mamba/FP8 KV (`2*1072=2144`) với headroom nhỏ. Đây là knob TTFT chính còn lại. |
| Prefill kernels | attention + GDN linear attention + GEMM | `--quantization=fp8`, `--kv-cache-dtype=fp8`, `--gdn-prefill-backend=flashinfer`, compile/cudagraph | Giữ FP8 weight, FP8 KV, FlashInfer GDN. v0.24 cải thiện ở kernel/runner hơn là scheduler defaults. |
| KV/GDN write | cache block allocator + recurrent state | `--kv-cache-dtype`, `--mamba-cache-dtype`, `--gpu-memory-utilization`, `--block-size` | Không memory-bound nặng. FP8 KV ở đây còn ảnh hưởng block size 1072, nên nếu bỏ FP8 KV phải retune batch token từ đầu. |
| First token output | sampler + output processor + detokenize | sampling params, structured output flags, logging | Không dùng structured output/speculative nếu không bắt buộc. |

### -1.2 TPOT/TBT gồm những gì

TPOT là vòng lặp decode sau token đầu, chịu tác động khác TTFT:

```
TPOT_client
= scheduler_tick_wait
+ decode_batch_build
+ graph_or_kernel_dispatch_overhead
+ per_token_attention_read_KV
+ per_token_GDN/Mamba_state_update
+ logits/sampler
+ output_processor
+ detokenize
+ network_stream_flush
```

Với v26/v31, TBT median 16ms nghĩa là decode đang ở vùng rất tốt so với floor 20ms
của scoring. Đây là lý do v26/v31 thắng v28 dù v28 pass 120/120 SLO TTFT: v28 TBT 35ms
bị phạt nặng hơn.

| Component | Cờ tác động | Quan sát từ submit | Tối ưu đúng cho v26/v31 |
|---|---|---|---|
| Scheduler tick/wait | `--max-num-seqs`, `--async-scheduling` | `seqs=10` -> TBT 15-16; `seqs=20` -> 21-22; default 128 -> 35 | Giữ `--max-num-seqs=10`. Đây là cờ quan trọng nhất cho TPOT/score. |
| Decode batch size | `--max-num-seqs`, active concurrent requests | Tăng concurrency làm TTFT đẹp nhưng TPOT xấu | Không tối ưu theo pass SLO; tối ưu theo ERS 50/50. |
| CUDAGraph/dispatch | cudagraph capture sizes, compile config, async scheduling | Với `seqs=10`, default capture size runtime thường đủ nhỏ | Chỉ set thủ công nếu log thấy graph miss; không thêm cờ compile bừa. |
| Attention KV read | `--kv-cache-dtype=fp8`, `--block-size`, prefix cache | FP8 KV giúp decode bandwidth nhưng ép block size lớn/align Mamba | Giữ FP8 KV trong v31; nếu A/B BF16 KV thì phải retune `max-num-batched-tokens`. |
| GDN/Mamba update | `--mamba-cache-dtype`, `--mamba-cache-mode` | `mamba-cache-dtype=bf16` từng giảm TBT 1ms nhưng xấu TTFT/net loss; `all` không hợp Qwen3.5 v0.24 | Không set `mamba-cache-mode=all`; không ưu tiên mamba dtype trừ khi sweep lại trên v0.24. |
| Sampler/output | sampling params, logprobs, guided decoding, logging | Nếu benchmark không cần logprobs/structured output thì nên tránh | Không bật guided/speculative/logprobs. Thêm no-log request là low-risk. |
| Detokenize/flush | streaming interval, tokenizer CPU | Ít hơn kernel/decode batch, nhưng nằm trong residual | Có thể thử `--stream-interval` nếu harness chấp nhận; đừng làm tăng first-token flush. |

### -1.3 Compose v26/v31 hiện tại: giữ gì, thử gì, tránh gì

V26 hiện tại:

```yaml
image: vllm/vllm-openai:v0.24.0
--max-model-len=48000
--gpu-memory-utilization=0.95
--tensor-parallel-size=1
--enable-prefix-caching
--language-model-only
--kv-cache-dtype=fp8
--calculate-kv-scales
--max-num-seqs=10
--quantization=fp8
--gdn-prefill-backend=flashinfer
--max-num-batched-tokens=2174
```

V31 = v26 hygiene-only:

```yaml
--no-enable-log-requests
--disable-log-stats
# bỏ --calculate-kv-scales
```

Kết quả submit: **v31 = 53.20**, v26 = 53.12. Chênh +0.08 nằm trong nhiễu, nhưng
nó xác nhận đúng hai điều: logging overhead không đáng kể ở regime `seqs=10`, và
`--calculate-kv-scales` thật sự không có tác dụng thực tế trên path hybrid GDN/Mamba
này.

**Giữ chắc:**

- `image: vllm/vllm-openai:v0.24.0`: v28/v29/v30 chứng minh 0.24 thắng 0.22 rõ rệt.
- `--max-num-seqs=10`: cờ quyết định TBT/score. v27 `seqs=20` và v28 default 128 đều thua dù TTFT đẹp hơn.
- `--max-num-batched-tokens=2174`: đủ vượt 2 block Mamba/FP8 KV (`2144`) với headroom nhỏ; đây là sweet spot v26/v31.
- `--gdn-prefill-backend=flashinfer`: backend GDN tốt nhất đã đo; v0.24 dùng FlashInfer mới hơn.
- `--quantization=fp8`: weight FP8 là đòn prefill/GEMM lớn.
- `--kv-cache-dtype=fp8`: không chỉ tiết kiệm KV, mà còn đang là một phần của regime block-size 1072; bỏ nó là đổi bài toán scheduler.
- `--enable-prefix-caching`: cần cho round sau; không phải cứu round 1 nhưng vẫn giữ.
- `--language-model-only`: tránh vision tower/runtime thừa.

**Đã đo và có thể giữ như v31:**

- Thêm `--no-enable-log-requests`: neutral/tốt nhẹ; không đổi scheduling/kernel.
- Thêm `--disable-log-stats`: neutral/tốt nhẹ khi giữ `seqs=10`; v27 regression là do `seqs=20`, không phải do cờ này.
- Bỏ `--calculate-kv-scales`: confirmed inert; v31 giữ TBT 16 và accuracy_drop 0.

**Có thể cải tiến/A-B sâu tiếp quanh v31:**

- `v32`: thêm `--default-chat-template-kwargs={"enable_thinking": false}`. Đây không phải scheduler knob; nó đi qua OpenAI chat serving -> `_effective_chat_template_kwargs` -> tokenizer/render. V28 có flag này nhưng bị nhiễu bởi default `seqs=128`, nên chưa cô lập được.
- `v33`: `--max-model-len=32768`. V28 chứng minh workload không reject ở 32k; nếu 48k là headroom thừa, 32k có thể giảm profile/cache pressure. Kỳ vọng nhỏ vì v31 không memory-bound.
- `v34`: `--async-scheduling`. Đây đổi class `Scheduler` -> `AsyncScheduler`, không chỉ đổi logging. Cửa thắng duy nhất là TBT 16->15; nếu TTFT tail/accuracy xấu thì bỏ.
- Micro-sweep `--max-num-batched-tokens` với `seqs=10`: `2174`, `2208`, `2304`, `3216`. Dự đoán `2174/2208` là vùng đáng thử; `2144` không nên ưu tiên vì chỉ cần decode token chen vào là budget còn dưới 2144 và chunk rơi về 1 block.

**Tránh cho v26/v31:**

- Không tăng `--max-num-seqs` lên 20/128 chỉ để đẹp TTFT. Bảng điểm đã chứng minh score giảm.
- Không set `--mamba-cache-mode=all`: Qwen3.5 v0.24 có guard không support; hơn nữa mode này có thể kéo attention block size theo alignment Mamba.
- Không bật speculative/MTP: v8 từng regression nặng; hybrid GDN/Mamba + scorer hiện tại không đáng rủi ro.
- Không bật partial prefill/concurrent partial prefill hoặc `--long-prefill-token-threshold=256`: các thử nghiệm v15/v19 crash/không hợp Mamba block alignment.
- Không đổi `--gdn-prefill-backend` sang triton/cutedsl: đã đo kém hơn hoặc có accuracy drop.

**Dead-end sâu sau khi trace code:**

- `--max-cudagraph-capture-size` / `--cudagraph-capture-sizes`: với `max-num-seqs=10`, vLLM tự set max graph size `min(max_num_seqs*2,512)=20`, đủ phủ decode batch <=10. Thêm thủ công khó giảm TBT hơn nữa, có thể chỉ tăng warmup/capture memory.
- `--stream-interval > 1`: output processor vẫn gửi token đầu, nhưng các token sau bị buffer tới interval. Nếu scorer đo streaming TBT theo client receive time, cờ này làm TBT nhìn xấu dù host overhead giảm.
- `--watermark`: chỉ áp vào waiting/preempted requests khi đã có scheduled reqs. v31 không có preemption/failed, nên watermark chủ yếu làm admission khó hơn -> TTFT xấu.
- `--prefill-schedule-interval`: code comment ghi cho data-parallel prefill balancing. Single GPU/TP=1 không phải chỗ thắng.
- `--scheduling-policy=priority`: nếu request không gửi priority khác 0 thì không đổi bản chất FCFS; chỉ thêm cơ chế preempt theo priority khi thiếu KV.
- Concurrent partial prefill: v0.24 vẫn có path, nhưng default threshold khi bật là `0.04*max_model_len=1920`, dưới 2 block 2144 của regime FP8-KV/Mamba. Muốn thử phải block-aware, nhưng rủi ro là thêm prefill chunks vào running set và làm TBT rời vùng 16ms.

### -1.4 Thứ tự tối ưu thực tế tiếp theo

Đã tạo v32/v33/v34 để submit theo thứ tự này, mỗi file đổi đúng một biến so với
v31:

1. **v32 chat-template A/B**: v31 + `enable_thinking=false`, vì đây là nhánh preprocess chưa được cô lập.
2. **v33 max-len A/B**: v31 + `max-model-len=32768`, vì v28 cho thấy có vẻ không reject workload.
3. **v34 async-only A/B**: v31 + `--async-scheduling`, chỉ giữ nếu TBT giảm mà accuracy không rơi.
4. **batch micro-sweep**: giữ nguyên v31, chỉ đổi `--max-num-batched-tokens` quanh `2174/2208/2304`.

Mục tiêu không phải làm TTFT đẹp nhất, mà là giữ điểm tổng: **TBT không được rời
vùng 15-16ms**; mọi cải tiến TTFT chỉ đáng nhận nếu không làm TBT tăng hoặc accuracy
drop xuất hiện.

Nếu được phép build custom vLLM image, frontier thật sự không còn là flag mà là
patch scheduler: giữ decode lane ở regime `seqs≈10`, nhưng cho prefill chen theo
quota block-aware khi `token_budget_after_decode >= 2144`. Nói cách khác, thay vì
một `token_budget` chung đang ép trade-off cứng TTFT-vs-TBT, tạo policy hai làn:
decode luôn được bảo vệ để TBT không vượt 16ms, còn prefill chỉ chạy ở chunk đúng
2 block Mamba/FP8-KV. Đây mới là hướng có khả năng giảm TTFT mà không trả giá TBT;
các flag public hiện tại không biểu diễn được policy này đủ mịn.

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
- `--kv-cache-dtype fp8` (= `fp8_e4m3`; enum v0.22.1: `auto, fp8, fp8_e4m3, fp8_e5m2, fp8_per_token_head, int8_per_token_head, nvfp4, ...`). `auto` = giữ nguyên dtype model.
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

## 3b. Bài học từ vLLM issues — prefix caching trên Qwen3.5 hybrid

Prefix caching cho hybrid (Mamba/GDN + attention) trong vLLM đang **phát triển dở**;
log của bạn xác nhận đang chạy **'align' mode (experimental)**. Ba issue đáng học:

| Issue | Nội dung | Bài học cho mình |
|---|---|---|
| [#26201](https://github.com/vllm-project/vllm/issues/26201) (tracking) | Có 2 mode: **`all`** (cơ bản, đã chạy được) và **`align`** (tối ưu Marconi-style, còn dở — có perf issue + nghẽn CPU-GPU sync). Hybrid mặc định = `align`. | **A/B `--mamba-cache-mode all` vs `align`** — mode `all` có thể **ổn định/nhanh hơn** cho mình. Đây là lever cụ thể. |
| [#43587](https://github.com/vllm-project/vllm/issues/43587) | Trên hybrid, request incremental (prefix tăng dần) trả **`num_cached_tokens=0` DÙ block hash khớp** → cache reuse **âm thầm hỏng**. (bug ở path multimodal, nhưng **đúng pattern multi-turn tăng dần** của trace mình) | **PHẢI verify hit% thật** bằng `cache_rd` (script 07) — **đừng tin** cache tự chạy đúng. Chính là lý do hạ tầng đo quan trọng. |
| [#40696](https://github.com/vllm-project/vllm/issues/40696) | block-size ép ~528–544 (align Mamba page); chỉ cache **block hoàn chỉnh** → prompt <544 hit ~0%, hit rate dao động theo ranh giới block. | Prompt mình 13k+ nên OK; nhưng **đuôi prefix lẻ block không cache**. Thử `--mamba-block-size` nếu muốn tinh chỉnh. |

**Kết:** prefix caching trên arch này **không đảm bảo tự chạy đúng** → phải verify bằng
số đo (`cache_rd`/`hit%`), và có sẵn 2 lever để thử: **`--mamba-cache-mode`** (all/align)
và **`--mamba-block-size`**. Nếu thấy `hit%≈0` dù prefix chung to → gần như chắc dính
biến thể của bug #43587 → đổi sang `all` mode / chỉnh block-size.

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
- [ ] Speculative decoding qua **MTP head**: **`--spec-method qwen3_5_mtp`** (có sẵn cho đúng model này ở v0.22.1) / `--spec-tokens`. CHỈ khi đo TPOT thực >20ms (đạt Floor rồi thì không thêm điểm). Rủi ro.

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

## 4c. FP8: chọn format, W8A8 vs W8A16, xử lý outlier

### Format: dùng **E4M3**
- **E4M3** (4 exp, 3 mantissa, range ±448) — **nhiều precision** → dùng cho **weight + activation**.
- **E5M2** (5 exp, 2 mantissa) — nhiều range, ít precision → gradient (training) / KV cache.
- **H200 (Hopper, tensor core gen 4)** và **L40S (Ada)** đều **native FP8** → GEMM FP8 chạy **~2× BF16**. Speedup FP8 **chỉ có thật khi** phần cứng có FP8 core **và** engine có FP8 kernel (cả hai đều có ✓). Thiếu 1 trong 2 → "FP8" chỉ là dequant → **chậm hơn** BF16.

### W8A8 (bắt buộc) vs W8A16 (bẫy)
| Scheme | Weight | Activation | Tăng tốc prefill? |
|---|---|---|---|
| **W8A16** (weight-only) | FP8 | BF16 | ❌ gần như không — matmul vẫn chạy BF16, chỉ tiết kiệm VRAM |
| **W8A8** (full FP8) | FP8 | FP8 | ✅ có — matmul chạy thẳng trên FP8 core, ~2× |

Prefill là **compute-bound** → muốn TTFT giảm thì **phép nhân ma trận phải chạy trên FP8 core** → **bắt buộc W8A8** (activation cũng FP8). Weight-only chỉ giúp bộ nhớ/decode, **TTFT gần như không đổi** — đúng thứ mình cần lại không được.

### Outlier — FP8 tự lo phần lớn
LLM có **outlier channel** (vài chiều activation biên độ cực lớn) — vấn đề kinh điển của INT8/INT4. Nhưng **FP8 là số dấu phẩy động** → có exponent → **dải động rộng hơn INT8 nhiều** → **nuốt outlier tốt**. Vì vậy FP8 thường **KHÔNG cần** SmoothQuant/AWQ (mấy kỹ thuật đó dành cho INT4/INT8).

Với FP8, "xử lý outlier" gói gọn trong 3 thứ nhẹ:
1. **Scale mịn**: per-channel cho weight + **dynamic per-token** cho activation → cô lập outlier vào scale riêng.
2. **Giữ tensor nhạy ở BF16** (ignore-list §4b): `lm_head`, norm, `A_log`/`dt_bias`/`conv1d`, `in_proj_a/b`.
3. **Không** SmoothQuant/AWQ (để dành cho INT4 nếu sau này cần).

### Chốt
**`FP8_DYNAMIC` (E4M3, per-channel weight + dynamic per-token activation, W8A8)** — data-free, tự chịu outlier, đúng recipe §4b. Đó là lý do FP8 quantize được **không cần calibrate**. Muốn vắt thêm accuracy (khi có harness): calibrate scale (per-tensor → per-head) bằng llm-compressor.

---

## 5. THAM CHIẾU CỜ vLLM ĐẦY ĐỦ (từ source `arg_utils.py`)

> ✅ **Đã verify với `vllm serve --help` trên image thật `v0.22.1`** — mọi cờ 🎯 dưới
> đây đều tồn tại ở version đó. Đánh dấu: 🎯 = đang để ý cho bài này.

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

### (7) CUDA graphs & compilation & preset cấp cao
`--enforce-eager` 🎯(TẮT nó để giữ CUDA graph) · `--cudagraph-capture-sizes` 🎯 · `--max-cudagraph-capture-size` 🎯 · `--compilation-config/-cc` 🎯 · **`--performance-mode {balanced,interactivity,throughput}`** 🎯(preset 1-cờ — thử `throughput`) · `--optimization-level` 🎯 · `--async-scheduling` 🎯

### (8) Attention & kernels
`--attention-backend` 🎯 · `--attention-config/-ac` · `--enable-flashinfer-autotune` 🎯 · `--linear-backend` 🎯(GDN linear) · `--ir-op-priority` · `--kernel-config`

### (9) Mamba / GDN (riêng arch này)
`--gdn-prefill-backend {flashinfer,triton,cutedsl}` 🎯 (default `triton` — thử flashinfer/cutedsl) · `--mamba-cache-mode {align,all,none}` 🎯 (default `align`; thử `all`; `none`=tắt) · `--mamba-backend` 🎯 · `--mamba-cache-dtype {auto,bfloat16,float16,float32}` 🎯 · `--mamba-ssm-cache-dtype` 🎯 · `--mamba-block-size` 🎯 · `--enable-mamba-cache-stochastic-rounding`

### (10) Speculative decoding (MTP)
`--speculative-config/-sc` 🎯 · `--spec-method` 🎯 · `--spec-model` · `--spec-tokens` 🎯

### (11) Multimodal (bỏ vision)
`--language-model-only` 🎯(bỏ xử lý multimodal) · `--limit-mm-per-prompt` 🎯 · `--skip-mm-profiling` 🎯 · `--mm-encoder-*` (nhiều cờ encoder) · `--media-io-kwargs` · `--mm-processor-kwargs`

### (12) Logging / API / observability
`--disable-log-stats` 🎯 · `--enable-log-requests`/`--no-enable-log-requests` 🎯(mặc định off — giữ off cho nhẹ CPU) · `--disable-uvicorn-access-log` 🎯 · **`--enable-force-include-usage`** 🎯(server tự nhét `usage` vào response → `cache_rd/hit%` hiện kể cả không cần `--use-server-token-count` bên AIPerf) · `--kv-cache-metrics` 🎯 / `--kv-cache-metrics-sample` · `--cudagraph-metrics` · `--otlp-traces-endpoint` · `--collect-detailed-traces` · `--enable-logging-iteration-details` · `--enable-mfu-metrics`

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
