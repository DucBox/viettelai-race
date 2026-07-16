# vLLM v0.24.0 — Breakdown TTFT & TPOT tới tận root cause + đòn bẩy tối ưu

> Nguồn: bản clone chính thức `vllm-project/vllm @ tag v0.24.0` (HEAD `ee0da84`), đặt tại
> `vendor/vllm-0.24.0/` (đã cho vào `.gitignore`). Mọi `file:line` dưới đây **trỏ đúng source thật**
> của v0.24.0, không phải suy đoán.
>
> **Đính chính doc cũ:** file `docs/vllm_v0.24.0_all_flags_deep_trace.md` map cờ→file. Tôi từng nghi
> nó bịa (vì có `oink_ops.py`, `turboquant_attn.py`, `gumbel_precision/...`), nhưng khi clone thật
> thì **các file đó CÓ TỒN TẠI** — v0.24.0 là bản đã tiến hoá rất xa (HEAD là PR #46888). Doc cũ dùng
> được cho tra cứu cờ; doc này là phần "hiểu bản chất" mà cờ→file không trả lời được.

---

## 0. Ý tưởng cốt lõi: đừng tối ưu "TTFT/TPOT" — tối ưu từng *interval* mà chính vLLM đo

vLLM **tự nó** đã chẻ vòng đời request thành các mốc thời gian (timestamp) trong
[`vllm/v1/metrics/stats.py`](../vendor/vllm-0.24.0/vllm/v1/metrics/stats.py). Đây là "chân lý nền" — mọi
breakdown phải neo vào đây, không được tự chế định nghĩa.

Các mốc (`RequestStateStats`, [stats.py:202-221](../vendor/vllm-0.24.0/vllm/v1/metrics/stats.py#L202-L221)):

| Mốc | Đồng hồ | Đặt ở đâu |
|---|---|---|
| `arrival_time` | **wall-clock** (`time.time()`) | frontend, [`async_llm.py:742`](../vendor/vllm-0.24.0/vllm/v1/engine/async_llm.py#L742) khi AsyncLLM nhận request |
| `queued_ts` | **monotonic** (engine core) | event `QUEUED`, phát tại [`scheduler.py:1986`](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L1986) `add_request` |
| `scheduled_ts` | monotonic | event `SCHEDULED`, phát tại [`scheduler.py:942-943`](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L942-L943) (lần đầu được admit) |
| `first_token_ts` | monotonic | = `engine_core_timestamp` của output sinh token đầu ([stats.py:397](../vendor/vllm-0.24.0/vllm/v1/metrics/stats.py#L397)) |
| `last_token_ts` | monotonic | = timestamp output token cuối ([stats.py:402](../vendor/vllm-0.24.0/vllm/v1/metrics/stats.py#L402)) |
| `iteration_timestamp` | wall-clock | `time.time()` lúc frontend xử lý output ([stats.py:329](../vendor/vllm-0.24.0/vllm/v1/metrics/stats.py#L329)) |

Công thức breakdown **của chính vLLM** ([`update_from_finished_request`, stats.py:437-459](../vendor/vllm-0.24.0/vllm/v1/metrics/stats.py#L437-L459)):

```
e2e_latency  = iteration_timestamp − arrival_time          # wall-clock
queued_time  = scheduled_ts   − queued_ts                  # (monotonic) = "Queue"
prefill_time = first_token_ts − scheduled_ts               # (monotonic) = "Prefill"
decode_time  = last_token_ts  − first_token_ts
TPOT         = decode_time / (num_generation_tokens − 1)   # stats.py:455-459
TTFT (đo)    = first_token_latency = iteration_timestamp − arrival_time   # stats.py:369
```

### 0.1 Suy ra định nghĩa "Residual" của bạn — chính xác về mặt code

`queued_time` và `prefill_time` đều đo bằng **đồng hồ engine-core (monotonic)**. Còn TTFT đo bằng
**wall-clock ở frontend**. Phần TTFT **không** nằm trong 2 interval engine-core kia chính là **Residual**:

```
TTFT = Residual_input  +  Queue (queued_time)  +  Prefill (prefill_time)  +  Residual_output
       └── arrival→QUEUED ──┘                                              └ first_token_ts→frontend ┘
```

- **Residual_input** = từ lúc `arrival_time` (frontend nhận) → lúc request được `add_request` vào waiting
  queue (phát event `QUEUED`). Gồm: **tokenize + validate + IPC frontend→core + chờ engine đọc input queue**.
- **Residual_output** = từ lúc engine có `first_token_ts` → lúc frontend `iteration_timestamp`. Gồm:
  **IPC core→frontend + detokenize token đầu + push AsyncStream + serialize HTTP**.

> ⚠️ Vì 2 đồng hồ khác nhau, bạn **không** trừ trực tiếp `queued_ts − arrival_time` được (vLLM cố ý chỉ
> report `queued_time`/`prefill_time` thuần monotonic). Muốn đo Residual thật phải chèn timestamp riêng
> ở frontend (xem §5 — cách instrument).

Tương tự, **TPOT không phải một khối** — mỗi decode step là một vòng `EngineCore.step()`, và mỗi step lại
gồm nhiều phase con (đọc tiếp §4).

---

## 0.2 ĐỐI SOÁT: cộng breakdown lại CÓ RA đúng TTFT & TPOT không? (bắt buộc đọc)

Breakdown chỉ có giá trị nếu **Σ các mảnh = tổng**. Dưới đây là chứng minh bằng chính source, không phải ước lượng.

### (a) TTFT — phép cộng là ĐÚNG TUYỆT ĐỐI (telescoping)

TTFT vLLM report = `first_token_latency = iteration_timestamp − arrival_time`
([stats.py:369](../vendor/vllm-0.24.0/vllm/v1/metrics/stats.py#L369), qua `_time_since` [:349-351](../vendor/vllm-0.24.0/vllm/v1/metrics/stats.py#L349-L351)).

Chèn 3 mốc trung gian `queued_ts → scheduled_ts → first_token_ts` vào giữa `arrival_time` và
`iteration_timestamp`, rồi cộng chuỗi hiệu — các số hạng **triệt tiêu telescoping**:

```
TTFT = iteration_timestamp − arrival_time
     = (queued_ts     − arrival_time )   ← Residual_input   ┐
     + (scheduled_ts  − queued_ts    )   ← Queue  = queued_time  (stats.py:440)
     + (first_token_ts− scheduled_ts )   ← Prefill= prefill_time (stats.py:444)
     + (iteration_ts  − first_token_ts)  ← Residual_output  ┘
```

Cộng vế phải: mọi `queued_ts`, `scheduled_ts`, `first_token_ts` xuất hiện đúng 1 lần `+` và 1 lần `−` ⇒ khử hết,
còn lại đúng `iteration_timestamp − arrival_time = TTFT`. **⇒ Σ 4 mảnh = TTFT, sai số = 0 (đồng nhất thức).**

**Bẫy đồng hồ (phải biết):** `arrival_time`/`iteration_timestamp` là **wall-clock** (`time.time()`), còn
`queued_ts`/`scheduled_ts`/`first_token_ts` là **monotonic** engine-core (xem chú thích [stats.py:207-214](../vendor/vllm-0.24.0/vllm/v1/metrics/stats.py#L207-L214)).
Gọi `Δ` = độ lệch 2 đồng hồ. Khi đó:
- `Queue` và `Prefill` **sạch** (đều monotonic, không dính `Δ`).
- `Residual_input` chứa `+Δ`, `Residual_output` chứa `−Δ` ⇒ **tách riêng in/out thì lệch `Δ`**, nhưng
  **tổng thì `Δ` triệt tiêu**. Nên đại lượng đo được chắc chắn đúng là:

```
Residual (gộp) = TTFT − Queue − Prefill = Residual_input + Residual_output     ← luôn sạch, = 0 sai số
```

Muốn tách riêng `Residual_input` vs `Residual_output` → phải tự chèn timestamp cùng-đồng-hồ ở frontend (§7.2).

### (b) TPOT — cộng theo STEP là đúng; trong 1 step là CRITICAL-PATH (không cộng ngây thơ)

```
decode_time = last_token_ts − first_token_ts            (stats.py:448)
TPOT        = decode_time / (num_generation_tokens − 1) (stats.py:455-459)
```

- **Theo trục step (ĐÚNG):** `decode_time = Σ (thời lượng mỗi decode step mà request tham gia)`. Không spec-decode,
  mỗi step sinh 1 token ⇒ số step ≈ `gen − 1` ⇒ **TPOT = thời lượng step trung bình** trong pha decode. Cộng số
  step ra đúng `decode_time` (telescoping trên `last_token_ts − first_token_ts`).
- **Bên trong 1 step (T1..T7): là ĐƯỜNG TỚI HẠN, KHÔNG phải tổng cộng.** Với `--async-scheduling`
  (`step_with_batch_queue`, [core.py:519](../vendor/vllm-0.24.0/vllm/v1/engine/core.py#L519), `max_concurrent_batches=2`
  [vllm.py:499-504](../vendor/vllm-0.24.0/vllm/config/vllm.py#L499-L504)) thì sched-CPU (T1) + input-prep (T2) của
  step kế **chồng lấn** GPU exec step trước ⇒ `thời lượng step ≈ max(GPU_exec, CPU_sched)`, **không** phải `T1+T2+...`.
  Vì vậy khi đối soát TPOT: cộng theo **step** (đúng bằng số), còn T1..T7 chỉ để biết **phase nào là critical path**
  cần tấn công — muốn định lượng phải chạy profiler (§7.3), không cộng tay.

### (c) Ví dụ số — kiểm tra "cộng vào ~ tổng"

TTFT (1 request, prompt vừa, tải nhẹ):

| Mảnh | Giá trị | Nguồn |
|---|--:|---|
| Residual_input (tokenize+IPC) | 3.0 ms | tự đo (§7.2) |
| Queue (`queued_time`) | 12.0 ms | stats.py:440 |
| Prefill (`prefill_time`, 2 chunk) | 40.0 ms | stats.py:444 |
| Residual_output (IPC+detokenize) | 5.0 ms | tự đo |
| **Σ = TTFT** | **60.0 ms** | = `iteration_ts − arrival_time` ✓ |

Đối soát Residual gộp: `60 − 12 − 40 = 8 ms = 3 + 5` ✓ (khớp, `Δ` đã khử).

TPOT: `gen = 101`, `decode_time = 2000 ms` ⇒ `TPOT = 2000 / (101−1) = 20.0 ms/token`.
100 step × 20 ms = 2000 ms = `decode_time` ✓. Trong mỗi step 20 ms (async): `≈ max(forward 18ms, sched 6ms) + sample 2ms`
≈ 20 ms (critical-path, **không** phải 18+6+2).

> **Chốt:** TTFT cộng 4 mảnh = đúng tổng (đồng nhất thức). TPOT cộng theo step = đúng tổng; phase trong step là
> critical-path (đo bằng profiler). Đây là cách "make sure cộng breakdown ≈ TTFT/TPOT".

---

## 0.3 Workload-specific action matrix — 20 users × 6 turns, không dừng ở public flags

Trace thật đã xác nhận:

| Thuộc tính | Giá trị |
|---|---|
| Request | 120 = **20 session × 6 round** |
| Burst | mỗi round 20 request trong **475ms**, IAT nội bộ **25ms** |
| Round start | 0, 5000, 10000, 15000, 20000, 25000ms |
| Think time giữa burst | 4525ms |
| Message count | 2, 4, 6, 8, 10, 12 |
| Token thật theo round | 12,949 → 15,835 → 18,720 → 21,603 → 24,488 → 27,373 |
| Max prompt+output | ~27,600 token, nên `max_model_len=32768` vẫn đủ |
| Shared prefix | system prompt chung `1/1`, prefix continuity `100/100` cặp session |

Điều này đổi cách tối ưu:

- Round 1 có **20 prompt ~13k token cùng chung system prefix** lao vào gần đồng thời. Prefix cache mặc định chỉ giúp sau khi có block đã cache; nếu 20 request cùng hỏi cache trước khi request đầu hoàn tất, phần system có thể bị tính lặp. Đây là vùng **code-level** đáng tiền nhất.
- Round 2-6 đến mỗi 5s, nhưng v31 TTFT p50 ~9.3s nghĩa là burst sau có thể đến khi burst trước chưa ra token đầu. Prefix/cache lợi ích bị giới hạn bởi overlap thời gian.
- TPOT v31 = 16ms, dưới floor 20ms. Có **4ms slack** để mua lại TTFT/Queue, miễn không vượt 20ms hoặc accuracy drop.

### 0.3.1 Residual_in: HTTP/chat-template/tokenize/ZMQ

| Leaf | Tình trạng trên trace | Flag-level | Client / workload trick | Code-mod thật sự |
|---|---|---|---|---|
| A1 chat-template render | 120 request đều là chat; mỗi request render lại toàn bộ lịch sử 13k→27k token | `--default-chat-template-kwargs={"enable_thinking": false}` nếu template Qwen dùng biến này; v32 đã cô lập | Nếu kiểm soát client, render chat template offline 1 lần/request, gửi sang `/v1/completions` thay vì `/chat/completions` | Thêm cache render theo hash `messages` trong OpenAI frontend; key là JSON messages + template kwargs |
| A1 tokenize | HF tokenizer CPU, cost tăng theo prompt length; round 6 ~27k token | `--renderer-num-workers` có thể giúp nếu frontend CPU-bound, nhưng phải đo vì thêm worker cũng thêm IPC/coordination | `/v1/completions` chấp nhận `prompt: list[int]`; gửi token ids để bỏ tokenizer. Không set `--skip-tokenizer-init` nếu vẫn cần detokenize text output | Patch `/v1/chat/completions` nhận thêm `prompt_token_ids` hoặc internal header để bypass tokenizer sau khi client/proxy đã tokenize |
| A2 validate | Nhỏ so với tokenize/prefill | Không đáng | Không đáng | Không đáng |
| A3 ZMQ/msgpack frontend→core | Text/token list copy qua process boundary | Không có cờ ngon | Token ids làm payload số nguyên lớn; text 14MB trace cũng không phải bottleneck chính | Muốn triệt để: single-process engine hoặc shared-memory request payload, nhưng rủi ro lớn |
| A4 engine input HOL | Engine busy-loop đọc input queue giữa các step; step prefill dài làm request mới chờ | Giảm step quá tay sẽ hại prefill throughput | Điều phối client không bắn 20 request vào cùng 475ms nếu luật cho phép; thường không được | Patch scheduler/engine để drain input queue cả trong lúc GPU step đang chạy hoặc tách input receiver thread đẩy thẳng vào waiting queue |

**Kết luận Residual_in:** nếu benchmark bắt buộc OpenAI chat API, cờ chỉ còn v32 và logging hygiene. Đòn thật là **pre-tokenized completions** hoặc patch chat endpoint nhận token ids/cache render. Đây không làm GPU nhanh hơn, nhưng cắt phần CPU lặp trên 120 prompt dài.

### 0.3.2 Queue: waiting → scheduled

| Leaf | Tình trạng trên trace | Flag-level không-hiển-nhiên | Trick / code-level |
|---|---|---|---|
| Q1 hết seat `max_num_seqs` | v31 cố tình để `seqs=10`: TTFT xấu, TBT tốt | Test **v35 seqs=14** và **v36 seqs=16**. Đây là dùng TPOT floor slack, không phải tăng bừa. v27 seqs=20 đã quá tay: TBT 22 | Patch tách `max_decode_seqs` và `max_prefill_seqs`: decode giữ ~10 để TBT 16-20, nhưng cho prefill-only chunks chen thêm |
| Q2 token budget | `2174` được chọn vì `2*1072=2144` + 30 headroom | `2208/2304` là sweep hợp lý hơn `2144`; `2144` dễ rơi về 1 block khi decode chen vào | Patch block-aware budget: chỉ admit prefill nếu `budget_after_decode >= 2144`; nếu không thì skip prefill để bảo vệ decode |
| Q3 KV block | v31 không failed/preempt, KV không phải bottleneck chính | `max-model-len=32768` (v33) có thể giảm reserved/profile pressure; max prompt+output ~27.6k nên đủ | Nếu build custom: profile KV với max_len thật của trace, không dùng headroom 48k |
| Q5 HOL trong waiting | 20 request trong burst có prefix chung; FCFS làm request sau chờ | `priority` chỉ có ích nếu request gửi priority, trace không có | Group-aware scheduling: nhận diện 20 request cùng round/prefix, schedule common prefix trước, rồi fan-out |

**Kết luận Queue:** public flags chỉ biểu diễn trade-off thô seat/budget. Đòn sâu là **hai-làn decode/prefill**: decode lane được bảo vệ để TBT không vượt 20ms; prefill lane chỉ chạy khi còn đúng 2-block budget.

### 0.3.3 Prefill: scheduled → first token

| Leaf | Tình trạng trên trace | Flag-level | Trick / code-level |
|---|---|---|---|
| P1 số chunk | FP8-KV/Mamba tạo block ~1072; 2-block chunk là sweet spot đã đo | Giữ quanh `2174-2304`; không dùng threshold 256; không bật partial prefill default vì threshold tự động `0.04*48000=1920 < 2144` | Patch scheduler để threshold/chunk luôn là bội số block, ưu tiên 2 block; không để chunk nhỏ lẻ |
| P2 decode tranh budget | Running decode được schedule trước waiting prefill | `seqs=14/16` dùng slack nhưng phải giữ TBT<=20 | Two-lane scheduler: tính decode trước, sau đó nếu budget còn >=2144 mới chạy prefill; không để prefill kéo decode step quá dài |
| P3 prefix cache hit round 2-6 | Trace có prefix continuity, nhưng overlap 5s có thể khiến cache chưa kịp sẵn | `prefix-caching-hash-algo=xxhash` có thể giảm hash CPU nhẹ; không phải lever lớn | **Prefix priming**: trước benchmark, prefill system prompt/common prefix để round1 không tính 20 lần. Nếu luật không cho extra request, patch startup prewarm từ file token ids trước khi health ready |
| P3 in-flight duplicate prefix | 20 request round1 cùng system prompt; cache chưa có khi cùng burst | Không có public flag đủ mạnh | **In-flight prefix dedup/promise**: request B thấy prefix đang được request A tính thì chờ block promise thay vì tự tính lại. Khi A commit blocks, B ref-count/share blocks rồi chỉ tính suffix user |
| P4/P5 GPU prefill compute | Round1 cold 258,984 token không né được nếu không priming/dedup | GDN FlashInfer đã là best measured; FP8 obvious đã bật | Kernel-level frontier: custom GDN/attention kernel hoặc true common-prefix prefill fanout. Public flags gần hết dư địa |

**Kết luận Prefill:** đòn lớn nhất không phải thêm flag, mà là **không tính lặp common system prefix trong burst 1**. Với trace này system prompt là phần lớn của 12.9k token round1; nếu 20 request cùng tính lại, Queue và Prefill cùng nổ.

### 0.3.4 Residual_out: first token → client sees token

| Leaf | Tình trạng | Action |
|---|---|---|
| B1 core→frontend ZMQ | Nhỏ, token đầu payload bé | Không tối ưu trước |
| B2 detokenize token đầu | Nhỏ nhưng nằm trong TTFT client | Chỉ bỏ được nếu scorer/client nhận token ids; nếu cần text thì giữ tokenizer |
| B3 streaming HTTP | `stream_interval>1` gửi token đầu vẫn ngay, nhưng token sau bị gộp | Không dùng cho scorer đo TPOT streaming; có thể làm TBT client nhìn xấu theo cụm |

**Kết luận Residual_out:** không phải mặt trận chính. Đừng đổi `stream_interval` để “tăng throughput” nếu metric là inter-token client.

### 0.3.5 TPOT/decode

| Leaf | Tình trạng v31 | Action |
|---|---|---|
| T1 scheduler CPU | v17 async từng giảm 1ms nhưng net loss ở regime cũ | v34 cô lập async trên v31; chỉ giữ nếu TBT 16→15 mà TTFT/accuracy không xấu |
| T3/T4 decode memory bandwidth | v31 TBT 16 đã dưới floor 20 | Không cần giảm TBT nữa; dùng slack để giảm Queue/TTFT |
| T5 sampling/logprobs | Trace `temperature=0`, không cần logprobs | Đảm bảo không bật logprobs/structured output/reasoning parser |
| T8 output per token | `stream_interval` có thể giảm CPU nhưng hại TPOT client | Không dùng |

---

## 1. Sơ đồ đường đi 1 request (v0.24.0, chế độ AsyncLLM multiprocess mặc định)

```
[HTTP] → AsyncLLM.add_request (async_llm.py:280, arrival_time=time.time())
      → InputProcessor.process_inputs  (input_processor.py:242)   ← TOKENIZE  ┐
      → EngineCoreClient.add_request  (msgpack serialize + ZMQ)               │ Residual_input
   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ (ranh giới tiến trình) ─ ─ ─ ─ ─ ─ ─ ─    │
      → EngineCore.run_busy_loop (core.py:1259)                              │
          _process_input_queue (core.py:1269) → add_request → QUEUED  ───────┘ = queued_ts
          ┌───────────────── vòng lặp step (core.py:479 / :519) ────────────┐
          │ scheduler.schedule()  (scheduler.py:388)  ← admit? → SCHEDULED  │ ← scheduled_ts
          │ model_executor.execute_model()  (gpu_model_runner.py:4056)      │   } Queue = chờ giữa
          │   preprocess(:4094) → forward(:4326) → postprocess(:4340)       │   } QUEUED và SCHEDULED
          │ sample_tokens (:4435): sample(:4470)/draft(:4495)/bookkeep      │
          │ scheduler.update_from_output → EngineCoreOutput (+timestamp)    │ ← first_token_ts (khi
          └─────────────────────────────────────────────────────────────────┘   token đầu ra)
      → output_queue → ZMQ → EngineCoreClient.get_output_async               ┐
      → output_handler (async_llm.py:656)                                    │ Residual_output
      → OutputProcessor.process_outputs (output_processor.py:675) ← DETOKENIZE│
      → AsyncStream → generate() yield → HTTP                                ┘ = iteration_timestamp
```

Điểm mấu chốt về kiến trúc:
- **2 tiến trình**: frontend (tokenize/detokenize/HTTP) và EngineCore (schedule + GPU). Nối bằng ZMQ +
  msgpack. → Residual = chi phí 2 tiến trình + 2 lần vượt ranh giới.
- **Busy loop single-thread** ở engine ([core.py:1259-1265](../vendor/vllm-0.24.0/vllm/v1/engine/core.py#L1259-L1265)):
  đọc input queue **rồi** mới step. Nếu một step đang chạy dài, request mới **kẹt** đến hết step (HOL blocking)
  → đây là một nguồn Residual_input/Queue ẩn dưới tải.

---

## 2. BREAKDOWN — Residual (phần "vô hình" của TTFT, không thuộc engine-core interval)

### 2.A — Residual_input (arrival → QUEUED)

| Leaf | Root cause | Code (v0.24.0) | Đòn bẩy (cờ / code-mod) |
|---|---|---|---|
| **A1. Tokenize prompt** | HF tokenizer chạy CPU, đơn luồng; cost ∝ độ dài prompt. Với chat: còn `apply_chat_template`. | [`input_processor.py:242`](../vendor/vllm-0.24.0/vllm/v1/engine/input_processor.py#L242) `process_inputs`; `renderer.tokenizer` | `--tokenizer-mode mistral` (tokenizer nhanh hơn nếu hợp lệ); **gửi thẳng `prompt_token_ids`** để bỏ hẳn bước encode; `--skip-tokenizer-init` khi client tự tokenize; tăng song song ở frontend qua `--renderer-num-workers`. |
| **A2. Validate input** | check `max_token_id` vs vocab, sampling params, max_model_len | [`input_processor.py:469-483`](../vendor/vllm-0.24.0/vllm/v1/engine/input_processor.py#L469-L483) | Không đáng kể; không đụng. |
| **A3. IPC serialize + gửi** | msgpack hoá request + ZMQ frontend→core | `EngineCoreClient.add_request` (`core_client.py`) | Text nhỏ → rẻ. Chỉ đáng lo với multimodal (dùng `--mm-tensor-ipc`). |
| **A4. Chờ engine đọc input queue** | busy loop đang bận 1 step dài ⇒ request kẹt tới hết step | [`core.py:1269-1298`](../vendor/vllm-0.24.0/vllm/v1/engine/core.py#L1269-L1298) | Giảm độ dài mỗi step: `--max-num-batched-tokens` nhỏ hơn / `--long-prefill-token-threshold` để step ngắn, phản hồi input nhanh hơn (đánh đổi throughput). |

### 2.B — Residual_output (first_token_ts → frontend)

| Leaf | Root cause | Code | Đòn bẩy |
|---|---|---|---|
| **B1. IPC core→frontend** | ZMQ + msgpack output | [`async_llm.py:660`](../vendor/vllm-0.24.0/vllm/v1/engine/async_llm.py#L660) `get_output_async` | Ít đụng. `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` chia nhỏ để không block event loop ([async_llm.py:654](../vendor/vllm-0.24.0/vllm/v1/engine/async_llm.py#L654)). |
| **B2. Detokenize token đầu** | `IncrementalDetokenizer.decode` chạy CPU | [`output_processor.py:675`](../vendor/vllm-0.24.0/vllm/v1/engine/output_processor.py#L675); `detokenizer.py` | `detokenize=False` nếu client chỉ cần token ids; token đầu **không** bị `stream_interval` gộp (stream_interval chỉ ảnh hưởng token sau — [output_processor.py:287-308](../vendor/vllm-0.24.0/vllm/v1/engine/output_processor.py#L287-L308)). |
| **B3. Event-loop asyncio + HTTP** | output_handler là task asyncio; nghẽn nếu nhiều request đồng thời chia CPU | [`async_llm.py:656-702`](../vendor/vllm-0.24.0/vllm/v1/engine/async_llm.py#L656-L702) | `--aggregate-engine-logging`/tắt log stats để bớt CPU frontend; tách nhiều API worker. |

**Khi nào Residual đáng kể?** Prompt ngắn + tải nhẹ → Residual (nhất là tokenize A1 + detokenize B2 +
2 lần IPC) có thể chiếm phần lớn TTFT vì Queue≈0, Prefill nhỏ. Prompt dài / tải nặng → Residual bị Queue+Prefill
nhấn chìm. **Phải instrument mới biết** (xem §5).

---

## 3. BREAKDOWN — Queue (`queued_time = scheduled_ts − queued_ts`)

Đây là thời gian request **nằm trong `waiting` queue** vì scheduler chưa admit được. Vòng admit các request
WAITING nằm ở [`scheduler.py:629-958`](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L629-L958). Mỗi
"lý do không admit được" là một root cause riêng:

| Leaf | Điều kiện chặn (code) | Root cause | Đòn bẩy |
|---|---|---|---|
| **Q1. Hết slot sequence** | `len(running) == max_num_running_reqs → break` [scheduler.py:630-631](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L630-L631) | Batch decode đã đầy `max_num_seqs` | `--max-num-seqs` **tăng** để admit nhiều hơn (đổi lấy TPOT cao hơn); hoặc **giảm** nếu đang bị preempt. Đây là cờ đánh đổi Queue↔TPOT trực tiếp. |
| **Q2. Hết token budget** | `while token_budget > 0` + `token_budget −= num_new_tokens` [scheduler.py:629,958](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L958); budget = `max_num_batched_tokens` [:408](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L408) | **Running decodes được xếp TRƯỚC** (vòng running [:431-580](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L431-L580)) ăn hết budget ⇒ prefill mới đói | `--max-num-batched-tokens` **tăng** để còn budget cho prefill mới ⇒ giảm Queue (đổi lấy step dài hơn ⇒ TPOT). |
| **Q3. Hết KV block** | `allocate_slots(...) is None → break` [scheduler.py:874-895](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L874-L895) | Không đủ block KV trống (các running req đang giữ) | `--gpu-memory-utilization` ↑; `--kv-cache-dtype fp8` (gấp đôi block); `--max-model-len` ↓; `--num-gpu-blocks-override`; `--block-size`; `--enable-prefix-caching` (tái dùng). |
| **Q4. Prefill không chunk được** | `not enable_chunked_prefill and num_new_tokens > budget → break` [scheduler.py:803-809](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L803-L809) | Chunked prefill tắt ⇒ prompt dài hơn budget bị chặn hoàn toàn | Giữ `--enable-chunked-prefill` (mặc định **True**, [scheduler.py:84](../vendor/vllm-0.24.0/vllm/config/scheduler.py#L84)). |
| **Q5. Thứ tự FCFS / priority** | `_select_waiting_queue_for_scheduling` + `peek_request` [scheduler.py:633-636](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L633-L636) | Request bị request trước nó chặn đầu hàng (HOL) | `--scheduling-policy priority` + set `priority` per-request để request quan trọng vượt hàng. |
| **Q6. Throttle prefill (DP)** | `defer_prefills` [scheduler.py:426-428](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L426-L428); `_should_throttle_prefills` [core.py:1916](../vendor/vllm-0.24.0/vllm/v1/engine/core.py#L1916) | Chỉ khi Data-Parallel bật; cân bằng prefill giữa các DP rank | `--prefill-schedule-interval` (mặc định 1 = không hoãn, [scheduler.py:153](../vendor/vllm-0.24.0/vllm/config/scheduler.py#L153)). Không DP thì bỏ qua. |
| **Q7. max_loras** | skip nếu vượt `max_loras` [scheduler.py:654-665](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L654-L665) | Chỉ khi dùng LoRA | `--max-loras` ↑. Không LoRA thì bỏ qua. |

> **Kết luận Queue:** với bài toán 1 model, text-only, không DP/LoRA — Queue bị chi phối bởi **Q1
> (`max_num_seqs`), Q2 (`max_num_batched_tokens`), Q3 (KV blocks)**. 3 cờ này là toàn bộ mặt trận Queue.

---

## 4. BREAKDOWN — Prefill (`prefill_time = first_token_ts − scheduled_ts`)

Interval này = từ lần **đầu tiên** được SCHEDULED → lúc sinh token đầu. Với **chunked prefill**, một prompt
dài bị chẻ nhiều chunk qua **nhiều step**, và token đầu chỉ ra sau chunk **cuối** ⇒ `prefill_time ≈ (số chunk)
× (thời gian mỗi step)`, mỗi step lại chia sẻ với decode của request khác. Chia 2 tầng:

### 4.1 Tầng lập lịch — bao nhiêu chunk / bao nhiêu step?

| Leaf | Root cause | Code | Đòn bẩy |
|---|---|---|---|
| **P1. Kích thước chunk** | chunk = `min(prompt_len, long_prefill_token_threshold, budget_còn_lại)`. Chunk nhỏ ⇒ nhiều step ⇒ prefill_time dài | [scheduler.py:796-811](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L796-L811). **Default `long_prefill_token_threshold = 0`** (= KHÔNG cap, chunk chỉ bị giới hạn bởi budget) — field default 0 ([config/scheduler.py:80](../vendor/vllm-0.24.0/vllm/config/scheduler.py#L80)). Công thức `int(max_model_len × 0.04)` ([config/scheduler.py:258-259](../vendor/vllm-0.24.0/vllm/config/scheduler.py#L258-L259)) **CHỈ áp dụng khi `--max-num-partial-prefills > 1`** (guard [:256](../vendor/vllm-0.24.0/vllm/config/scheduler.py#L256)); config 1-model mặc định (`max_num_partial_prefills=1`) ⇒ threshold giữ **0**. | `--long-prefill-token-threshold` **tăng** (chunk to hơn, ít step hơn) khi ưu tiên TTFT của prompt dài; `--max-num-batched-tokens` ↑ để chunk to. |
| **P2. Tranh chấp với decode** | budget bị running decode ăn trước ⇒ chunk prefill nhỏ đi ⇒ nhiều step hơn | vòng running trước waiting [scheduler.py:431](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L431) | Cùng cờ Q2. Đánh đổi: prefill nhanh (ít contention) ⇔ TPOT của batch hiện tại. |
| **P3. Prefix-cache hit** | token đã cache ⇒ `num_computed_tokens` bỏ qua ⇒ ít token phải tính ⇒ prefill ngắn | `get_computed_blocks` [scheduler.py:710](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L710); `PrefillStats.set` [stats.py:260-273](../vendor/vllm-0.24.0/vllm/v1/metrics/stats.py#L260-L273) | `--enable-prefix-caching` (+`--prefix-caching-hash-algo`). Vàng nếu prompt có prefix chung (system prompt). |

> **⚠️ Cạm bẫy V0→V1: `--max-num-partial-prefills` & `--max-long-partial-prefills` là NO-OP trong v0.24.0.**
> Grep toàn repo: 2 cờ này **không được đọc ở bất kỳ đâu lúc runtime** — chỉ `long_prefill_token_threshold`
> được scheduler dùng ([scheduler.py:468](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L468),[:797](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L797)).
> Chúng là khái niệm của scheduler **V0**; V1 chuyển sang thuần token-budget (comment [scheduler.py:390-399](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L390-L399):
> *"There's no 'decoding phase' nor 'prefill phase'"*).
>
> **Hệ quả (trả lời trực tiếp câu "budget 2500, A prefill nốt 500, B có vào 2000 dư không?"):** CÓ.
> Số prompt cùng prefill trong 1 step **KHÔNG** bị `max_num_partial_prefills=1` giới hạn — điều kiện admit ở
> vòng waiting ([scheduler.py:629-631](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L629-L631)) chỉ là
> `token_budget > 0` **và** `len(running) < max_num_seqs` **và** còn KV block. Tại iteration N: vòng running cấp
> nốt 500 cho A (budget 2500→2000), rồi vòng waiting admit B với `min(prompt_B, 2000)` token — **A và B cùng
> prefill trong một batch**. Chặn "bao nhiêu prompt cùng prefill" thực chất = `max_num_batched_tokens` + `max_num_seqs` + KV.

### 4.2 Tầng compute — thời gian 1 step prefill (GPU)

Phase con lộ qua marker profiler của chính vLLM trong `execute_model`
([gpu_model_runner.py:4056+](../vendor/vllm-0.24.0/vllm/v1/worker/gpu_model_runner.py#L4056)):

| Leaf | Phase / code | Root cause | Đòn bẩy |
|---|---|---|---|
| **P4. Preprocess (CPU + H2D)** | `"gpu_model_runner: preprocess"` [:4094](../vendor/vllm-0.24.0/vllm/v1/worker/gpu_model_runner.py#L4094): `_update_states` + `_prepare_inputs` [:4140](../vendor/vllm-0.24.0/vllm/v1/worker/gpu_model_runner.py#L4140) (copy input_ids/positions/slot_mapping/block_table lên GPU, dựng attention metadata) | Overhead CPU + copy H2D | `--async-scheduling` (overlap với GPU, §4.3); prefill shape động ⇒ **không** được cudagraph. |
| **P5. Forward (compute-bound)** | `"gpu_model_runner: forward"` [:4326](../vendor/vllm-0.24.0/vllm/v1/worker/gpu_model_runner.py#L4326) | Prefill là **compute-bound**: FLOPs ∝ prompt_len × params (GEMM lớn) | `--quantization fp8` (GEMM nhanh); `--compilation-config`/piecewise cudagraph; `-tp` chia layer. |
| **P6. Attention prefill** | trong forward; backend chọn qua config | Full causal attention trên toàn prompt | `--attention-backend FLASH_ATTN`/`FLASHINFER`; `--disable-cascade-attn` (hoặc để cascade tận dụng prefix chung). |
| **P7. Postprocess + sample token đầu** | `"postprocess"` [:4340](../vendor/vllm-0.24.0/vllm/v1/worker/gpu_model_runner.py#L4340) → `sample_tokens` [:4435](../vendor/vllm-0.24.0/vllm/v1/worker/gpu_model_runner.py#L4435) → `"sample"` [:4470](../vendor/vllm-0.24.0/vllm/v1/worker/gpu_model_runner.py#L4470) | GEMM logits trên vocab (Qwen vocab lớn) + sampling; **structured output** thêm grammar bitmask | Tắt `logprobs` nếu không cần; cân nhắc chi phí `--structured-outputs-config`; `--logits-processors` tối giản. |
| **P8. Eager vs graph** | prefill thường chạy eager (shape động) | Launch overhead từng kernel nếu eager thuần | `--enforce-eager` **tắt** graph toàn cục — tránh; để piecewise-compile lo prefill. |

### 4.3 `--async-scheduling` với Prefill/TTFT

`async_scheduling` bật ⇒ dùng `step_with_batch_queue` ([core.py:519](../vendor/vllm-0.24.0/vllm/v1/engine/core.py#L519))
với `max_concurrent_batches = 2` ([vllm.py:499-504](../vendor/vllm-0.24.0/vllm/config/vllm.py#L499-L504)):
schedule step kế **chồng lấn** với GPU exec step trước ⇒ giấu overhead CPU (P4) khỏi đường tới hạn. Lưu ý comment
[core.py:609-611](../vendor/vllm-0.24.0/vllm/v1/engine/core.py#L609-L611): xử lý deferred "hơi thiên vị TTFT so với TPOT".

---

## 5. BREAKDOWN — TPOT (`decode_time / (gen−1)`, phân rã mỗi decode step)

Mỗi decode step = 1 vòng `EngineCore.step()` sinh 1 token/req (hoặc >1 nếu spec-decode). Phân rã 1 step
(cùng các marker profiler ở §4.2, nhưng ở chế độ decode: mỗi req chỉ 1 token, **memory-bandwidth-bound**):

| Leaf | Phase / code | Root cause | Đòn bẩy |
|---|---|---|---|
| **T1. Scheduler CPU overhead** | `scheduler.schedule()` dựng batch mỗi step [scheduler.py:388-580](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L388-L580) | Python overhead/step (build block table, budget…) | **`--async-scheduling`** — đòn bẩy TPOT SỐ 1. Overlap schedule với GPU exec ([core.py:519](../vendor/vllm-0.24.0/vllm/v1/engine/core.py#L519), `max_concurrent_batches=2`). |
| **T2. Input prep (H2D)/step** | `"preprocess"` [:4094](../vendor/vllm-0.24.0/vllm/v1/worker/gpu_model_runner.py#L4094) | Copy CPU→GPU mỗi step | Bị `--async-scheduling` giấu; cudagraph cố định buffer. |
| **T3. Forward decode** | `"forward"` [:4326](../vendor/vllm-0.24.0/vllm/v1/worker/gpu_model_runner.py#L4326) | **Memory-bandwidth-bound**: nạp toàn bộ weight + KV mỗi token. Byte weight / BW = sàn TPOT | `--quantization fp8` (½ byte weight); giữ **cudagraph** (đừng `--enforce-eager`) để bỏ launch overhead; `--cudagraph-capture-sizes` phủ đúng batch decode; `-tp` chia weight. |
| **T4. Attention decode** | trong forward | Đọc KV cache: byte KV ∝ context_len × layers × heads | `--kv-cache-dtype fp8` (½ byte KV read); `--attention-backend`; `--block-size`; `--decode-context-parallel-size` cho context dài. |
| **T5. Sampling/step** | `"sample"` [:4470](../vendor/vllm-0.24.0/vllm/v1/worker/gpu_model_runner.py#L4470) | GEMM logits trên vocab mỗi token + sampling params | Tắt `logprobs`; giảm chi phí structured-output/logits-processors. |
| **T6. Spec-decode** | `"draft"` [:4495](../vendor/vllm-0.24.0/vllm/v1/worker/gpu_model_runner.py#L4495), `propose_draft_token_ids` [:4864](../vendor/vllm-0.24.0/vllm/v1/worker/gpu_model_runner.py#L4864) | Nếu bật: đề xuất k token/step, chấp nhận j ⇒ **chia TPOT cho j**. Ăn thua ở acceptance rate | **`--speculative-config`** (ngram/eagle/medusa) — đòn bẩy TPOT SỐ 2. Chỉnh `num_speculative_tokens`. Verify = rejection sampler. |
| **T7. Bookkeep/output** | `"bookkeep"` [:4586](../vendor/vllm-0.24.0/vllm/v1/worker/gpu_model_runner.py#L4586), `"ModelRunnerOutput"` [:4621](../vendor/vllm-0.24.0/vllm/v1/worker/gpu_model_runner.py#L4621) | CPU postprocess + `_to_list` (event.synchronize) | Phần lớn overlap; `--async-scheduling` giúp. |
| **T8. Detokenize/IPC/step** | output_handler async [async_llm.py:656](../vendor/vllm-0.24.0/vllm/v1/engine/async_llm.py#L656) | Detokenize + IPC mỗi token (overlap với step kế) | `--stream-interval N` gộp N token/lần gửi ([output_processor.py:287](../vendor/vllm-0.24.0/vllm/v1/engine/output_processor.py#L287)) — giảm tải frontend, **không** giảm TPOT engine. |
| **T9. Preemption khi decode** | preempt khi hết KV [scheduler.py:562-565](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L562-L565); recompute tính vào decode_time ([stats.py:447-448](../vendor/vllm-0.24.0/vllm/v1/metrics/stats.py#L447-L448)) | Hết KV giữa chừng ⇒ preempt + recompute ⇒ TPOT tăng vọt (spike) | `--gpu-memory-utilization` ↑; `--max-num-seqs` ↓ (ít req đồng thời ⇒ ít preempt); `--scheduler-reserve-full-isl` (reserve full ISL, [scheduler.py:883](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L883)) đánh đổi Queue lấy 0 preempt. |
| **T10. Batch size** | càng nhiều req running → step lâu hơn nhưng amortize | `max_num_seqs`/budget | `--max-num-seqs` là núm cân TPOT (per-req) ↔ throughput ↔ Queue. |

---

## 6. Bảng "một cờ tác động interval nào" (chỉ các cờ đáng chỉnh cho 1-model text-only)

| Cờ / hành động | Residual | Queue | Prefill | TPOT | Ghi chú đánh đổi |
|---|:--:|:--:|:--:|:--:|---|
| `--async-scheduling` | | | ⬇︎(nhẹ) | ⬇︎⬇︎ | Giấu CPU sched. Rủi ro rất thấp. **Bật trước tiên.** |
| `--speculative-config` (ngram/eagle) | | | | ⬇︎⬇︎ | TPOT ÷ acceptance. Ăn thua ở tỉ lệ chấp nhận. |
| `--max-num-seqs` ↑ | | ⬇︎ | | ⬆︎ | Admit nhiều hơn ⇒ Queue↓ nhưng TPOT↑ + nguy cơ preempt. |
| `--max-num-batched-tokens` ↑ | ⬆︎(A4) | ⬇︎ | ⬇︎(chunk to) | ⬆︎(step dài) | Núm cân Queue/Prefill ↔ TPOT + Residual_input. |
| `--long-prefill-token-threshold` ↑ | | | ⬇︎ | ⬆︎ | Chunk prefill to ⇒ ít step ⇒ TTFT prompt dài ↓, nhưng chèn decode nặng hơn. |
| `--quantization fp8` | | ⬇︎(nhiều block) | ⬇︎ | ⬇︎ | Ít byte weight ⇒ nhanh cả prefill lẫn decode. |
| `--kv-cache-dtype fp8` | | ⬇︎(nhiều block) | | ⬇︎(T4) | ½ byte KV. Cần calibrate/scale. |
| `--gpu-memory-utilization` ↑ | | ⬇︎(Q3) | | ⬇︎(T9 ít preempt) | Nhiều KV block. Coi chừng OOM/fragmentation. |
| `--enable-prefix-caching` | | ⬇︎(Q3) | ⬇︎⬇︎(P3) | | Vàng nếu có system prompt / prefix chung. |
| KHÔNG `--enforce-eager` (giữ cudagraph) | | | | ⬇︎⬇︎(T3) | enforce-eager làm TPOT tệ hẳn. |
| `--stream-interval N` | ⬇︎(B) | | | ~ | Giảm tải frontend, không đổi TPOT engine. |
| `--scheduler-reserve-full-isl` | | ⬆︎ | | ⬇︎(T9) | Đổi Queue lấy khử preempt-spike. |
| `--scheduling-policy priority` | | ⬇︎(req ưu tiên) | | | Cho SLA/vượt hàng có chọn lọc. |

---

## 7. Cách ĐO thật từng interval (không đo thì mọi tối ưu là mù)

vLLM đã tính sẵn `queued_time / prefill_time / decode_time / mean_time_per_output_token` **cho từng request**
trong `FinishedRequestStats` ([stats.py:224-239](../vendor/vllm-0.24.0/vllm/v1/metrics/stats.py#L224-L239)). Để lấy
per-request thay vì chỉ histogram Prometheus:

1. **Bật `--collect-detailed-traces`** (OTLP) hoặc log iteration details
   (`--enable-logging-iteration-details`, [core.py:435](../vendor/vllm-0.24.0/vllm/v1/engine/core.py#L435)).
2. **Đo Residual**: cần chèn timestamp ở frontend vì engine-core interval không thấy Residual. Patch tối thiểu ở
   `async_llm.add_request` (ghi `t_arrival`) và ở `output_handler` khi token đầu ra (ghi `t_first_out`); rồi
   `Residual = TTFT − queued_time − prefill_time`. (Đây là chỗ **code-mod nhỏ** đáng làm cho competition.)
3. **Đo phase GPU (P4-P8, T1-T7)**: bật `record_function` (torch profiler) — các marker `"gpu_model_runner:
   preprocess/forward/postprocess/sample/draft/bookkeep"` đã có sẵn, chỉ cần chạy profiler 1 lần để biết phase
   nào chiếm phần lớn step, từ đó chọn đúng cờ ở §3-§5. `--enable-layerwise-nvtx-tracing` cho chi tiết layer.

---

## 8. Root cause → đòn bẩy: thứ tự nên thử (cho L40S, Qwen3.5-2B, điểm = 50% TTFT + 50% TPOT)

1. **`--async-scheduling`** — giảm TPOT gần như free (giấu CPU sched). Bật đầu tiên.
2. **Cudagraph ON** (đừng `--enforce-eager`) + `--cudagraph-capture-sizes` phủ batch decode — nền TPOT.
3. **`--speculative-config` (ngram trước, eagle nếu có draft)** — đòn TPOT lớn nhất còn lại; đo acceptance rate.
4. **`--kv-cache-dtype fp8` + `--quantization fp8`** — ⚠️ **KHÔNG giảm cả 4 interval** (xem §9: thực đo L40S cho thấy fp8 **hại TTFT qua QUEUE**, chỉ lợi decode/TPOT + tiết kiệm KV).
5. **Cân `--max-num-seqs` & `--max-num-batched-tokens`** theo profiler: nếu Queue lớn ⇒ nới; nếu preempt-spike
   (T9) ⇒ siết + `--gpu-memory-utilization` ↑.
6. **`--enable-prefix-caching`** nếu trace có prefix chung — cắt thẳng Prefill (P3) + Queue (Q3).
7. **Residual**: gửi thẳng `prompt_token_ids` (bỏ tokenize A1) nếu pipeline cho phép; xét `--stream-interval`.
8. **Code-mod** (khi cờ hết dư địa): chèn instrument Residual (§7.2); tinh chỉnh điểm "deferred sampling"
   ([core.py:609-630](../vendor/vllm-0.24.0/vllm/v1/engine/core.py#L609-L630)) để thiên vị TTFT hoặc TPOT theo mục tiêu điểm.

> Mọi con số cụ thể (chunk size tối ưu, acceptance rate, batch sweet-spot) phải lấy từ **profiler + trace thật**
> trên GPU, không đoán. §7 là cách lấy chúng.

---

## 9. CẬP NHẬT (bench L40S thực đo, 2026-07) — sửa root-cause fp8 + thêm token-tax & overhead

> Mục §1-§8 phân rã theo interval là đúng. Nhưng §8 xếp `fp8` là "giảm cả 4 interval" — **SAI**.
> Đo thật (`output/l40s-abfp8-20260712`, `l40s-batchtok-20260712`, `l40s-profile-20260712`) lật lại
> như dưới. Đồng thời bổ sung 2 thành phần §1-§4 chưa có: **token-tax** và **launch-overhead**.

### 9.0 Trục thứ 2: HAI ĐỒNG HỒ (thiếu ở §4)
Mọi node compute phải đọc **2 số**:
- `wall` = đồng hồ thật (cái grader chấm).
- `gpu_busy` = Σ thời lượng kernel.
- **`overhead = wall − gpu_busy`** = launch/dispatch CPU + gap giữa kernel. **Đây là node hạng nhất**, và là
  chỗ chi phí fp8 trên Ada ẩn nấp (GPU-busy KHÔNG thấy nó).

### 9.1 Prefill mở rộng — thêm T3c (token-tax) và T3d (overhead)
```
Prefill (wall = gpu_busy + overhead)
├─ T3a own_compute vs interleave        (patch_sched_trace pref_ids)
├─ T3b GPU-BUSY compute (X) theo layer-group:
│    ├─ GEMM: Full-attn(qkv/o_proj,6 lớp) + GDN(in/out_proj,18 lớp) + MLP(gate/up/down) ← ~72% prefill
│    ├─ GDN-core (chunk_gated_delta_rule + causal_conv1d_fwd)   ← KHÔNG quantize
│    ├─ FullAttn kernel (flashinfer BatchPrefill)               ← KHÔNG quantize
│    └─ KV_write / norm / rope
├─ T3c TOKEN-TAX (Y) — CHỈ fp8:  quantize-activation mỗi Linear + cast BF16↔FP8 ở ranh giới đảo
└─ T3d LAUNCH/OVERHEAD (wall−busy): dispatch nhiều kernel hơn + nhiều chunk hơn (block-align 1072) +
     gap (Ada KHÔNG có DeepGEMM để fuse quant → gap lộ ra thành wall)
```
Vì sao Ada không vào fast path: `support_deep_gemm` chỉ Hopper/Blackwell
([cuda.py:663](../vendor/vllm-0.24.0/vllm/platforms/cuda.py#L663)); GDN prefill FlashInfer cũng chỉ SM90/Blackwell
([qwen_gdn_linear_attn.py:150](../vendor/vllm-0.24.0/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L150)).
`--quantization=fp8` là **online** (quantize weight lúc load + dynamic activation lúc chạy):
[online/fp8.py:105](../vendor/vllm-0.24.0/vllm/model_executor/layers/quantization/online/fp8.py#L105).

### 9.2 ROOT-CAUSE fp8 (sửa §8) — lỗ TTFT qua QUEUE, KHÔNG phải prefill
Bằng chứng per-request (mean 3 rep), turn0 từng user — cột **queue fp8 > noquant ở CẢ 20 user, đơn điệu**:

| user cuối burst | queue noquant | queue fp8 | Δ |
|---|---|---|---|
| user1 | 115 | 194 | +79 |
| user10 | 1297 | 1661 | +364 |
| user19 | 2596 | 3296 | **+700** |

- fp8 lỗ TTFT là do `kv-cache=fp8` ép **mamba block-align = 1072** (bf16 = 544):
  [scheduler.py `_mamba_block_aligned_split`](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L330).
  Ở budget 2048: 1072 lát ra **1 block** (phí 976, kẹt **1-lane** `n_prefilling=1`); 544 lát **3 block=1632** (2-lane).
- **KHÔNG phải block-align tồi vốn dĩ** — chỉ tồi khi budget=2048. Đo throughput thật:
  `fp8@b3216 → 2144 tok/iter (2 block, 2-lane) > noquant@b2048 → 1632 tok/iter`. Chỉnh budget là hết kẹt.
  (b2144 vô dụng: khít 2×1072, 1 token decode rớt xuống 1 block.)
- Nhưng queue fp8@3216 (467ms) vẫn chỉ **≈** noquant@2048 (440ms) — vì **token-tax kéo tok/giây fp8 về ngang**.
  → "sêm sêm, thế quantize làm gì" trên L40S là đúng.

### 9.3 Prefill COMPUTE của fp8 KHÔNG chậm — user0/turn0 là OUTLIER (không phải cold-start)
- Chuẩn hóa theo **uncached-token**: 18/20 user turn0 fp8 prefill **nhanh hơn** noquant; mọi turn-aggregate fp8 nhanh hơn.
- Chỉ **user0** (unc=12947, request đầu, 1 chunk lớn) fp8 chậm (30.6 vs 22.6 µs/utok) — đây là chỗ cả narrative
  cũ bị dựng lên. Trace code: `apply()` fp8 **không** có lazy/autotune/first-call
  ([online/fp8.py:348](../vendor/vllm-0.24.0/vllm/model_executor/layers/quantization/online/fp8.py#L348);
  grep `autotune|lazy|cache` trong `scaled_mm/cutlass.py` = rỗng) → **KHÔNG có cold-start fp8**. Chậm ổn định qua
  mọi budget (396@2048, 386@3216) ⇒ là **per-token tax/overhead**, không phải one-time.
- GPU-busy trace (interleave-free) xác nhận fp8 GEMM nhanh hơn (16.9 vs 20.7ms) — nên phần fp8-slower ở wall là
  **overhead (T3d) + tax (T3c)**, không phải compute (X).

### 9.4 Decode/TPOT — nơi fp8 THẮNG (khớp Tầng 5 §profiler)
`gpu_busy` decode per-step: fp8 **5.7ms** vs noquant **7.3ms** (−22%), dồn vào **GEMM weight-read** (fp8 nửa byte).
KV-fp8 tiết kiệm memory + băng thông đọc KV. Đây là lý do trên **H200-MIG 18GB (memory-bind)** fp8 thắng cả TTFT
(giữ đủ lane, ít preempt) — ngược L40S (VRAM dư, không bind).

### 9.5 Công cụ đo (userspace, KHÔNG sudo — GPU thuê không có root)
| script | đo | ghi chú no-sudo |
|---|---|---|
| `scripts/gpu-l40s-bench/measure_preflight.sh` | check + **log clock/nhiệt/power 1s** | không lock được clock → log để loại mẫu throttle |
| `scripts/gpu-l40s-bench/profile_compute_ab.py` | wall-sạch (perf_counter) + trace gpu_busy, A/B, prefill+decode | `VLLM_ENABLE_V1_MULTIPROCESSING=0` cho NVTX hook |
| `scripts/gpu-l40s-bench/parse_compute.py` | T3b layer-group + T3c tax + **T3d overhead=wall−busy** + D1 | offline thuần Python |
| `scripts/gpu-l40s-bench/nsys_compute.sh` | launch-gap sạch (tùy chọn) | `--sample=none --cpuctxsw=none` (không cần root) |
| `scripts/gpu-l40s-bench/run_compute_breakdown.sh` | orchestrator tầng 4-5 | chạy sau `run_v28_baseline.sh` |

> Chưa đo sạch: X-vs-Y ở đúng shape 12947-token (GPU-busy hiện có ở 1040-token, ngoại suy có thể sai). `nsys`
> hoặc `wall−busy` ở đúng shape (script trên) sẽ chốt. **Mọi A/B phải trong CÙNG 1 lần thuê GPU.**
