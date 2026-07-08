# Hướng dẫn đọc metrics AIPerf (Track 3)

Tài liệu này giải thích **ý nghĩa từng metric** mà AIPerf thu được khi chạy
`MODE=replay` trên trace thật, kèm **ví dụ minh hoạ lấy từ run thật** (20 user x
6 turn). Mục tiêu: nhìn một con số là hiểu nó nói gì về serving.

> Các ví dụ dưới đây lấy từ một lần chạy thật (`scripts/07_per_request_report.py`).
> Con số cụ thể sẽ khác tuỳ GPU/config, nhưng *cách đọc* thì không đổi.

---

## 0. AIPerf sinh ra gì — 3 tầng dữ liệu

Mỗi lần chạy, AIPerf ghi vào `artifacts/<run>/`:

```
                      ┌─────────────────────────────────────────────┐
   client (AIPerf) ──►│ profile_export.jsonl   (1 dòng / 1 request)  │
      bắn request     │   • metrics   : đo từ phía client (TTFT,...)  │
                      │   • metadata  : mốc thời gian + định danh     │
                      └─────────────────────────────────────────────┘
                      ┌─────────────────────────────────────────────┐
   cào /metrics ~333ms│ server_metrics_export.jsonl (time-series)    │
      của vLLM        │   • trạng thái ENGINE toàn cục (KV%, queue,  │
                      │     prefix cache, preemption, ...)           │
                      └─────────────────────────────────────────────┘
```

Phân biệt quan trọng:

- **profile_export** = đo lường **từng request** (chính xác cho request đó).
- **server_metrics** = trạng thái **toàn engine tại thời điểm cào**, dùng chung
  cho mọi request đang chạy → chỉ khớp được theo *thời điểm gần nhất*, không phải
  số riêng của 1 request. (Vì vậy report ghi chú rõ cột KV%/hits là "global".)

---

## 1. Đường đời một request — các mốc thời gian

Tất cả mốc là nanosecond tuyệt đối; report đổi về **ms tương đối so với request
đầu tiên** cho dễ đọc.

```
 credit_issued        request_start      (first token)                 request_end
      │                     │                  │                             │
      ▼                     ▼                  ▼                             ▼
  [ AIPerf phát ]──queue──►[ lên wire ]──── TTFT ────►[ token1 token2 ... ]─►[ xong ]
   theo lịch trace        thực sự gửi                  │◄── TPOT giữa các token ──►│
                                                        │◄──────── request_latency ───────►│
```

| Cột report | Field gốc | Nghĩa |
|---|---|---|
| `arr`   | `credit_issued_ns` | Lúc AIPerf **phát** request theo lịch của trace (≈ `timestamp_ms` gốc). Đây là "giờ đến". |
| `start` | `request_start_ns` | Lúc request **thực sự lên wire**. |
| `queue` | `start − arr` | Trễ dispatch phía **client** (thường chỉ 2–3ms). |
| `end`   | `request_end_ns` | Lúc **response stream xong** hoàn toàn. |
| `lat`   | `request_latency` | `end − start` = tổng thời gian request. |

> **Ví dụ (user 0, turn 0):** `arr=0.0  start=3.6  end=6594.2  queue=3.6  lat=6590.5`
> → Request đến lúc t=0, chờ 3.6ms rồi lên wire, chạy tổng cộng 6.59 giây.
>
> ⚠️ Lưu ý: cột `queue` chỉ là trễ phía client (~3ms). Còn việc **chờ vì 20 user
> ập vào cùng lúc** thì nằm BÊN TRONG `TTFT` (server xếp hàng trước khi prefill).
> Muốn tách riêng phần chờ ở server → dùng `vllm:request_queue_time_seconds` (§5,
> hiện chưa khai thác).

---

## 2. Latency phía client

| Cột / field | Nghĩa | Ví dụ (real) |
|---|---|---|
| `TTFT` (`time_to_first_token`) | Từ lúc gửi đến **token đầu tiên**. Gồm: chờ ở server + **prefill** cả prompt. Đây là thứ prefix-cache tác động trực tiếp. | turn0=1253ms, turn1=262ms |
| `TPOT` (`inter_token_latency`) | Trễ **trung bình giữa các token** khi decode. Phản ánh tốc độ sinh, KHÔNG bị prefix-cache ảnh hưởng. | ~18–27ms/token |
| `time_to_second_token` | Riêng khoảng từ token 1 → token 2 (tách khỏi TTFT). | (chưa hiện) |
| `inter_chunk_latency` | **List trễ của TỪNG token** (không chỉ trung bình) — dùng để thấy jitter / token bị khựng. | `[11.5, 10.4, 11.9, ...]` |
| `output_token_throughput_per_user` | Token/giây/user riêng request này. | ~60 tok/s |
| `e2e_output_token_throughput` | Throughput end-to-end (tính cả TTFT). | ~42 tok/s |

> **Đọc TTFT cho ra câu chuyện prefix-cache:**
> - turn 0: `TTFT=1253ms` — 20 user ập vào t=0..475ms, 20 prefill dài (~13k token)
>   tranh nhau → xếp hàng + prefill đầy đủ. Đây là chi phí "cold".
> - turn 1: `TTFT=262ms` — lịch sử chung đã nằm trong KV cache, chỉ còn ~2.9k
>   token mới phải prefill → TTFT tụt ~5 lần. **Prefix cache đang làm việc.**
>
> TPOT gần như không đổi (~18ms) qua các turn → đúng như lý thuyết: cache giúp
> prefill (TTFT), không giúp decode (TPOT).

---

## 3. Đếm token (in / out / cache)

| Cột / field | Nghĩa | Ví dụ (real) |
|---|---|---|
| `in_tok` (`input_sequence_length` / `usage_prompt_tokens`) | Số token prompt request này gửi vào. | turn0≈12.9k → turn5≈27k |
| `out` (`output_sequence_length`) | Số token sinh ra. | 200 (max_tokens của trace) |
| `cache_rd` (`usage_prompt_cache_read_tokens`) | **Số token prompt HIT prefix-cache** — phần không phải prefill lại. | (cần `--use-server-token-count`) |
| `hit%` (`cache_rd / in_tok`) | **% prompt của request này tái dùng từ KV cache.** Đây chính là "request tận dụng được bao nhiêu % cache". | turn0≈0% → turn sau tăng dần |
| `usage_prompt_cache_miss_tokens` | Phần prompt phải tính lại (miền cache). | (trong CSV) |

> **Vì sao `in_tok` tăng dần mỗi turn?** Vì mỗi turn AIPerf gửi LẠI cả lịch sử:
> turn 1 = system + Q1 + A1 + Q2; turn 2 = ... + A2 + Q3; ... Nên prompt phình ra
> ~2.9k token/turn. Nhưng `hit%` cũng tăng theo → tuy input to hơn, **phần lặp lại
> đã cache** nên prefill vẫn rẻ → TTFT không tăng tương ứng. Đó là mục tiêu của bài.
>
> **Ví dụ đọc:** nếu turn 3 có `in_tok=19000` và `cache_rd=15000` → `hit%≈79%`,
> nghĩa là 79% prompt lấy từ cache, chỉ 21% (~4000 token) thực sự phải prefill.

---

## 4. Bóc tách HTTP (nhóm `http_req_*`)

Các field này bóc thời gian một request HTTP thành từng chặng — hữu ích khi nghi
ngờ **độ trễ do mạng/kết nối** chứ không phải do model.

| Field | Nghĩa |
|---|---|
| `http_req_waiting` | Chờ server xử lý trước byte đầu (~TTFB mạng). |
| `http_req_receiving` / `sending` | Thời gian nhận / gửi dữ liệu trên wire. |
| `http_req_connecting` / `dns_lookup` / `blocked` | Chi phí bắt tay kết nối, tra DNS. |
| `http_req_duration` / `total` / `connection_overhead` | Tổng hợp các chặng trên. |
| `http_req_chunks_received` / `sent` | Số SSE chunk (≈ số token + 1). |
| `http_req_data_received` / `sent` (KB) | Bytes truyền đi/về. |

> Khi benchmark trên localhost (cùng máy) các số này rất nhỏ — chỉ đáng quan tâm
> khi client và server ở **máy khác nhau** (độ trễ mạng làm nhiễu TTFT).

---

## 5. Server-side vLLM (`server_metrics_export.jsonl`)

Đây là trạng thái **engine toàn cục**, cào ~333ms/lần. Report khớp mỗi request với
lần cào **gần nhất lúc nó kết thúc**. KHÔNG phải số riêng từng request (trừ các
histogram `vllm:request_*`).

### Đã dùng trong report
| Field | Nghĩa | Ví dụ |
|---|---|---|
| `vllm:kv_cache_usage_perc` | % pool KV cache đang dùng. **Thấp = còn headroom**, không phải ít cache. Tăng `gpu-memory-utilization` → pool to hơn → % này tụt. | 25% → 46% qua các turn |
| `vllm:num_requests_running` | Số request chạy đồng thời. | 20–32 |
| `vllm:prefix_cache_hits` / `queries` | Đếm token hit / query prefix (toàn cục, tích luỹ). `hit_rate = hits/queries`. | ≈ 4.99M/6.47M ≈ **77%** |

### Có sẵn nhưng CHƯA khai thác (đáng giá)
| Field | Nghĩa | Vì sao đáng lấy |
|---|---|---|
| `vllm:num_preemptions` | Số lần **evict/preempt** request vì thiếu KV. | `>0` = pool KV quá nhỏ → tăng gpu-mem-util. Tín hiệu trực tiếp. |
| `vllm:num_requests_waiting` (+ `_by_reason`: capacity/deferred) | Số request đang chờ + lý do. | Thấy điểm nghẽn thực sự. |
| `vllm:request_prefill_time_seconds` | **Thời gian prefill thuần** (server đo). | Tách prefill khỏi queue trong TTFT — chính xác hơn suy tay. |
| `vllm:request_prefill_kv_computed_tokens` | **Số token KV thực sự phải tính** khi prefill (phần KHÔNG cache). | Đo trực tiếp "công prefill thật sự" mỗi request. |
| `vllm:request_queue_time_seconds` | Thời gian chờ hàng đợi server-side. | Phần "chờ vì burst" thật sự (khác cột `queue` client ~3ms). |
| `vllm:request_decode_time_seconds` / `time_per_output_token_seconds` | Thời gian decode server-side. | Đối chiếu với TPOT client. |
| `vllm:time_to_first_token_seconds` / `inter_token_latency_seconds` | TTFT/ITL đo TỪ SERVER. | Đối chiếu client vs server (lọc nhiễu mạng). |
| `vllm:prompt_tokens_cached` / `prompt_tokens_by_source` | Token cache / nguồn prompt (mới vs cache). | Bức tranh cache toàn cục. |
| `vllm:estimated_flops_per_gpu` / `read_bytes` / `write_bytes` | Ước lượng tải compute / băng thông. | Xem nghẽn compute hay memory-bound. |
| `vllm:iteration_tokens_total` / `generation_tokens` | Throughput theo iteration. | Xu hướng throughput theo thời gian. |

### Ít giá trị / không áp dụng
`vllm:mm_cache_*` (multimodal — model text không dùng), `process_*`, `python_gc_*`,
`http_request_*` (metric hệ thống của process vLLM).

---

## 6. Đọc trọn vẹn report — giải thích TỪNG cột

Header đầy đủ (khi run có usage cache + server metrics):

```
user turn      arr    start      end  queue    TTFT   TPOT in_tok   out      lat cache_rd hit%    KV%  run   Δhits    Δqry
   0    0      0.0      3.4   9038.4    3.4  2498.0   32.8  12947   200   9035.1        -    -  30.2%   40 1398080 2357238   ← turn 0: cold, chưa cache
   0    3  14999.2  15001.5  18978.8    2.4   269.8   18.6  21599   200   3977.3        -    -  35.7%   20 2971872 4209224   ← turn 3: input TO HƠN mà TTFT nhỏ hơn 9× (cache ăn)
```

### Giải thích từng cột (kèm level)

| Cột | Ví dụ (u0 t0) | Nghĩa | Level |
|---|---|---|---|
| `user` | 0 | User 0..19, **= session_num % 20** | 🟡 script 07 phái sinh |
| `turn` | 0 | Lượt 0..5, **= session_num // 20** | 🟡 phái sinh |
| `arr` | 0.0 | **Arrival** — lúc AIPerf phát request theo lịch trace (`credit_issued_ns`), ms từ t0. ≈ `timestamp_ms` gốc | ① per-request |
| `start` | 3.4 | Lúc request **lên wire** (`request_start_ns`) | ① per-request |
| `end` | 9038.4 | Lúc **response stream xong** (`request_end_ns`) | ① per-request |
| `queue` | 3.4 | `start − arr` = trễ dispatch **phía client** (~vài ms). *Chờ vì burst nằm trong TTFT, không phải ở đây* | 🟡 phái sinh |
| `TTFT` | 2498.0 | **Time to first token** (ms) = chờ server + prefill. Prefix cache tác động thẳng vào đây | ① per-request |
| `TPOT` | 32.8 | **Inter-token latency** trung bình (ms/token) khi decode | ① per-request |
| `in_tok` | 12947 | **Prompt tokens** (`usage_prompt_tokens` / ISL) | ① per-request |
| `out` | 200 | **Output tokens** sinh ra | ① per-request |
| `lat` | 9035.1 | **Request latency** = `end − start` (ms) | ① per-request |
| `cache_rd` | – | **Token prompt HIT prefix-cache** (`cached_tokens`). `–` khi server chưa trả — cần serve `--enable-prompt-tokens-details` | ① per-request |
| `hit%` | – | `cache_rd / in_tok` = **% prompt tái dùng từ cache** của request đó | 🟡 phái sinh |
| `KV%` | 30.2% | `vllm:kv_cache_usage_perc` tại scrape gần lúc `end`. **GLOBAL**, không riêng request | ③ server global |
| `run` | 40 | `vllm:num_requests_running` — số request chạy song song lúc đó. **GLOBAL** | ③ server global |
| `Δhits` | 1398080 | `vllm:prefix_cache_hits` **tích luỹ từ đầu run** (đơn vị token). **GLOBAL** | ③ server global |
| `Δqry` | 2357238 | `vllm:prefix_cache_queries` tích luỹ (token). **GLOBAL**. Hit rate tổng = `Δhits/Δqry` | ③ server global |

### Đọc 2 dòng ví dụ ra "tiếng người"

> **u0 t0 (cold):** request đến lúc t=0, lên wire sau 3.4ms. Prompt **12.947 token**
> (turn đầu — chưa có gì trong cache). 20 user ập vào cùng lúc (`run=40` do cả turn
> trước còn chạy) → xếp hàng + prefill đầy → **TTFT 2498ms**. Sinh 200 token, mỗi
> token ~33ms, xong lúc t=9038ms, tổng **9.0 giây**.
>
> **u0 t3 (cached):** prompt giờ **21.599 token — TO HƠN** (lịch sử dồn thêm), nhưng
> **TTFT chỉ 270ms — nhỏ hơn 9 lần!** Vì phần lịch sử chung đã nằm trong KV cache →
> chỉ prefill ~2.9k token mới. **Đây chính là prefix caching, nhìn thẳng trên bảng:
> input phình mà TTFT tụt.**

> **Đọc cache toàn cục ngay cả khi cột `hit%` trống:** lấy `Δhits/Δqry` dòng cuối
> cùng của run → ví dụ `5.236.544 / 6.799.373 ≈ 77%` = tỉ lệ token prompt tái dùng
> cache cho cả run.

---

## 7. Công thức hữu ích

| Muốn biết | Tính bằng |
|---|---|
| Hit rate prefix toàn cục | `vllm:prefix_cache_hits / vllm:prefix_cache_queries` |
| % cache 1 request tận dụng | `cache_rd / in_tok` (cột `hit%`) |
| Token thật sự phải prefill | `in_tok − cache_rd` (≈ `request_prefill_kv_computed_tokens`) |
| TPOT từ latency | `(request_latency − TTFT) / (out_tok − 1)` |
| Prefill time (xấp xỉ) | `TTFT − queue_server` (chính xác: `vllm:request_prefill_time_seconds`) |
| Throughput tổng | `tổng_out_tok / benchmark_duration` |

---

## 8. Trạng thái khai thác (recap nhanh)

- ✅ **Đang dùng:** TTFT, TPOT, latency, in/out token, cache_rd + hit% (per-request),
  mốc arr/start/end/queue, KV%, running, prefix hits/queries, mapping user/turn.
- 🟡 **Thủ công/suy ra:** user/turn (từ `session_num`), ISL fallback bằng tokenizer,
  hit rate tổng, bảng aggregate của AIPerf (chưa parse lại).
- ⬜ **Chưa khai thác (đáng giá nhất):** `inter_chunk_latency` (jitter từng token),
  `vllm:request_prefill_time_seconds` + `request_prefill_kv_computed_tokens` (bóc
  tách prefill vs cache chính xác), `vllm:num_preemptions` (eviction),
  `num_requests_waiting_by_reason` (nghẽn), và **GPU telemetry** (util/power/mem —
  run hiện báo "No GPU telemetry collected", chưa cấu hình).
