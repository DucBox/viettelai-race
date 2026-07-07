# Huong dan doc metrics AIPerf (Track 3)

Tai lieu nay giai thich **y nghia tung metric** ma AIPerf thu duoc khi chay
`MODE=replay` tren trace that, kem **vi du minh hoa lay tu run that** (20 user x
6 turn). Muc tieu: nhin mot con so la hieu no noi gi ve serving.

> Cac vi du duoi day lay tu mot lan chay that (`scripts/07_per_request_report.py`).
> Con so cu the se khac tuy GPU/config, nhung *cach doc* thi khong doi.

---

## 0. AIPerf sinh ra gi — 3 tang du lieu

Moi lan chay, AIPerf ghi vao `artifacts/<run>/`:

```
                      ┌─────────────────────────────────────────────┐
   client (AIPerf) ──►│ profile_export.jsonl   (1 dong / 1 request)  │
      ban request     │   • metrics   : do tu phia client (TTFT,...)  │
                      │   • metadata  : moc thoi gian + dinh danh     │
                      └─────────────────────────────────────────────┘
                      ┌─────────────────────────────────────────────┐
   cao /metrics ~333ms│ server_metrics_export.jsonl (time-series)    │
      cua vLLM        │   • trang thai ENGINE toan cuc (KV%, queue,  │
                      │     prefix cache, preemption, ...)           │
                      └─────────────────────────────────────────────┘
```

Phan biet quan trong:

- **profile_export** = do luong **tung request** (chinh xac cho request do).
- **server_metrics** = trang thai **toan engine tai thoi diem cao**, dung chung
  cho moi request dang chay → chi khop duoc theo *thoi diem gan nhat*, khong phai
  so rieng cua 1 request. (Vi vay report ghi chu ro cot KV%/hits la "global".)

---

## 1. Duong doi mot request — cac moc thoi gian

Tat ca moc la nanosecond tuyet doi; report doi ve **ms tuong doi so voi request
dau tien** cho de doc.

```
 credit_issued        request_start      (first token)                 request_end
      │                     │                  │                             │
      ▼                     ▼                  ▼                             ▼
  [ AIPerf phat ]──queue──►[ len wire ]──── TTFT ────►[ token1 token2 ... ]─►[ xong ]
   theo lich trace        thuc su gui                  │◄── TPOT giua cac token ──►│
                                                        │◄──────── request_latency ───────►│
```

| Cot report | Field goc | Nghia |
|---|---|---|
| `arr`   | `credit_issued_ns` | Luc AIPerf **phat** request theo lich cua trace (≈ `timestamp_ms` goc). Day la "gio den". |
| `start` | `request_start_ns` | Luc request **thuc su len wire**. |
| `queue` | `start − arr` | Tre dispatch phia **client** (thuong chi 2–3ms). |
| `end`   | `request_end_ns` | Luc **response stream xong** hoan toan. |
| `lat`   | `request_latency` | `end − start` = tong thoi gian request. |

> **Vi du (user 0, turn 0):** `arr=0.0  start=3.6  end=6594.2  queue=3.6  lat=6590.5`
> → Request den luc t=0, cho 3.6ms roi len wire, chay tong cong 6.59 giay.
>
> ⚠️ Luu y: cot `queue` chi la tre phia client (~3ms). Con viec **cho vi 20 user
> ap vao cung luc** thi nam BEN TRONG `TTFT` (server xep hang truoc khi prefill).
> Muon tach rieng phan cho o server → dung `vllm:request_queue_time_seconds` (§5,
> hien chua khai thac).

---

## 2. Latency phia client

| Cot / field | Nghia | Vi du (real) |
|---|---|---|
| `TTFT` (`time_to_first_token`) | Tu luc gui den **token dau tien**. Gom: cho o server + **prefill** ca prompt. Day la thu prefix-cache tac dong truc tiep. | turn0=1253ms, turn1=262ms |
| `TPOT` (`inter_token_latency`) | Tre **trung binh giua cac token** khi decode. Phan anh toc do sinh, KHONG bi prefix-cache anh huong. | ~18–27ms/token |
| `time_to_second_token` | Rieng khoang tu token 1 → token 2 (tach khoi TTFT). | (chua hien) |
| `inter_chunk_latency` | **List tre cua TUNG token** (khong chi trung binh) — dung de thay jitter / token bi khung. | `[11.5, 10.4, 11.9, ...]` |
| `output_token_throughput_per_user` | Token/giay/user rieng request nay. | ~60 tok/s |
| `e2e_output_token_throughput` | Throughput end-to-end (tinh ca TTFT). | ~42 tok/s |

> **Doc TTFT cho ra cau chuyen prefix-cache:**
> - turn 0: `TTFT=1253ms` — 20 user ap vao t=0..475ms, 20 prefill dai (~13k token)
>   tranh nhau → xep hang + prefill day du. Day la chi phi "cold".
> - turn 1: `TTFT=262ms` — lich su chung da nam trong KV cache, chi con ~2.9k
>   token moi phai prefill → TTFT tut ~5 lan. **Prefix cache dang lam viec.**
>
> TPOT gan nhu khong doi (~18ms) qua cac turn → dung nhu ly thuyet: cache giup
> prefill (TTFT), khong giup decode (TPOT).

---

## 3. Dem token (in / out / cache)

| Cot / field | Nghia | Vi du (real) |
|---|---|---|
| `in_tok` (`input_sequence_length` / `usage_prompt_tokens`) | So token prompt request nay gui vao. | turn0≈12.9k → turn5≈27k |
| `out` (`output_sequence_length`) | So token sinh ra. | 200 (max_tokens cua trace) |
| `cache_rd` (`usage_prompt_cache_read_tokens`) | **So token prompt HIT prefix-cache** — phan khong phai prefill lai. | (can `--use-server-token-count`) |
| `hit%` (`cache_rd / in_tok`) | **% prompt cua request nay tai dung tu KV cache.** Day chinh la "request tan dung duoc bao nhieu % cache". | turn0≈0% → turn sau tang dan |
| `usage_prompt_cache_miss_tokens` | Phan prompt phai tinh lai (mien cache). | (trong CSV) |

> **Vi sao `in_tok` tang dan moi turn?** Vi moi turn AIPerf gui LAI ca lich su:
> turn 1 = system + Q1 + A1 + Q2; turn 2 = ... + A2 + Q3; ... Nen prompt phinh ra
> ~2.9k token/turn. Nhung `hit%` cung tang theo → tuy input to hon, **phan lap lai
> da cache** nen prefill van re → TTFT khong tang tuong ung. Do la muc tieu cua bai.
>
> **Vi du doc:** neu turn 3 co `in_tok=19000` va `cache_rd=15000` → `hit%≈79%`,
> nghia la 79% prompt lay tu cache, chi 21% (~4000 token) thuc su phai prefill.

---

## 4. Boc tach HTTP (nhom `http_req_*`)

Cac field nay bocs thoi gian mot request HTTP thanh tung chang — huu ich khi nghi
ngo **do tre do mang/ket noi** chu khong phai do model.

| Field | Nghia |
|---|---|
| `http_req_waiting` | Cho server xu ly truoc byte dau (~TTFB mang). |
| `http_req_receiving` / `sending` | Thoi gian nhan / gui du lieu tren wire. |
| `http_req_connecting` / `dns_lookup` / `blocked` | Chi phi bat tay ket noi, tra DNS. |
| `http_req_duration` / `total` / `connection_overhead` | Tong hop cac chang tren. |
| `http_req_chunks_received` / `sent` | So SSE chunk (≈ so token + 1). |
| `http_req_data_received` / `sent` (KB) | Bytes truyen di/ve. |

> Khi benchmark tren localhost (cung may) cac so nay rat nho — chi dang quan tam
> khi client va server o **may khac nhau** (do tre mang lam nhieu TTFT).

---

## 5. Server-side vLLM (`server_metrics_export.jsonl`)

Day la trang thai **engine toan cuc**, cao ~333ms/lan. Report khop moi request voi
lan cao **gan nhat luc no ket thuc**. KHONG phai so rieng tung request (tru cac
histogram `vllm:request_*`).

### Da dung trong report
| Field | Nghia | Vi du |
|---|---|---|
| `vllm:kv_cache_usage_perc` | % pool KV cache dang dung. **Thap = con headroom**, khong phai it cache. Tang `gpu-memory-utilization` → pool to hon → % nay tut. | 25% → 46% qua cac turn |
| `vllm:num_requests_running` | So request chay dong thoi. | 20–32 |
| `vllm:prefix_cache_hits` / `queries` | Dem token hit / query prefix (toan cuc, tich luy). `hit_rate = hits/queries`. | ≈ 4.99M/6.47M ≈ **77%** |

### Co san nhung CHUA khai thac (dang gia)
| Field | Nghia | Vi sao dang lay |
|---|---|---|
| `vllm:num_preemptions` | So lan **evict/preempt** request vi thieu KV. | `>0` = pool KV qua nho → tang gpu-mem-util. Tin hieu true tiep. |
| `vllm:num_requests_waiting` (+ `_by_reason`: capacity/deferred) | So request dang cho + ly do. | Thay diem nghen thuc su. |
| `vllm:request_prefill_time_seconds` | **Thoi gian prefill thuan** (server do). | Tach prefill khoi queue trong TTFT — chinh xac hon suy tay. |
| `vllm:request_prefill_kv_computed_tokens` | **So token KV thuc su phai tinh** khi prefill (phan KHONG cache). | Do truc tiep "cong prefill that su" moi request. |
| `vllm:request_queue_time_seconds` | Thoi gian cho hang doi server-side. | Phan "cho vi burst" that su (khac cot `queue` client ~3ms). |
| `vllm:request_decode_time_seconds` / `time_per_output_token_seconds` | Thoi gian decode server-side. | Doi chieu voi TPOT client. |
| `vllm:time_to_first_token_seconds` / `inter_token_latency_seconds` | TTFT/ITL do TU SERVER. | Doi chieu client vs server (loc nhieu mang). |
| `vllm:prompt_tokens_cached` / `prompt_tokens_by_source` | Token cache / nguon prompt (moi vs cache). | Buc tranh cache toan cuc. |
| `vllm:estimated_flops_per_gpu` / `read_bytes` / `write_bytes` | Uoc luong tai compute / bang thong. | Xem nghen compute hay memory-bound. |
| `vllm:iteration_tokens_total` / `generation_tokens` | Throughput theo iteration. | Xu huong throughput theo thoi gian. |

### It gia tri / khong ap dung
`vllm:mm_cache_*` (multimodal — model text khong dung), `process_*`, `python_gc_*`,
`http_request_*` (metric he thong cua process vLLM).

---

## 6. Doc tron ven MOT dong report

```
user turn      arr    start      end  queue    TTFT   TPOT in_tok   out      lat    KV%  run
   0    0      0.0      3.6   6594.2    3.6  1253.5   26.8   12900   200   6590.5  25.4%   32
```

Dich ra tieng nguoi:

> "Request cua **user 0, turn 0** den luc **t=0ms**, len wire sau **3.6ms**.
> Prompt **~12.900 token** (turn dau, chua co gi trong cache). Vi 20 user ap vao
> cung luc nen phai xep hang + prefill day → mat **1.25 giay** toi token dau
> (`TTFT`). Sau do sinh 200 token, moi token ~**27ms** (`TPOT`), xong luc
> **t=6594ms**, tong **6.59 giay**. Luc no xong, engine dung **25%** KV pool va co
> **32** request chay song song."

So sanh voi turn 1 cung user (`TTFT=262ms`): lich su da cache → prefill it hon
nhieu → nhanh gap ~5 lan. Do la gia tri cua prefix caching, nhin thang tren bang.

---

## 7. Cong thuc huu ich

| Muon biet | Tinh bang |
|---|---|
| Hit rate prefix toan cuc | `vllm:prefix_cache_hits / vllm:prefix_cache_queries` |
| % cache 1 request tan dung | `cache_rd / in_tok` (cot `hit%`) |
| Token that su phai prefill | `in_tok − cache_rd` (≈ `request_prefill_kv_computed_tokens`) |
| TPOT tu latency | `(request_latency − TTFT) / (out_tok − 1)` |
| Prefill time (xap xi) | `TTFT − queue_server` (chinh xac: `vllm:request_prefill_time_seconds`) |
| Throughput tong | `tong_out_tok / benchmark_duration` |

---

## 8. Trang thai khai thac (recap nhanh)

- ✅ **Dang dung:** TTFT, TPOT, latency, in/out token, cache_rd + hit% (per-request),
  moc arr/start/end/queue, KV%, running, prefix hits/queries, mapping user/turn.
- 🟡 **Thu cong/suy ra:** user/turn (tu `session_num`), ISL fallback bang tokenizer,
  hit rate tong, bang aggregate cua AIPerf (chua parse lai).
- ⬜ **Chua khai thac (dang gia nhat):** `inter_chunk_latency` (jitter tung token),
  `vllm:request_prefill_time_seconds` + `request_prefill_kv_computed_tokens` (boc
  tach prefill vs cache chinh xac), `vllm:num_preemptions` (eviction),
  `num_requests_waiting_by_reason` (nghen), va **GPU telemetry** (util/power/mem —
  run hien bao "No GPU telemetry collected", chua cau hinh).
