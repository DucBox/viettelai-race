# TTFT/TPOT — trace tận code implementation vLLM 0.22.1 + bản đồ cờ

> Trace trực tiếp source trong image (`/venv/main` = `dist-packages/vllm`, v0.22.1).
> Mục tiêu: bóc TTFT/TPOT tới lá nhỏ nhất **kèm file:line code thật**, rồi map mỗi lá
> với cờ tune được nó + **logic code của cờ**. Mọi kết luận "cờ X không giúp" đều có
> A/B đo thật hoặc dòng code chứng minh (không đoán từ help-text).

---

## PHẦN 1 — Vòng đời 1 request, đóng dấu timestamp ở đâu

```
client_send
 │  network + HTTP parse (uvicorn/h11)
 ▼
create_chat_completion()                      entrypoints/openai/chat_completion/serving.py:228
 │  259:  await render_chat_request()  ──►  render_chat (CHAT TEMPLATE + TOKENIZE)
 │                                          renderers/hf.py  (tokenize = tokenizer.encode)
 │  350:  engine_client.generate()  ──►  AsyncLLM.add_request()  v1/engine/async_llm.py:280
 │            └ 349: input_processor.process_inputs(arrival_time=None)
 ▼                     v1/engine/input_processor.py:234
 ┌─ 270/281:  arrival_time = time.time()      ◄══ ARRIVAL STAMP (SAU render+tokenize!)
 │  288:  current_platform.validate_request()
 │  291:  _validate_model_inputs()             (quét token, dài theo prompt)
 │  307-320: sampling_params.clone()+update
 │  build EngineCoreRequest
 │  412:  engine_core.add_request_async()  ──► IPC sang engine-core process
 ▼
ENGINE-CORE (process riêng, ĐƠN LUỒNG):       v1/engine/core.py
 │  input socket đọc request TUẦN TỰ
 │  add_request(): 208-212 request_block_hasher  ──► HASH block prefix (dài theo prompt)
 │  scheduler.add_request()  ──►  QUEUED event   ◄══ queued_ts   (stats.py:418)
 ▼
 │  ... chờ FCFS ...                              ◄══ queue = scheduled_ts − queued_ts
 │  scheduler.schedule()  ──► SCHEDULED event     ◄══ scheduled_ts (stats.py:421)
 │  ... chunked prefill nhiều iteration ...
 │  first NEW_TOKEN                               ◄══ first_token_ts   prefill = first_token_ts − scheduled_ts
 ▼
 │  detokenize token đầu + stream SSE + network
 ▼
client_first_token_arrival                       ◄══ client ttft_ms
```

### Hệ quả — 4 khoảng đo được (đã verify reconcile |Σ−cha|<1ms, 720/720 request)

| Khoảng | = | Gồm gì (theo code) |
|---|---|---|
| **client_transport** | client_ttft − ftl | net + HTTP + **render_chat + TOKENIZE** (trước arrival) + detok/stream token đầu. **Đo ~27ms, phẳng** → tokenize RẺ (~13ms cho 27k tok, Rust fast-tokenizer) |
| **frontend_prep** | ftl − queue − prefill | validate + sampling_build + IPC + **engine-core xử input TUẦN TỰ + HASH block**. **300-570ms, scale theo prompt** |
| **queue** | scheduled_ts − queued_ts | chờ FCFS trong scheduler |
| **prefill** | first_token_ts − scheduled_ts | chunked prefill (compute + interleave) |

> ⚠️ **ĐÍNH CHÍNH mô hình trực giác:** `render_chat + tokenize` KHÔNG nằm trong frontend_prep
> mà nằm trong **client_transport** (vì stamp `arrival_time` ở `input_processor.py:270`, SAU
> render ở `serving.py:259`). Đo được: chúng chỉ ~27ms (rẻ). **frontend_prep (35% TTFT) =
> engine-core xử input đơn luồng, chi phối bởi HASH block + validate, KHUẾCH ĐẠI bởi burst 20
> request nối đuôi.** Đây là lý do CHỨNG MINH BẰNG A/B: `api-server-count` (parallel HTTP) và
> `renderer-num-workers` (parallel tokenize) đều KHÔNG giảm frontend_prep — vì nút thắt là
> engine-core đơn luồng, không phải HTTP/tokenize.

### TPOT (mỗi token decode)
```
TPOT = pure_decode_step (9ms: 82% GEMM fp8 + 12% GDN state + 1% attn)
     + mixed_penalty  (7% step bị gộp chung 1 prefill chunk → phình 9→67ms)
```
Nguồn: sched trace phân loại step `n_prefilling==0` (pure) vs `>0` (mixed).

---

## PHẦN 2 — Bản đồ cờ → LÁ + logic code (chỉ cờ có nghĩa cho Track 3)

Ký hiệu: ✅ đã dùng · 🧪 đáng test · ❌ vỡ/no-op cho model này (source-verified) · ⚙️ mặc định giữ.

### A. Chạm RESIDUAL / frontend_prep (hash block, input serial)

| Cờ | default | Logic code | Verdict |
|---|---|---|---|
| `--prefix-caching-hash-algo` | sha256 | `core.py:208 get_hash_fn_by_name()` → hàm băm block. xxhash nhanh hơn ~10× | ❌ **xxhash VỠ preprocessing** (core.py:1566 error) cho Qwen3.5 |
| `--block-size` | None→hybrid ép | token/block KV; prefix-cache & hash theo block. To hơn → **ít block → ít hash op** | 🧪 lever gián tiếp giảm hash |
| `--renderer-num-workers` | 1 | `hf.py:814 maybe_make_thread_pool(tok, N+1)` — parallel tokenize (nhưng tokenize ở client_transport) | ❌ A/B: không giảm frontend_prep (residual 296→331). Cần kèm `--mm-processor-cache-gb=0` |
| `--api-server-count` | None | nhiều process frontend HTTP | ❌ A/B: frontend_prep 274→321 (HẠI, vì engine-core vẫn đơn luồng) |
| `--enable-prompt-tokens-details` | False | nhét `cached_tokens` vào usage | ⚙️ off (bớt tính) |
| `--disable-log-stats`, `--disable-uvicorn-access-log`, `--enable-log-requests` | — | tắt log → nhẹ CPU path | ⚙️ tắt lúc chấm |
| `--stream-interval` | 1 | flush SSE mỗi N token | ⚙️ giữ 1 (TTFT/client_transport) |

### B. Chạm QUEUE (throughput prefill phía trước)

| Cờ | default | Logic code | Verdict |
|---|---|---|---|
| `--max-num-batched-tokens` | None | trần token/iteration; `scheduler.py:392 num_new_tokens=min(_,budget)`. Chunk prefill = `floor((budget−decode)/block)×block` | ✅ 3216 (2-block, đỉnh chữ U) |
| `--max-num-seqs` | None | `scheduler.py` số running tối đa; nạn nhân preempt = `running.pop()` (mới nhất) | ✅ lever TPOT thật (hạ→tpot↓); 10-20 |
| `--mamba-cache-mode` | **none** | `qwen3_5.py:459`: **`all` RAISE NotImplementedError** → chỉ `align`/`none`. align = snapshot GDN state để cache | ❌ A/B: `align` prefill PHẲNG (base245↔align245 round2) — không cắt vì prefill 72% GEMM, GDN chỉ 12% |
| `--gdn-prefill-backend` | None(auto) | `gdn_attn.py:99 _resolve_gdn_prefill_backend()`; kernel 18 layer GDN | ✅ flashinfer |
| `--enable-chunked-prefill` | True | `scheduler.py:662`; cắt prefill xen decode | ⚙️ giữ bật |
| `--long-prefill-token-threshold` | 0 | `scheduler.py:390 if 0<thr<num_new_tokens: num_new_tokens=thr` — **CAP chunk prefill/request** | 🧪 knob TTFT↔TPOT: nhỏ→chunk nhỏ→ít cướp step decode (TPOT↑) nhưng nhiều iter (queue↑) |
| `--scheduler-reserve-full-isl` | True | `scheduler.py:730 full_sequence_must_fit=` — new prefill chỉ vào nếu đủ block cả ISL | 🧪 tắt = admit sớm (queue↓) đổi rủi ro preempt |
| `--kv-sharing-fast-prefill` | False | grep: **chỉ gemma3n/gemma4 implement** | ❌ NO-OP cho Qwen |
| `--async-scheduling` | None | overlap schedule↔compute | 🧪 nhẹ sched_ovhd; A/B cũ tbt−1ms |
| `--max-num-partial-prefills` | 1 | **0 USE trong V1 scheduler** | ❌ no-op V1 |
| `--scheduling-policy` | fcfs | fcfs vs priority (chọn nạn nhân preempt) | ⚙️ fcfs (trace có timestamp) |
| `--disable-hybrid-kv-cache-manager` | None | `kv_cache_utils.py:1631`; tắt trình quản KV lai (attn+mamba chung) | 🧪 A/B nếu nghi bug hybrid cache |

### C. Chạm PREFILL / TPOT (compute)

| Cờ | default | Logic code | Verdict |
|---|---|---|---|
| `--quantization` | None | GEMM weight fp8 (72% prefill, 82% decode) | ✅ fp8 — đòn bẩy compute |
| `--kv-cache-dtype` | auto | fp8 KV: nửa VRAM (BẮT BUỘC fit 18GB) nhưng ép block-align 1072 → chunk vào vùng chữ U xấu nếu batch không tune | ✅ fp8 (bắt buộc); phải kèm batch3216 |
| `--calculate-kv-scales` | False | scale fp8 KV lúc warmup | ⚙️ trơ với GDN (force scale=1.0) — bỏ được |
| `--attention-backend` | None | kernel 6 layer full-attn | ✅ FLASH_ATTN |
| `--disable-cascade-attn` | **True** | `gpu_model_runner.py:482 cascade_attn_enabled = not disable`. Cascade attn = tối ưu khi nhiều req share prefix DÀI | 🧪 trace share prefix 100% → `--no-disable-cascade-attn` đáng thử |
| `--enforce-eager` | False | tắt CUDA graph. Giữ False = graph BẬT (bảo vệ TPOT) | ⚙️ giữ False |
| `--cudagraph-capture-sizes` | None | batch-size chụp graph | ⚙️ phủ 20 |
| `--linear-backend` | auto | kernel GEMM (marlin/cutlass/machete) | ⚙️ auto |
| `--enable-flashinfer-autotune` | None | dò kernel flashinfer lúc warmup | 🧪 nhặt tốc độ, warmup lâu |
| `--spec-method=qwen3_5_mtp` | None | MTP head sinh nhiều token/step | ⚪ tpot đã kịch trần → không thêm điểm |
| `--mamba-cache-dtype`, `--mamba-ssm-cache-dtype` | auto | dtype GDN state | ⚪ state nhỏ, lợi ít |

### D. Startup / hạ tầng (không đụng điểm nhưng cần đúng)

| Cờ | Verdict |
|---|---|
| `--max-model-len` | ✅ 48000 (đủ 27.6k prompt); hạ ~32768 nới KV pool |
| `--gpu-memory-utilization` | ✅ 0.37 (mimic 18GB MIG) — **cố định để KV pool ổn định** |
| `--language-model-only` | ✅ bỏ vision tower (rẻ) |
| `--mm-processor-cache-gb=0` | ⚙️ **bắt buộc nếu renderer-num-workers>1** (cache mm không thread-safe) |
| `--safetensors-load-strategy=prefetch` | 🧪 rút startup nếu network FS |

---

## PHẦN 3 — Kết luận thực nghiệm (A/B đo thật, cùng session L40S 2026-07-14)

**TTFT ~1040ms bóc ra:** queue ~45% + prefill ~24% + residual ~31% (trong đó frontend_prep ~93% residual, scale theo prompt, dominant round 5).

**Đã LOẠI bằng A/B fully-instrumented (không lever nào thắng):**
- `api-server-count=8`: frontend_prep 274→321 (hại). Engine-core đơn luồng.
- `renderer-num-workers=8`: residual 296→331 (không giúp). Tokenize ở client_transport, không phải frontend_prep.
- `mamba-cache-mode=align`: prefill phẳng (GDN chỉ 12% prefill, GEMM 72% không cache được).
- `prefix-caching-hash-algo=xxhash`: VỠ preprocessing.
- `kv-sharing-fast-prefill`: no-op (gemma-only).

**Còn đáng test (chưa chạy):** `--no-disable-cascade-attn` (prefill full-attn), `--long-prefill-token-threshold` (TTFT↔TPOT), `--block-size` lớn (giảm hash op → frontend_prep), `--scheduler-reserve-full-isl=False` (admit sớm → queue).

**Ý nghĩa chiến lược:** frontend_prep (35% TTFT) là engine-core serial input — **không có cờ đơn giản parallel hóa được** → TTFT có sàn cứng → củng cố chiến lược seqs thấp (tpot-max, ~51 điểm) là gần trần, trừ khi giảm được hash/burst-serialization.
