# Bench toolkit + các TẦNG metric (bóc từ bề nổi xuống sâu)

Toàn bộ script ở `scripts/gpu-l40s-bench/`. Trên GPU mới, copy cả thư mục lên
`/root/` (xem [gpu-setup-checklist.md](gpu-setup-checklist.md)). Model tải bằng
`hf download Qwen/Qwen3.5-2B` (hash verify trong memory), KHÔNG scp.

---

## 1. Pipeline đo (1 rep)

```
serve (fp8 config)  →  warmup_stub.py (1-2 req ảo, ổn định)  →  wait health=200
   →  replay_trace_detailed.py (bắn 120 req đúng timestamp, sampler GPU/CPU 0.5s)
   →  merge_request_metrics.py (join client × server REQSTAT log)
   →  ers_and_stats.py (ERS + percentile + per-turn + system)
```
Driver A/B (`run_ab_*.sh`) tự lặp interleaved, cold mỗi rep, median-of-3.

**BẮT BUỘC trước khi serve:** `patch_loggers.py apply` (bật REQSTAT per-request).
Muốn đo tầng sâu nhất: thêm `patch_sched_trace.py apply` + `SCHED_TRACE=<file>`.

---

## 2. Bảng file — làm gì

**Serve / driver**
| file | vai trò |
|---|---|
| `serve_v12.sh` | baseline chuẩn = config submit v12 (fp8+fp8KV+flashinfer). Gốc mọi A/B |
| `run_ab_fp8.sh` | A/B noquant vs fp8 |
| `run_ab_maxseqs.sh` / `run_ab_seqs2.sh` | A/B quét `--max-num-seqs` |
| `run_ab_batchtok.sh` / `run_ab_batchtok2.sh` | A/B quét `--max-num-batched-tokens` |
| `run_ab_calc.sh` | A/B có/không `--calculate-kv-scales` |
| `run_sched_diag.sh` / `run_sched_batchtok.sh` | serve + bật SCHED_TRACE để đo tầng iteration |

**Patch runtime (sửa file vLLM đã cài, có apply/revert)**
| file | vai trò |
|---|---|
| `patch_loggers.py` | emit **REQSTAT** mỗi request: queued/prefill/decode_time, mean_tpot, num_cached_tokens. **Nền tảng tầng 2** |
| `patch_sched_trace.py` | emit mỗi **iteration**: n_running, n_prefilling, tokens_sched, exec_gap_ms. **Nền tảng tầng 3-4**. Gate bằng env `SCHED_TRACE` |
| `patch_serving_priority.py` | patch SPF (priority=len prompt). Đã chứng minh vô dụng, để tham khảo |

**Replay / đo**
| file | vai trò |
|---|---|
| `replay_trace_detailed.py` | công cụ đo chính. 120 req đúng timestamp + sampler GPU/CPU 0.5s. Ghi `replay_detailed_requests.json` + `replay_detailed_samples.json` |
| `merge_request_metrics.py` | join client↔server REQSTAT theo `server_request_id` → `*_full.json` |
| `warmup_stub.py` | 1-2 req ảo tự sinh (KHÔNG lấy trace) để ổn định cold-start |

**Phân tích (chạy LOCAL trên `*_full.json` / `sched_*.jsonl`)**
| file | tầng | vai trò |
|---|---|---|
| `ers_and_stats.py` | 0-2 | ERS thật + percentile queue/prefill/decode/ttft/tpot + per-turn + GPU/CPU |
| `analyze_sched.py` | 3 | step time (prefill vs decode), tokens_sched, admission, step-time vs n_running |
| `analyze_prefill.py` | 4 | chunk prefill thực mỗi step, histogram số block |
| `analyze_rule.py` | 4 | rút RULE `chunk = floor((budget−decode)/1072)×1072`, kiểm khớp % |

---

## 3. CÁC TẦNG METRIC — bóc từ bề nổi xuống sâu

> **Nguyên tắc:** score là bề nổi nhất; mỗi tầng xuống trả lời "TẠI SAO tầng trên như vậy".
> queue_time / prefill_time (tầng 2) MỚI CHỈ LÀ BỀ NỔI — bóc tiếp xuống tầng 3-4.

### Tầng 0 — ĐIỂM (cái BTC chấm)
`ers_and_stats.py` / `08_ers_score.py`
```
ERS = mean 120 req của 0.5·s_ttft + 0.5·s_tpot
s = clamp((C−x)/(C−F),0,1)²   TTFT: F=100 C=1500 | TPOT: F=20 C=45
```
Đọc: `score`, `s_ttft`, `s_tpot`. → biết THUA Ở TRỤC NÀO (ttft hay tpot).
⚠️ Trên L40S tpot<20ms (dưới floor) → s_tpot~1.0 **mù trục tpot**. Phải đọc **raw tpot (ms)**, không đọc s_tpot.

### Tầng 1 — per-request TTFT/TPOT (client-observed)
`replay_detailed_requests.json`: mỗi (user_id, turn_index) có `ttft_ms`,
`client_mean_chunk_gap_ms` (=TPOT). → request nào chậm, turn nào chậm.

### Tầng 2 — TÁCH TTFT = queue + prefill (server-side)  ← "bề nổi" bạn nhắc
`*_full.json` (từ REQSTAT): `server_queue_ms`, `server_prefill_ms`,
`server_decode_ms`, `num_cached_tokens`, `num_prompt_tokens`.
- **TTFT = queue + prefill.** queue = chờ trước khi được schedule; prefill = compute.
- Đọc bằng `ers_and_stats.py`: mean/median/**p90/p95/p99/max** + **per-turn**.
- num_cached_tokens giải thích turn 2-5 prefill thấp (prefix cache hit).
- **Nhưng queue/prefill CHƯA phải đáy** — nó là hệ quả của tầng 3.

### Tầng 3 — PER-ITERATION scheduler  ← chọc sâu 1
`patch_sched_trace.py` + `analyze_sched.py`. Mỗi step ghi:
- `n_running` / `n_prefilling` → **bao nhiêu seq decode vs prefill trong 1 iteration**
- `tokens_sched` → so với budget (`max_num_batched_tokens`) → **nghẽn budget?**
- `exec_gap_ms` → **thời gian wall-clock 1 iteration** (step prefill vs decode-only)
- `n_new_admit`, `n_waiting` → tốc độ rút hàng đợi
→ Trả lời: **queue dài = SỐ iteration nhiều hay iteration lâu?** decode step vs
prefill step chênh bao nhiêu? (đo được: decode-only ~9-13ms, mixed-prefill ~50-170ms).

### Tầng 4 — BLOCK-TILING (đáy scheduler)  ← chọc sâu 2
`analyze_prefill.py` + `analyze_rule.py`. Bóc chunk prefill thực = `tokens_sched − decode`:
```
RULE:  chunk_prefill = floor((budget − decode_tokens) / block) × block
block (fp8 KV) = 1072 tokens   (độc lập GPU; từ mamba/attention page align)
```
→ Trả lời TẬN GỐC (phía scheduler): tại sao prefill xả nhanh/chậm. Vd budget 3216,
11 decode → floor((3216−11)/1072)=2 → chunk = 2144 (2 block).

### Tầng 5 — PER-OP GPU PROFILER (đáy kernel)  ← chọc sâu NHẤT
`profile_config.py` (offline LLM + torch profiler, ĐÚNG kernel serving) → `parse_dir.py`.
Profile **PREFILL** (1 seq ~8k tok) + **DECODE** (batch N × 40 tok) riêng, gộp GPU-kernel
theo nhóm: GEMM / GDN conv / GDN scan-state / Attention / KV-write / copy.
```
python3 profile_config.py --tag v23 --seqs 20 --batch 3216 --decode-batch 20
python3 parse_dir.py /root/prof      # in bảng PREFILL + DECODE
```
→ Trả lời TẬN KERNEL: **thời gian mỗi ms đi đâu**, config nào cắt ở kernel nào.
Lưu ý: vLLM tạo profiler-wrapper 1 lần → 2 window ghi 2 file cùng prefix; parse theo
mtime (file sớm=prefill, muộn=decode). **Từ giờ mỗi A/B nên chạy kèm 1 profile-pass.**

---

## 4. RULE + phát hiện đã đúc kết (đừng để mất)

1. **block = 1072** (fp8 KV; 544 nếu bf16 KV). Chunk prefill = bội số của block.
2. **RULE chunk** = `floor((budget − decode)/1072)×1072`. Cần **margin** cho decode:
   budget khít N×1072 → 1 token decode là rớt xuống (N−1) block. b2144 vô dụng vì thế.
3. **Throughput prefill hình chữ U** (L40S): 1blk=21, **2blk=34 (sweet spot)**, 3blk=19 tok/ms.
   → budget ~3216 (2 block + margin) tối ưu QUEUE. (U-shape magnitude phụ thuộc GPU.)
4. **1 iteration: decode gần free (1 token/seq), prefill hog budget (cả chunk ~2144
   cho 1 seq)** → thường **1 seq prefill/step**, phần còn lại decode. Prefill bị
   serialize theo FCFS → đó CHÍNH LÀ queue.
5. **batch-tokens = lever QUEUE/TTFT, KHÔNG phải tpot** (v23: tbt 28 y nguyên,
   ttft_p95 5870→5155). raw tpot phẳng 12.6-12.8 khi quét batch 2048-6432.
6. **tpot lever = max-num-seqs**, nhưng chỉ kích hoạt **dưới ~20** (L40S: seqs20≈32
   tpot~12.7; seqs10 → 9.47, −26%). Cái giá: queue +155%, rủi ro failed_count.
7. **`--calculate-kv-scales` TRƠ** với model GDN (vLLM force-disable → scale 1.0).
8. **L40S mù trục tpot** (dưới floor) + không resolve được seqs 20 vs 32 → tune tpot
   phải trên H200. L40S chỉ đo được: raw-tpot tương đối, queue/ttft, failed_count.
9. **PER-OP (Tầng 5, profiler L40S):** prefill = ~72% GEMM fp8 + 12% attention + 12% GDN.
   **DECODE (=TPOT) = ~82% GEMM fp8** (matmul nhỏ nhưng memory-bound = ĐỌC WEIGHT) +
   12% GDN state + 1% attention. fp8 vs bf16: **decode −21% total, đến TỪ GEMM −24%**
   (weight bf16→fp8 nửa byte); GDN không đổi. → W4AFP8 (weight ¼ bf16) tấn công đúng
   82% đó, NHƯNG chỉ ~½ GEMM là weight-read (tách: bf16 weight-read 136ms, phần khác
   144ms) → W4AFP8 lợi thực **~13% decode trên fp8** (không phải 21%); ăn nhiều hơn ở
   batch thấp (weight ít amortize). KV-fp8 tiết kiệm memory chứ decode-attention chỉ 1%.

## 6. CÂY PHÂN RÃ CÓ KIỂM CHỨNG (reconcile TTFT + TPOT)

> Mục tiêu: không chỉ `TTFT = queue + prefill + residual` mà **bóc tiếp mỗi nhánh**
> và kiểm `|Σcon − cha| < ε` per-request. Phát hiện quan trọng: **residual ~35% TTFT**
> (server-side thật, verify bằng `server_e2e−queue−inference`), scale theo prompt,
> thống trị TTFT ở round 2-5. queue/prefill KHÔNG cứu được residual.

**3 script mới (viết offline, chạy khi có GPU):**
| file | vai trò |
|---|---|
| `patch_residual_ts.py` | patch `stats.py::update_from_finished_request` → ghi `rests.jsonl` mỗi request: `arrival_time, ftl (first_token_latency), queued_ts/scheduled_ts/first_token_ts/last_token_ts (monotonic core)`. Gate env `RESIDUAL_TRACE`. **Nền tảng tách residual.** |
| `patch_sched_trace.py` | (mở rộng) thêm `new_ids`/`pref_ids` per-iteration → correlate per-request queue & prefill-interleave |
| `reconcile_trace.py` | join `full.json + rests.jsonl + sched.jsonl` → in cây TTFT/TPOT đầy đủ + sai số reconcile mỗi tầng |

**Cây đạt được:**
```
TTFT = queue + prefill + residual
├─ queue    = wait_prefill + wait_decode + sched_ovhd      (sched window [queued_ts,scheduled_ts])
├─ prefill  = own_compute + interleave                     (window [scheduled_ts,first_token_ts]; own = pref_ids)
│            └─ own_compute → GEMM/attn/GDN/KV (Tầng 5 profiler)
└─ residual = frontend_prep + client_transport            (tách EXACT bằng ftl)
             ├─ frontend_prep    = ftl − queued_time − prefill_time   (tokenize+hash+handoff, ~scale prompt)
             └─ client_transport = client_ttft − ftl                  (net+HTTP+detok/stream)
TPOT = pure_decode_step + mixed_interleave_penalty         (phân loại step n_prefilling==0 vs >0)
```
**Clock:** `arrival_time`/`ftl` = frontend (time.time); `*_ts` = core (monotonic, CÙNG clock `t`
của sched trace → correlate được). `frontend_prep` hợp lệ vì cả `ftl` lẫn `queued_time+prefill_time`
kết thúc ở CÙNG biến cố (first token). `reconcile_trace.py` tự cảnh báo nếu window sched rỗng (clock lệch).

**Chạy (1 rep, bật cả 3 tầng sâu):**
```
patch_loggers.py apply; patch_residual_ts.py apply; patch_sched_trace.py apply
RESIDUAL_TRACE=$O/rests.jsonl SCHED_TRACE=$O/sched.jsonl <serve...>   # rồi replay như thường
merge_request_metrics.py ... --out $O/full.json
reconcile_trace.py --full $O/full.json --rests $O/rests.jsonl --sched $O/sched.jsonl --out $O/recon.json
```
⚠️ Còn hộp đen chưa tách: `frontend_prep` gộp tokenize+hash+handoff (muốn tách nữa cần thêm
timestamp tại API-server trước/sau tokenize); `own_compute` cần chạy kèm 1 profile-pass để đổ về kernel.

## 7. ENV ỔN ĐỊNH (điều kiện để A/B CÓ NGHĨA)

> Mọi rep phải chạy cùng 1 env make-sense, nếu không A/B vô nghĩa. **CPU giờ là
> hạng nhất** vì residual=tokenize CPU-bound (~35% TTFT) — chỉ khóa GPU là chưa đủ.

| file | vai trò |
|---|---|
| `env_setup.sh` | chạy 1 LẦN sau khi thuê: check GPU độc quyền, **khóa GPU clock** (best-effort), **CPU governor=performance**, **chia core server↔client** → xuất `$SRV_PIN`/`$CLI_PIN` (taskset). `source env_setup.sh` |
| `env_gate.py` | per-rep: đọc `*_samples.json` → FAIL nếu **cpu_throttle>1%** / **sm_clock tụt>5%** / neighbor ồn → driver chạy lại rep đó |

**Tích hợp driver** (mẫu ở `run_ab_fp8.sh`): `source /root/env_pins.sh`; serve = `$SRV_PIN <vllm>`;
replay = `$CLI_PIN <replay>`; sau replay gọi `env_gate.py` → rep FAIL thì loại khỏi median.

**Quy trình chuẩn khi thuê GPU:**
```
source env_setup.sh          # khóa clock + governor + core split (1 lần)
# rồi mọi run_ab_*.sh tự source env_pins.sh, pin core, gate mỗi rep
```
⚠️ Rented thường **cấm khóa GPU clock** → env_gate là chốt chặn (loại rep bị boost-drift).
Nếu GPU không khóa được clock và boost dao động rộng, nới `--sm-clock-drop 0.08`.
⚠️ Box **<6 core** → không tách được server/client → residual nhiễu → A/B TTFT kém tin.
**Nên thuê box ≥6 core, GPU độc quyền.** reconcile_trace cho biết nhiễu rơi vào nhánh nào
(residual/client_transport tăng = CPU hỏng; queue/prefill ổn) → chẩn đoán env ngay.

## 5. Điểm submit đã biết (results.csv)
- v12 (seqs32, default) = 29.71 | v23 (seqs32 + batch3216) = 29.92 (batch giúp TTFT tail)
- **v20 (seqs20 + batch6144) = 44.09** | v21 (seqs20 + batch8192) = 43.32
- → hạ seqs (trade TTFT lấy tpot) là cú nhảy lớn. Hướng tiếp: v24 seqs20+batch3216,
  v25 seqs~14 (canh failed_count trên H200).
