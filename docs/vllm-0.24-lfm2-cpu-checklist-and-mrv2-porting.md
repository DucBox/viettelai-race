# vLLM 0.24 + LFM2 CPU-bound Checklist va MRV2 -> V1 Porting

Scope:
- vLLM `0.24`
- model `LiquidAI/LFM2.5-1.2B-Instruct`
- bai toan CPU-bound: MIG H200 `18GB VRAM`, `3 CPU cores`, `8GB RAM`
- uu tien TTFT/TPOT, chap nhan trade-off accuracy trong gioi han bai thi

## 0. Guardrail anti-cheat

Chi lam cac toi uu serving / scheduling / kernel / tokenizer / allocator / OS pinning.
Khong lam:
- precompute dap an
- branch khac nhau giua latency-run va grading-run
- heuristic dua vao request ID / trace order / phase cua he thong cham
- cache output mang tinh "dap an"

## 1. Bat ngay: env / package / OS setup

### 1.1 CPU thread hygiene

Bat:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

Ly do:
- vLLM da chu dong ep `OMP_NUM_THREADS=1` neu khong set de giam CPU contention tren GPU serving path; xem [multiproc_executor.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/v1/executor/multiproc_executor.py:1051).
- Voi may chi co `3` core, de BLAS/OpenMP tu spawn them thread thuong chi lam xau scheduling.

### 1.2 Rust frontend

Bat:

```bash
export VLLM_USE_RUST_FRONTEND=1
export TOKIO_WORKER_THREADS=1
```

Ghi chu:
- `VLLM_USE_RUST_FRONTEND` la env chinh thuc; xem [envs.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/envs.py:1279).
- Rust frontend tu dung `mimalloc`; xem [main.rs](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/rust/src/cmd/src/main.rs:14).
- Neu khong set `TOKIO_WORKER_THREADS`, runtime tu suy ra theo so CPU; xem [main.rs](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/rust/src/cmd/src/main.rs:20).

Khuyen nghi:
- Voi `3` core, bat dau tu `TOKIO_WORKER_THREADS=1`.
- Chi A/B `1 vs 2`; dung de cao hon.

### 1.3 fast tokenizer backend

Bat:

```bash
export VLLM_USE_FASTOKENS=1
```

Can package:
- `fastokens>=0.2.0`

Ly do:
- vLLM docs noi ro `VLLM_USE_FASTOKENS=1` tang toc encode/decode va streaming detokenization voi HF fast tokenizer; xem [optimization.md](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/docs/configuration/optimization.md:273).

Ghi chu:
- Co tac dung ro hon khi bottleneck nam o tokenizer / detokenize CPU.
- Neu model khong di qua HF fast tokenizer thi co the bi ignore.

### 1.4 Telemetry / tracking / logging level

Bat:

```bash
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export VLLM_LOGGING_LEVEL=ERROR
```

Ly do:
- `VLLM_NO_USAGE_STATS` va `VLLM_DO_NOT_TRACK` la env chinh thuc; xem [envs.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/envs.py:749).
- `VLLM_LOGGING_LEVEL` la env chinh thuc; xem [envs.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/envs.py:763).

### 1.5 FlashInfer sampler

Co the set ro rang:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=1
```

Nhung can hieu dung:
- Day da la default `True` neu ho tro; xem [envs.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/envs.py:793).
- Chi dung duoc tren CUDA va compute capability hop le; xem [topk_topp_sampler.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/v1/sample/ops/topk_topp_sampler.py:21).
- Sampler se tu fallback neu gap case greedy / explicit seed / processed logprobs; xem [gpu sample sampler.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/v1/worker/gpu/sample/sampler.py:220).

Khuyen nghi:
- Giu `=1` cho ro.
- Dung ky vong day la "game changer"; no la micro-optimization cho sampling path.

### 1.6 Thread pinning / CPU affinity

Can package:
- `taskset` tu `util-linux`

Khuyen nghi:

```bash
taskset -c 0-2 ...
```

Ly do:
- Khong phai flag cua vLLM, nhung rat hop voi bai toan `3` core de giam context switching va giu locality.
- Day la OS-level tuning an toan, khong lien quan anti-cheat.

### 1.7 TCMalloc

Can package:
- `libtcmalloc-minimal4` hoac goi tuong duong cua gperftools

Nguon tham chieu:
- vLLM setup khuyen nghi cai tcmalloc "for best performance"; xem [setup.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/setup.py:167).
- CPU docs cua vLLM huong dan them vao `LD_PRELOAD`; xem [cpu.x86.inc.md](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/docs/getting_started/installation/cpu.x86.inc.md:35).
- CPU platform cua vLLM tu preload tcmalloc de giam memory allocation overhead; xem [cpu.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/platforms/cpu.py:283).

Khuyen nghi cho bai nay:
- Day khong phai lever "official CUDA serving" cua vLLM.
- Van dang test duoc vi workload cua ban CPU-bound rat ro.
- Xep vao nhom `nen A/B`, khong xep vao nhom `bat buoc`.

Vi du:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:${LD_PRELOAD}
```

## 2. Bat ngay: server flags / scheduler flags

### 2.1 Logging va access log

Bat:

```bash
--disable-log-stats
--disable-uvicorn-access-log
```

Khong bat:

```bash
--enable-log-requests
```

Tham chieu:
- `--disable-log-stats`: [arg_utils.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/engine/arg_utils.py:1540)
- `--enable-log-requests`: default `False`; [arg_utils.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/engine/arg_utils.py:2660)
- `--disable-uvicorn-access-log`: [cli_args.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/entrypoints/openai/cli_args.py:245), [api_server.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/entrypoints/openai/api_server.py:589)

### 2.2 Async scheduling

Nen A/B:

```bash
--async-scheduling
```

Ly do:
- La lever that su cho scheduling overhead; xem [arg_utils.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/engine/arg_utils.py:1448).
- Voi case CPU-bound, day la knob dang thu.

Khuyen nghi:
- Giu no trong shortlist benchmark.
- Khong mac dinh khang dinh la se thang; phai do.

### 2.3 Batching knobs

Bat / retune:

```bash
--max-num-seqs=...
--max-num-batched-tokens=...
--enable-chunked-prefill
```

Tham chieu:
- [arg_utils.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/engine/arg_utils.py:1392)

Y nghia:
- `max-num-seqs`: lever quan trong cho decode concurrency va TPOT.
- `max-num-batched-tokens`: lever quan trong cho prefill / TTFT / scheduling budget.
- `enable-chunked-prefill`: nen giu trong bai toan nhieu request va context dai.

### 2.4 Prefix caching cho LFM2

Nen A/B co kiem soat:

```bash
--enable-prefix-caching
--mamba-cache-mode=align
```

Khong dung:

```bash
--mamba-cache-mode=all
```

Vi:
- Hybrid prefix caching van bi danh dau experimental; xem [model.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/config/model.py:1832).
- Default cua `enable_prefix_caching` se khong tu bat cho hybrid; xem [arg_utils.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/engine/arg_utils.py:2492).
- `LFM2` hard-reject `mamba_cache_mode=all`; xem [lfm2.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/model_executor/models/lfm2.py:484).

Ghi chu quan trong:
- Voi `LFM2`, path nay hop le nhung van la `experimental`.
- Uu diem cua `LFM2` la effective block size mac dinh van nho (`16`), nen prefix reuse co cua hon so voi Qwen3.5.

### 2.5 KV cache / memory / block sizing

Can retune:

```bash
--kv-cache-dtype=fp8
--gpu-memory-utilization=...
```

Chi thu khi can:

```bash
--block-size=...
--mamba-block-size=...
--kv-cache-memory-bytes=...
```

Tham chieu:
- cache flags: [arg_utils.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/engine/arg_utils.py:1140)
- hybrid auto-alignment: [interface.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/platforms/interface.py:721)

Khuyen nghi:
- Dung `fp8` cho model/KV neu benchmark cua ban da xac nhan tot.
- De `block-size` auto tru khi ban dang co ly do rat cu the de A/B.
- Voi `LFM2`, `mamba-block-size` khong phai lever uu tien cao.

### 2.6 Compilation / cudagraph knobs

Thu sau khi scheduler da on:

```bash
--compilation-config='...'
--optimization-level=...
--performance-mode=...
```

Tham chieu:
- [arg_utils.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/engine/arg_utils.py:1518)

Khuyen nghi:
- Day la nhom `secondary knobs`.
- Voi bai toan cua ban, no khong dung truoc `max-num-seqs`, `max-num-batched-tokens`, `fastokens`, Rust frontend, prefix caching.

## 3. Khong phai lever chinh hoac nen de tat

### 3.1 `--renderer-num-workers`

Khuyen nghi:
- Doi voi `LFM2` text-only, khong xem day la lever uu tien.
- De `1` tru khi ban co ly do ro rang khac.

Tham chieu:
- flag ton tai trong model args: [arg_utils.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/engine/arg_utils.py:875)
- rang buoc thread-safety nam o multimodal renderer cache: [model.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/config/model.py:703)

Nhan dinh:
- Day la suy luan tu source va ten flag: no khong phai knob trung tam cho text-only serving.

### 3.2 `--stream-interval`

Khuyen nghi:
- Giu `1` neu metric nhin theo thoi diem client nhan token.
- Chi tang len neu ban do duoc ro rang harness khong bi phat do buffered streaming.

### 3.3 `--kv-sharing-fast-prefill`

Khuyen nghi:
- De tat.

Vi:
- vLLM tu canh bao flag nay can model-side changes de dung va de co saving that su; xem [vllm.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/config/vllm.py:1313).

### 3.4 `--disable-hybrid-kv-cache-manager`

Khuyen nghi:
- De mac dinh, khong uu tien dung cho `LFM2`.

Vi:
- Day la debug / compatibility style flag hon la latency lever chinh.

### 3.5 Speculative decode / ngram / MTP

Khuyen nghi:
- Khong xep vao shortlist dau tay cho `LFM2` tren ha tang nay.

Ly do:
- Ban da co ket qua thuc te cho thay speculative co regression.
- Voi `3` CPU cores, overhead bo sung cua proposer / verifier / state handling rat de an mat loi ich.

Neu van muon test:
- n-gram can `numba`; xem `requirements/cuda.txt` ghi ro "Required for N-gram speculative decoding".

## 4. Cau hinh baseline goi y de benchmark

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VLLM_USE_FASTOKENS=1
export VLLM_USE_RUST_FRONTEND=1
export TOKIO_WORKER_THREADS=1
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_USE_FLASHINFER_SAMPLER=1

taskset -c 0-2 vllm serve ... \
  --disable-log-stats \
  --disable-uvicorn-access-log \
  --async-scheduling \
  --enable-chunked-prefill \
  --max-num-seqs=... \
  --max-num-batched-tokens=... \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=... \
  --enable-prefix-caching \
  --mamba-cache-mode=align
```

Ghi chu:
- `--enable-prefix-caching` + `align` la optional A/B vi hybrid van experimental.
- Neu run nay xau hon, rollback prefix caching truoc khi rollback scheduler knobs.

## 5. MRV2 -> V1: backlog porting co ROI cao nhat

Muc tieu chung:
- Giu `V1 runner` de khong mat `align` path cua hybrid.
- Port cac y tuong `GPU-first` cua MRV2 sang nhung diem V1 dang CPU-heavy.

### P0. Port persistent request-state kieu V2

Nguon:
- V2 `RequestState`: [states.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/v1/worker/gpu/states.py:9)
- V1 `CachedRequestState`: [gpu_input_batch.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/v1/worker/gpu_input_batch.py:34)

Nen port:
- fixed request slots
- `req_id -> slot index`
- `all_token_ids` dang `UVA` / staged write thay vi list Python + copy lai nhieu lan
- `num_computed_tokens`, `prompt_len`, `prefill_len` giu dang tensor state lau dai

Dong co:
- giam CPU object churn
- giam copy / reorder state moi step
- giam Python overhead o nhung doan bi lap lai cho moi request

### P0. Port `UvaBackedTensor` / `StagedWriteTensor` / `UvaBufferPool`

Nguon:
- [buffer_utils.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/v1/worker/gpu/buffer_utils.py:44)

Nen port vao V1 cho:
- block tables
- all token ids
- per-request scalar states
- bat ky tensor lon ma CPU update incrementally

Dong co:
- thay vi rebuild + H2D copy ca khoi, chi stage diff va apply diff
- tan dung pinned/UVA path
- rat hop voi bai toan CPU-bound

### P0. Port `BlockTables` moi

Nguon:
- [block_table.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/v1/worker/gpu/block_table.py:17)

Nen port:
- `StagedWriteTensor` cho block tables
- `FusedStagedWriter` cho multi-group path
- `gather_block_tables()` kernel
- `compute_slot_mappings()` kernel

Dong co:
- block-table va slot-mapping la vung nong cua serving hybrid/paged attention
- day la mot trong nhung noi MRV2 day viec len GPU rat manh

### P0. Port pipeline chuan bi input tren GPU

Nguon:
- V2 runner: [model_runner.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/v1/worker/gpu/model_runner.py:844)

Nen port:
- `prepare_prefill_inputs`
- `prepare_pos_seq_lens`
- `expand_idx_mapping`
- `combine_sampled_and_draft_tokens`

Dong co:
- Day la phan "giai phong CPU" ro nhat cua MRV2.
- V1 con ganh nhieu viec gom / xep / chuyen tensor bang CPU va Python.

### P1. Port sampling state kieu V2

Nguon:
- [gpu sample sampler.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/v1/worker/gpu/sample/sampler.py:30)

Nen port:
- sampling states dang `UVA` / staged write
- top-k/top-p + bad-words + logit-bias state theo request slot
- tach prompt logprobs / output logprobs theo worker rieng neu can

Dong co:
- ban khong focus logprobs, nen co the cat gon path nay va uu tien sample path "lean"
- giup giam state mutation bang Python moi step

### P1. Port `PromptLogprobsWorker` / logprobs path thanh optional path tach biet

Ly do:
- Ban uu tien TTFT/TPOT, khong quan tam logprobs.
- Nen bien logprobs thanh slow-path that su, tranh de no lam nong path mac dinh.

### P1. Port cach chia file / helper giong V2 de mo complexity ra khoi hot path

Nguon:
- V2 runner co hot path toi gian va day utility sang file rieng; xem [model_runner.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/v1/worker/gpu/model_runner.py:3)

Dong co:
- V1 `gpu_model_runner.py` rat lon va qua nhieu branch.
- Tach branch CPU-heavy / rare-feature ra khoi hot path thuong giup de toi uu hon.

## 6. Khong nen lam som trong MRV2 -> V1

### 6.1 Khong bat dau bang full MRV2 cho LFM2

Vi:
- Worker van chon giua V1/V2; xem [gpu_worker.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/v1/worker/gpu_worker.py:354).
- `align` mode chua duoc support tren MRV2; xem [vllm.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/config/vllm.py:2140).

### 6.2 Khong bat dau bang "port align support sang MRV2"

Vi:
- Cai nay khong chi la bo 1 guard.
- Phai ghep them hybrid metadata/state/recompute pipeline cho `align`.
- Trong MRV2, `MambaHybridModelState` moi chi giu metadata chung; xem [mamba_hybrid.py](/Users/ngoquangduc/Desktop/workspace/viettelai-race/vendor/vllm-0.24.0/vllm/v1/worker/gpu/model_states/mamba_hybrid.py:57).

### 6.3 Khong bat dau bang speculative path

Vi:
- Khong dung pain point chinh hien tai.
- Them overhead CPU/branching trong khi bai toan dang ket o scheduling + input prep.

## 7. Thu tu uu tien that te

### Uu tien A: khong sua source

1. `OMP/MKL/OPENBLAS=1`
2. `VLLM_USE_FASTOKENS=1`
3. `VLLM_USE_RUST_FRONTEND=1`, bat dau `TOKIO_WORKER_THREADS=1`
4. `--disable-log-stats`, `--disable-uvicorn-access-log`
5. `--async-scheduling`
6. retune `--max-num-seqs`, `--max-num-batched-tokens`
7. `--enable-prefix-caching --mamba-cache-mode=align` cho LFM2
8. thu `taskset`
9. thu `LD_PRELOAD=libtcmalloc...`

### Uu tien B: sua source nhe, ROI cao

1. Port `UvaBackedTensor` / `StagedWriteTensor`
2. Port `BlockTables`
3. Port GPU input prep kernels
4. Port fixed-slot request state

### Uu tien C: de sau

1. sampler/logprobs path clean-up
2. compile / cudagraph micro-tuning
3. nghien cuu phan hybrid align rong hon

## 8. Chot nhanh

Neu can mot checklist cuc ngan de bat tay ngay, thi shortlist la:

- Bat: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- Bat: `VLLM_USE_FASTOKENS=1`
- Bat: `VLLM_USE_RUST_FRONTEND=1`, thu `TOKIO_WORKER_THREADS=1`
- Bat: `VLLM_NO_USAGE_STATS=1`, `VLLM_DO_NOT_TRACK=1`, `VLLM_LOGGING_LEVEL=ERROR`
- Bat: `--disable-log-stats`, `--disable-uvicorn-access-log`
- Thu: `--async-scheduling`
- Retune: `--max-num-seqs`, `--max-num-batched-tokens`
- Thu: `--enable-prefix-caching --mamba-cache-mode=align`
- Thu: `taskset -c 0-2`
- Thu sau cung: `LD_PRELOAD=libtcmalloc...`

Neu can shortlist porting:

- Port `UvaBackedTensor/StagedWriteTensor`
- Port `BlockTables`
- Port `prepare_prefill_inputs` + `prepare_pos_seq_lens`
- Port fixed-slot `RequestState`

