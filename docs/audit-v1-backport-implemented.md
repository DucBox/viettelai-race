# Audit — V1 backport đã implement (vLLM 0.24.0 + LFM2)

Ngày audit: 2026-07-18. Người audit: đọc trực tiếp source trong `vendor/vllm-0.24.0/`.

Scope: nhánh backport "GPU-first ideas của MRV2 → V1 runner" cho bài toán CPU-bound
(MIG H200 18GB / 3 CPU / 8GB RAM), model `LiquidAI/LFM2.5-1.2B-Instruct`, ưu tiên TTFT/TPOT,
chạy greedy để giữ accuracy gate.

Nguyên tắc thiết kế đã tuân thủ: **mọi thay đổi đều opt-in qua env flag, off mặc định giữ
nguyên hành vi stock**. Không có branch khác nhau giữa latency-run và grading-run.

---

## 1. Bảng tổng hợp trạng thái

| # | Env flag | File | Correctness | ROI (contest greedy) | Mặc định |
|---|----------|------|-------------|----------------------|----------|
| 1 | `VLLM_V1_BACKPORT_CLEAR_STALE_ALLOWED_MASK` | `gpu_input_batch.py` | ✅ vá bug thật | Trơ (greedy không dùng `allowed_token_ids`) | off |
| 2 | `VLLM_V1_BACKPORT_ASYNC_OUTPUT_REPAIR_OPT` | `gpu_input_batch.py` | ✅ đúng | Dương nhẹ (hot path async decode) | off |
| 3 | `VLLM_V1_BACKPORT_SKIP_NOOP_METADATA_REFRESH` | `gpu_model_runner.py` | ✅ đúng | Nhỏ | off |
| 4 | `VLLM_V1_BACKPORT_DEBUG` (+ `_DEBUG_DUMP_PATH`) | cả 2 file | ✅ zero-overhead khi off | Công cụ đo | off |
| 5 | `VLLM_V1_BACKPORT_ROW_STATE_CACHE` | `gpu_input_batch.py`, `gpu_model_runner.py` | ✅ invariant OK | **Âm** → giữ off | off |
| 6 | `VLLM_V1_BACKPORT_SKIP_REORDER_ONLY_METADATA_REBUILD` | `gpu_input_batch.py` | ✅ **đã kín sau fix** | **Dương** — đáng A/B | off |

Ngoài 6 cờ trên, còn 1 patch MRV2 riêng (mục 4) không thuộc nhánh flag.

---

## 2. Chi tiết từng hạng mục (đã verify source)

### #1 — Clear stale `allowed_token_ids` mask
- Vị trí: `gpu_input_batch.py`, nhánh `elif` trong `add_request`.
- Vấn đề gốc: `sampler.py` áp `allowed_token_ids_mask` cho **toàn bộ** row `[:num_reqs]`
  bằng `masked_fill_`, **không** gate theo từng request. Row cũ còn sót → mask nhầm request mới.
- Fix: request mới không dùng `allowed_token_ids` mà tái dùng slot → `fill_(False)` row đó.
- Đánh giá: đúng, vá bug correctness thật. **Trơ với contest** (greedy, không ai set
  `allowed_token_ids` → `allowed_token_ids_mask_cpu_tensor` mãi `None`, nhánh không chạy).

### #2 — Async output repair opt
- Vị trí: `update_async_output_token_ids` trong `gpu_input_batch.py`.
- Trước: sync event + `tolist()` rộng tay hơn.
- Sau: quét CPU-side để gom `rows_to_repair` trước → early-return nếu rỗng → chỉ
  `synchronize()` + convert **từng row** cần sửa.
- Đánh giá: đúng thứ tự (đọc placeholder CPU-side không cần sync; chỉ sync trước khi đọc
  dữ liệu GPU-copied). Nằm trên hot path async decode → tiết kiệm CPU thật.

### #3 — Skip no-op metadata refresh
- Vị trí: `_update_states()` trong `gpu_model_runner.py` (~L1595-1624).
- Logic: `should_refresh_metadata = has_pending_batch_update if flag else True`.
  Chỉ skip `refresh_metadata()` khi không có `batch_changed/removed/added/moved`.
- Đánh giá: đúng. Đã chứng minh: gate=False ⟹ `get_and_reset()` cũng trả `None`, nên
  không bao giờ bỏ sót rebuild cần thiết. Cờ off → luôn refresh (giữ hành vi stock).
- Lưu ý: phần đắt (`_make_sampling_metadata`) vốn đã sau `if batch_update` ở bản gốc, nên
  lời lãi của cờ này nhỏ.

### #4 — Debug instrumentation
- Vị trí: `_update_states` + `refresh_metadata` (timer/counter theo pha).
- Cờ: `VLLM_V1_BACKPORT_DEBUG=1`, dump ra `VLLM_V1_BACKPORT_DEBUG_DUMP_PATH`.
- **Zero-overhead khi off (đã verify):** mọi `time.perf_counter()` đều theo pattern
  `start = perf_counter() if self._backport_debug_enabled else None`, và mọi phép trừ delta
  đều bọc `if start is not None:`. Khi `DEBUG=0`: không một lời gọi `perf_counter` nào trong
  hot path, không có nguy cơ trừ `None`.
- Stats gồm: `update_states.{remove_finished,remove_unscheduled,add_request,condense,reorder,
  refresh_metadata,total}_seconds`, các counter req, `refresh_metadata_{rebuilds,skips,
  reorder_only_skips}`, async repair wait.

### #5 — Row-state cache (fixed-slot nhẹ)
- Vị trí: `req_states: list[CachedRequestState | None]` trong `InputBatch` + helper
  `get_cached_req_state` / `iter_cached_req_states`; đồng bộ ở `add_request` / `remove_request`
  (set `None`) / `swap_states`.
- Bug đã gặp & vá: batch drain về 0 rồi add lại làm `req_states` lệch `None`; đã fix bằng
  clear khi batch empty. Smoke test `check_row_state_cache_invariants` assert
  `iter_cached_req_states()` khớp `req_ids` qua add/swap/remove/drain.
- Đánh giá: invariant đúng, nhưng **benchmark thật ROI âm** → giữ off, coi là experimental.

### #6 — Skip reorder-only metadata rebuild  ⭐ cờ core dương duy nhất
- Vị trí: `_can_skip_sampling_metadata_rebuild` trong `gpu_input_batch.py` (~L921-950),
  gọi trong `refresh_metadata`.
- Ý tưởng: nếu `batch_update` chỉ có `moved` (reorder), không `added/removed`, và workload
  không dùng feature nào phụ thuộc rebuild → skip `_make_sampling_metadata()`.
- **Lỗ hổng đã phát hiện & đã vá:** `swap_row` chỉ swap tensor **CPU**
  (`temperature_cpu/top_p_cpu/top_k_cpu`), tensor **GPU** chỉ được cập nhật qua `copy_slice`
  bên trong `_make_sampling_metadata`. Guard ban đầu **thiếu** check greedy/top-p/top-k →
  với sampling khác greedy sẽ skip nhầm → GPU tensor giữ thứ tự cũ → **sai output**.
- Fix (đã áp): thêm điều kiện
  ```python
  if not (self.all_greedy and self.no_top_p and self.no_top_k):
      return False
  ```
- Đã đối chiếu đủ: các tensor per-row mà `_make_sampling_metadata` refresh chỉ gồm
  temperature / top_p / top_k / 3 penalties. Guard cover đủ (temp/top_p/top_k qua dòng trên,
  penalties qua `no_penalties`). `min_p` không tồn tại trong `gpu_input_batch.py` bản này.
- Đánh giá: **correctness đã kín**, an toàn A/B rộng kể cả sau này chạy sampling khác greedy.

---

## 3. Test & harness

### Smoke test — `scripts/vllm_v1_backport_smoke.py`
Chạy pass trên remote GPU (Python 3.12, import vLLM 0.24 thật). Các check:
- `check_stale_allowed_mask` (#1)
- `check_async_output_repair` (#2)
- `check_runner_refresh_behavior` (#3)
- `check_row_state_cache_invariants` (#5)
- `check_reorder_only_metadata_skip` (#6) — **đã nâng cấp thành test thật**:
  - greedy (temp=0, top_p=1, top_k=0) → assert **skip xảy ra**;
  - non-greedy, **giá trị khác nhau** (temp 0.7/1.3, top_p 0.91/0.73, top_k 17/9) → assert
    **rebuild không skip** + **check nội dung GPU tensor theo đúng thứ tự đã swap**
    (`temperature==[1.3,0.7]`, `top_p==[0.73,0.91]`, `top_k==[9,17]`);
  - penalty → rebuild không skip.
  - Test này **sẽ fail nếu bỏ dòng guard mới**, nên thật sự canh được regression.

### A/B harness — `scripts/vllm_v1_backport_ab.py`
Launch server theo từng config/env → replay trace → parse `REQSTAT` → xuất
`summary.csv/json` + breakdown queue/prefill/decode. Dùng để đo từng cờ trên GPU thật.

---

## 4. Patch MRV2 riêng (ngoài nhánh flag)

- File: `vendor/vllm-0.24.0/vllm/v1/worker/gpu/model_states/mamba_hybrid.py`.
- Bug upstream: `MambaHybridModelState` không forward `seq_lens_cpu_upper_bound` vào
  `build_attn_metadata` (trong khi `DefaultModelState` có) → `mamba_attn.py:_compute_common_metadata`
  assert `seq_lens_cpu is not None` → EngineCore chết lúc init khi `VLLM_USE_V2_MODEL_RUNNER=1`.
- Fix: thêm `seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound` vào lời gọi.
- Trạng thái chiến lược: cho phép MRV2-native boot với LFM2, **nhưng** đường này mất
  APC cross-turn (prefix caching off là điều kiện để tránh `align`) → nhiều khả năng thua
  V1+align+APC trên trace prefix-heavy. Chỉ dùng để A/B đối chứng, không phải hướng chính.

### Image delivery
- `submit_r2/Dockerfile.experiments`: đã chuyển sang **overlay `.py`** lên package cài sẵn
  (`/usr/local/lib/python3.12/dist-packages/vllm/...`) thay vì `pip install` từ source
  (vốn sẽ trigger compile CUDA và fail trong runtime image). Có assert build-time kiểm patch
  đã "ăn".

---

## 5. Chưa implement (backlog ROI lớn hơn)

- Giữ unscheduled request trong persistent batch mà không remove/re-add.
- Diff-update sâu cho metadata / block tables thay vì contract hiện tại.
- Deferred / cục bộ hóa mạnh hơn ở scheduler-state path.
- Port các phần MRV2 lớn hơn sang V1 (UVA/StagedWrite, BlockTables, GPU input-prep kernels).
- Full hybrid/align-aware runner rewrite.

---

## 6. Kết luận audit

- **Không còn vấn đề correctness tồn đọng** sau khi vá guard #6.
- Nhóm đã implement: 3 micro-opt/correctness (#1–3), 1 bộ instrumentation zero-overhead (#4),
  2 cờ thử nghiệm (#5 âm ROI, #6 dương ROI), smoke + A/B harness đo thật trên GPU.
- **Cờ đáng đẩy tiếp:** #6 (`SKIP_REORDER_ONLY_METADATA_REBUILD`) — dương và đã an toàn.
  #2 (`ASYNC_OUTPUT_REPAIR_OPT`) — dương nhẹ, đáng giữ trong shortlist A/B.
- **Giữ off:** #5 (ROI âm). #1 correctness tốt nhưng trơ với workload greedy.
- Bước tiếp hợp lý: chạy A/B harness bật riêng #6 (± #2) trên `data/trace_grading_public.jsonl`,
  đọc breakdown queue/prefill/decode + stats `DEBUG=1` để quyết định trước khi đụng backlog mục 5.
