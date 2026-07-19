# V1 -> V2 Adaptive Backport Plan

Ngày lập: 2026-07-18

## Mục tiêu

Biến legacy V1 path đang chạy thật với hybrid/mamba (`gpu_model_runner.py`,
`gpu_input_batch.py`) thành một runner mang tinh thần MRV2 đúng nghĩa:

- persistent batch cư trú ổn định hơn
- giảm remove/add/condense/reorder churn
- delta update thay vì rebuild rộng tay
- đẩy nhiều state/input-prep sang GPU/UVA hơn
- giảm CPU sync/copy trong async path
- mọi thay đổi đều `opt-in` qua env flag để A/B dễ dàng

Không đổi public API `vllm serve`.

## Thực trạng source quan trọng

Legacy path đang dùng:

- `vendor/vllm-0.24.0/vllm/v1/worker/gpu_model_runner.py`
- `vendor/vllm-0.24.0/vllm/v1/worker/gpu_input_batch.py`

Trong cùng tree đã có sẵn các khối MRV2 đáng tái sử dụng:

- `vllm/v1/worker/gpu/buffer_utils.py`
- `vllm/v1/worker/gpu/states.py`
- `vllm/v1/worker/gpu/block_table.py`
- `vllm/v1/worker/gpu/input_batch.py`
- `vllm/v1/worker/gpu/sample/*`

Chiến lược đúng là **backport có chọn lọc từ các module này sang legacy path**
thay vì viết lại từ đầu.

## Feature Flags

Đã tồn tại:

- `VLLM_V1_BACKPORT_DEBUG`
- `VLLM_V1_BACKPORT_DEBUG_DUMP_PATH`
- `VLLM_V1_BACKPORT_CLEAR_STALE_ALLOWED_MASK`
- `VLLM_V1_BACKPORT_ASYNC_OUTPUT_REPAIR_OPT`
- `VLLM_V1_BACKPORT_SKIP_NOOP_METADATA_REFRESH`
- `VLLM_V1_BACKPORT_ROW_STATE_CACHE`
- `VLLM_V1_BACKPORT_SKIP_REORDER_ONLY_METADATA_REBUILD`

Planned core flags:

- `VLLM_V1_BACKPORT_RESIDENT_UNSCHEDULED_BATCH`
- `VLLM_V1_BACKPORT_DELTA_BLOCK_TABLE_WRITES`
- `VLLM_V1_BACKPORT_DELTA_SAMPLING_METADATA`
- `VLLM_V1_BACKPORT_GPU_NATIVE_INPUT_PREP`
- `VLLM_V1_BACKPORT_GPU_NATIVE_POS_SEQ`
- `VLLM_V1_BACKPORT_DEFER_ACCEPTED_SYNC`
- `VLLM_V1_BACKPORT_GPU_REQ_STATE_STORE`
- `VLLM_V1_BACKPORT_UVA_SAMPLING_STATE`

## Phase 0 - Foundation

### Scope

- gom toàn bộ feature flags về registry chung
- giữ instrumentation đang có và mở rộng nếu cần
- map rõ module MRV2 nào sẽ được hút sang legacy path

### Deliverable

- `vllm/v1/worker/backport_flags.py`
- compile pass với legacy runner

### Gate

- không đổi behavior khi mọi flag đều off
- py_compile pass

## Phase 1 - Resident Persistent Batch

### Goal

Không remove request khỏi persistent batch chỉ vì request đó không được schedule
ở step hiện tại nhưng vẫn còn sống.

### Flag

- `VLLM_V1_BACKPORT_RESIDENT_UNSCHEDULED_BATCH`

### Core change

- tách khái niệm:
  - `resident rows`: mọi request còn active
  - `scheduled rows`: subset được dùng để build input step hiện tại
- thêm mapping `scheduled_batch_idx -> resident_row_idx`
- các path hiện đang assume mọi `input_batch.req_ids` đều có
  `scheduler_output.num_scheduled_tokens[req_id]` phải đổi sang duyệt
  scheduled subset thay vì toàn resident batch

### Files likely touched

- `gpu_model_runner.py`
- `gpu_input_batch.py`

### Invariants

- row cư trú của request không đổi trong suốt lifetime trừ khi request finish/preempt
- unscheduled-alive request không bị mất token history, block_ids, num_computed
- scheduled subset luôn phản ánh đúng ordering step hiện tại

### Tests

- add -> unschedule 3 step -> reschedule
- finish xen kẽ resident row khác
- async scheduling on/off
- hybrid align path

## Phase 2 - Delta Block Table Writes

### Goal

Port cơ chế staged ragged writes của MRV2 để block table không còn append/copy
theo kiểu legacy rộng tay.

### Flag

- `VLLM_V1_BACKPORT_DELTA_BLOCK_TABLE_WRITES`

### Core change

- tái dùng ý tưởng từ `gpu/block_table.py`
- staged write theo row + apply theo batch
- gather block tables cho scheduled subset thay vì phụ thuộc layout resident trực tiếp

### Tests

- append block ids
- overwrite khi resume/preempt
- nhiều kv groups
- slot mapping đúng trước/sau reorder

## Phase 3 - Delta Sampling Metadata

### Goal

Không rebuild full sampling metadata chỉ vì một số row thay đổi.

### Flags

- `VLLM_V1_BACKPORT_DELTA_SAMPLING_METADATA`
- `VLLM_V1_BACKPORT_UVA_SAMPLING_STATE`

### Core change

- sampling states theo row dùng kiểu UVA-backed / staged update
- chỉ cập nhật row thay đổi cho:
  - temperature
  - top_p
  - top_k
  - penalties
  - allowed ids / bad words / logprob token ids nếu feature dùng

### Tests

- greedy
- non-greedy với giá trị khác nhau giữa rows
- penalties on/off
- allowed_token_ids, bad_words, logprobs

## Phase 4 - Deferred Sync / GPU-Authoritative State

### Goal

Giảm các điểm CPU phải chờ GPU trả state quay lại sớm hơn mức cần thiết.

### Flag

- `VLLM_V1_BACKPORT_DEFER_ACCEPTED_SYNC`

### Core change

- trì hoãn accepted-token repair/copyback
- chỉ sync khi path downstream thật sự cần
- với async path, ưu tiên GPU-authoritative tensors lâu hơn

### Tests

- async scheduling bật/tắt
- spec decode path
- hybrid align path
- deterministic output sanity

## Phase 5 - GPU Request State Store

### Goal

Port dần `RequestState` kiểu MRV2 để token history / lens / num_computed / next prefill
không còn lệ thuộc nặng vào CPU mirror legacy.

### Flag

- `VLLM_V1_BACKPORT_GPU_REQ_STATE_STORE`

### Core change

- tái dùng ý tưởng từ `gpu/states.py`
- staged token history writes
- prompt_len / prefill_len UVA-backed
- num_computed_tokens mirror rõ ràng hơn

### Tests

- fresh prefill
- resumed/preempted request
- chunked prefill
- multimodal prompt spans

## Phase 6 - GPU-Native Input Prep

### Goal

Port các kernel input-prep thiết thực nhất từ `gpu/input_batch.py`.

### Flags

- `VLLM_V1_BACKPORT_GPU_NATIVE_INPUT_PREP`
- `VLLM_V1_BACKPORT_GPU_NATIVE_POS_SEQ`

### Core change

- GPU-side prepare prefill inputs
- GPU-side positions / seq_lens prep
- scheduled subset gather dùng `idx_mapping`

### Tests

- prefill only
- mixed prefill + decode
- async scheduling
- mrope / xdrope / multimodal

## Phase 7 - Integration / Validation

### Scope

- smoke/invariant suite cho từng feature
- compose / image build dùng vendor patched
- benchmark harness sau khi toàn bộ feature hoàn tất

### Gate

- no-flag path giữ nguyên
- mỗi flag bật riêng đều pass invariant tests
- combo an toàn mới đưa vào benchmark

## Nguyên tắc rollout

1. Không merge logic lớn vào path mặc định.
2. Mỗi phase chỉ bật bằng env flag riêng.
3. Mỗi phase phải có test correctness trước benchmark.
4. Nếu phase làm gãy hybrid/align hoặc prefix caching hiện tại, phase đó dừng lại
   và bị tách riêng.
