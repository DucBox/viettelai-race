# vLLM v0.24.0 vs v0.22.1 Line-by-line Diff Trace

Muc tieu:
- dung cac file trace da tao cho v0.22.1 lam baseline
- trace nguoc source v0.24.0 trong image local
- ghi ro dong nao giu nguyen, dong nao doi, if/else nao moi
- noi thang thay doi nao co kha nang anh huong TTFT/TPOT cua Qwen3.5 race

Nguon:
- v0.22.1 source: `/private/tmp/vllm_0_22_1_src`
- v0.24.0 image local: `vllm/vllm-openai:v0.24.0`
- v0.24.0 source copied: `/private/tmp/vllm_0_24_0_src`
- v0.24.0 build commit: `ee0da84ab9e04ac7610e28580af62c365e898389`
- v0.24.0 package path in image: `/usr/local/lib/python3.12/dist-packages/vllm`

Baseline docs duoc doi chieu:
- `docs/vllm-v0.22.1-line-by-line-traces.md`
- `docs/vllm-v0.22.1-latency-flag-trace.md`
- `docs/vllm-v0.22.1-implementation-map.md`

Ket luan ngan:
- Scheduler default/validation gan nhu khong doi.
- Qwen3.5 van khong support `mamba-cache-mode=all`.
- GDN backend selector gan nhu khong doi.
- Khac biet lon nam o runtime scheduler 0.24.0 va GDN metadata path:
  DP prefill throttling, Marconi common-prefix admission, hybrid connector cache-hit
  path, async KV-load reservations, dynamic spec scheduling, va prefill-only GDN
  metadata trong mixed decode+prefill batch.

---

## 1. Chunked Prefill & Batching

Baseline section:
- `docs/vllm-v0.22.1-line-by-line-traces.md`, section 1

### 1.1 Default resolve: khong doi logic, chi shift line

v0.22.1:
- `engine/arg_utils.py`, L2365-L2433

v0.24.0:
- `engine/arg_utils.py`, L2458-L2527

Same:
- `default_chunked_prefill = model_config.is_chunked_prefill_supported`
- `default_prefix_caching = model_config.is_prefix_caching_supported`
- user unset thi lay model default
- user disable generate chunked prefill thi warning
- user enable pooling unsupported thi warning
- RISCV CPU force disable chunked prefill va prefix caching

Khac:
- khong co thay doi semantic quan trong trong block nay.

Tac dong:
- Ket luan cu van dung: khong doc CLI default raw; phai doc resolved
  `ModelConfig`/platform path.

### 1.2 `max_num_batched_tokens` default: khong doi logic

v0.22.1:
- `engine/arg_utils.py`, L2475-L2564

v0.24.0:
- `engine/arg_utils.py`, L2568-L2655

Same:
- `world_size = pipeline_parallel_size * tensor_parallel_size`
- neu user da set `max_num_batched_tokens`, performance preset khong override
- `performance_mode == "throughput"` chi double gia tri originally `None`
- neu chunked prefill tat, default budget phai cover `max_model_len`
- multimodal prefix-LM co the raise floor
- default cap: `max_num_seqs * max_model_len`
- default `max_num_seqs = min(max_num_seqs, max_num_batched_tokens)`

Tac dong:
- v26/v28 gain 0.24 khong den tu default scheduler flags neu compose da set tay
  `max-num-batched-tokens` va `max-num-seqs`.

### 1.3 SchedulerConfig validation: khong doi logic

v0.22.1:
- `config/scheduler.py`, L224-L308

v0.24.0:
- `config/scheduler.py`, L237-L320

Same:
- encoder-decoder force disable chunked prefill/prefix caching
- encoder compute/cache budget = `max_num_batched_tokens`
- partial prefill threshold default = `int(max_model_len * 0.04)` khi
  `max_num_partial_prefills > 1`
- raise neu chunked prefill off va `max_num_batched_tokens < max_model_len`
- raise neu `max_num_batched_tokens < max_num_seqs`
- warning neu budget > `max_num_seqs * max_model_len`
- partial prefill > 1 yeu cau chunked prefill
- `max_long_partial_prefills <= max_num_partial_prefills`

Tac dong:
- Validation invariants trong docs 0.22.1 van dung cho 0.24.0.

### 1.4 Scheduler init: co 2 thay doi nho nhung can ghi

v0.22.1:
- `v1/core/sched/scheduler.py`, L102-L108

v0.24.0:
- `v1/core/sched/scheduler.py`, L107-L122

Diff logic:

```text
0.22.1:
self.max_num_scheduled_tokens = (
    scheduler_config.max_num_scheduled_tokens
    if scheduler_config.max_num_scheduled_tokens
    else scheduler_config.max_num_batched_tokens
)

0.24.0:
self.max_num_scheduled_tokens = (
    scheduler_config.max_num_scheduled_tokens
    if scheduler_config.max_num_scheduled_tokens is not None
    else scheduler_config.max_num_batched_tokens
)
self.num_sampled_tokens_per_step = 1 if not diffusion else 0
```

Line-by-line:
- 0.24 L109-L112: `0` bay gio la gia tri hop le, khong fallback ve
  `max_num_batched_tokens`. Neu ai set `max_num_scheduled_tokens=0`, scheduler
  se co token budget 0.
- 0.24 L119-L122: diffusion model khong sample token moi moi step, nen cap
  max-model-len ve sau khong con hardcode `-1`.

Tac dong:
- Submit hien tai khong set `max_num_scheduled_tokens`, nen khong anh huong.
- Qwen3.5 khong phai diffusion, nen `num_sampled_tokens_per_step=1`, equivalent
  voi hardcode `-1` cu.

### 1.5 RUNNING loop: 0.24 them prefill throttling va decode cadence

v0.22.1:
- `scheduler.py`, L329-L423

v0.24.0:
- `scheduler.py`, L388-L520

New code:

```text
def schedule(self, throttle_prefills: bool = False)
self.current_step += 1

defer_prefills = (
    throttle_prefills and not self.prefill_capacity_bound
) and any(not r.is_prefill_chunk for r in self.running)

if self.current_step < request.next_decode_eligible_step:
    continue

if defer_prefills and request.is_prefill_chunk:
    continue
```

Line-by-line:
- L388: signature them `throttle_prefills`.
- L389: scheduler co `current_step`.
- L424-L428: DP prefill balancing. Neu step bi throttle va lan release truoc chua
  capacity-bound, scheduler defer prefill de cho decode lap day step.
- L451-L455: V2 + pipeline parallel + async can giu cadence giua cac decode cua
  cung request.
- L457-L461: prefill chunk dang RUNNING co the bi skip trong throttled step.
- L463-L470: core token calculation va long threshold giong cu.
- L474-L479: max-len cap doi tu `max_model_len - 1 - computed` thanh
  `max_model_len - computed - num_sampled_tokens_per_step`.
- L499-L502: Mamba block-aligned split van con.
- L504-L520: num_new_tokens = 0 van `continue`, comment ly do van co Mamba align.

Tac dong TTFT/TPOT:
- TPOT co the tot hon khi scheduler uu tien decode trong throttled step.
- TTFT/prefill co the bi delay vi prefill chunks bi defer.
- Neu workload race co concurrency cao, branch nay co the giai thich 0.24 giam
  TTFT tail/queue behavior khac 0.22.1, nhung phu thuoc caller co pass
  `throttle_prefills=True` khong.

### 1.6 WAITING loop: 0.24 them hybrid connector path va prefill defer

v0.22.1:
- `scheduler.py`, L544-L804

v0.24.0:
- `scheduler.py`, L640-L987

New/changed code:

```text
num_uncached_common_prefix_tokens = 0

if connector and has_mamba_layers and coordinator is HybridKVCacheCoordinator:
    computed, per_group_hits =
      find_longest_cache_hit_per_group(request.block_hashes, request.num_tokens - 1)
    new_computed_blocks = create_kv_cache_blocks(computed)
    num_new_local_computed_tokens = max(per_group_hits)
else:
    get_computed_blocks(request)

if has_mamba_layers:
    num_uncached_common_prefix_tokens =
      coordinator.num_uncached_common_prefix_tokens

elif defer_prefills and request.num_computed_tokens == 0:
    break

if need_mamba_block_aligned_split and not load_kv_async:
    _mamba_block_aligned_split(..., num_uncached_common_prefix_tokens)

reserved_blocks = _inflight_prefill_reserved_blocks() if load_kv_async else 0
allocate_slots(..., reserved_blocks=reserved_blocks, has_scheduled_reqs=bool(self.running))
```

Line-by-line:
- L670: introduce `num_uncached_common_prefix_tokens`.
- L675-L701: khi co KV connector + Mamba hybrid coordinator, prefix-cache local
  hit khong con la `get_computed_blocks` chung. No tim hit per group, lay max
  per-group hit lam FA hit length, va comment noi Mamba state duoc transfer
  unconditional trong worker.
- L714-L720: lay hint `num_uncached_common_prefix_tokens` cho Marconi APC logic.
- L782-L790: async KV loads van duoc start trong throttled step, nhung new prefill
  compute bi defer.
- L801-L812: chunked prefill branch cu van con.
- L832-L840: block alignment bi skip khi `load_kv_async`. 0.22.1 khong co guard
  `not load_kv_async`.
- L866-L884: async load admission co reserved blocks tu inflight prefills.
- L884-L886: `allocate_slots` co args moi `reserved_blocks`,
  `has_scheduled_reqs`.
- L937 va L961-L963: request async load / request con prefill duoc add vao
  `_inflight_prefills`.
- L984-L987: sau pass, neu khong defer, record `prefill_capacity_bound`.

Tac dong:
- TTFT: 0.24 co the schedule prefix-cache/hybrid connector chinh xac hon theo
  group, va co Marconi common-prefix admission hint.
- Queue: async KV load khong con de deadlock/preemption de bang reserved-block
  accounting.
- TPOT: prefill defer co the bao ve decode cadence.

### 1.7 `_mamba_block_aligned_split`: 0.24 them Marconi common-prefix admission

v0.22.1:
- `scheduler.py`, L279-L327

v0.24.0:
- `scheduler.py`, L330-L386

Diff logic:

```text
0.22.1:
assert num_external_computed_tokens == 0

0.24.0:
no assert
extra arg: num_uncached_common_prefix_tokens = 0

after normal block alignment:
if num_uncached_common_prefix_tokens >= block_size
   and num_new_tokens > num_uncached_common_prefix_tokens:
    num_new_tokens = num_uncached_common_prefix_tokens
    num_new_tokens = num_new_tokens // block_size * block_size
```

Line-by-line:
- L336: new arg `num_uncached_common_prefix_tokens`.
- L338-L342: computed token formula van gom local + external.
- L343-L375: block alignment core y het 0.22.1.
- L377-L385: new Marconi cache admission optimization. Neu common prefix chua
  cache du lon hon block, scheduler cat chunk dung common-prefix length va giu
  alignment.

Tac dong:
- TTFT round sau: co the tang cache admission cho common prefix, giam prefill
  that su o request sau.
- Queue: co the cat chunk nho hon planned token budget de uu tien cache boundary.
- Day la thay doi rat lien quan Qwen3.5 + prefix caching + Mamba align.

### 1.8 `_update_after_schedule`: same core, them deferred free/inflight cleanup

v0.22.1:
- `scheduler.py`, L951-L967

v0.24.0:
- `scheduler.py`, L1130-L1155

Same:
- cong `request.num_computed_tokens += num_scheduled_token`
- set `request.is_prefill_chunk = computed < tokens + placeholders`
- structured output chi active khi khong con prefill chunk

New:
- L1144-L1146: neu `defer_block_free`, record `request.last_sched_seq`.
- L1153-L1155: neu request het prefill, remove khoi `_inflight_prefills`.

Tac dong:
- Safer async scheduling/KV connector behavior.
- Giam nguy co free/reuse block qua som khi multiple inflight batches.

---

## 2. GDN Prefill Backend

Baseline section:
- `docs/vllm-v0.22.1-line-by-line-traces.md`, section 2

### 2.1 Backend resolve: khong doi

v0.22.1:
- `qwen_gdn_linear_attn.py`, L150-L211

v0.24.0:
- `qwen_gdn_linear_attn.py`, L150-L211

Same:
- non-CUDA -> Triton/FLA
- SM90 -> FlashInfer supported
- Blackwell requires family 100, `head_k_dim == 128`, CUDA runtime >= 13
- Blackwell FlashInfer requires intact `nvidia-cutlass-dsl-libs-cu13`
- requested `flashinfer/auto` + support -> active `flashinfer`
- requested `cutedsl` + support -> active `cutedsl`
- otherwise -> `triton`

Tac dong:
- Neu v0.24 nhanh hon, khong phai do selector logic nay doi.
- Co the do package/kernel version trong image moi hon, hoac metadata path ben
  duoi doi.

### 2.2 CustomOp dispatch va FlashInfer wrapper: khong doi

v0.24.0:
- `fi_chunk_gated_delta_rule`: L241-L287
- `ChunkGatedDeltaRule.__init__`: L290-L312
- `forward_cuda/native/cutedsl`: L313-L416

Same voi 0.22.1:
- q/k L2 norm optional
- q/k/v/g/beta squeeze + contiguous
- state/g/beta cast float32 cho FlashInfer
- FlashInfer returns output or `(output, final_state)`
- active backend bind sang `forward_cuda` / `forward_cutedsl` / `forward_native`

### 2.3 GDN metadata path: 0.24 doi dang ke

v0.22.1:
- `v1/attention/backends/gdn_attn.py`, L323-L472

v0.24.0:
- `v1/attention/backends/gdn_attn.py`, L328-L510

New fields:
- `prefill_query_start_loc`
- `prefill_state_indices`
- `prefill_has_initial_state`

Diff logic:

```text
if num_prefills > 0:
    if spec_sequence_masks is None and num_decodes > 0:
        prefill_query_start_loc =
            non_spec_query_start_loc[num_decodes:] - num_decode_tokens
        prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
    else:
        prefill_query_start_loc = non_spec_query_start_loc
        prefill_state_indices = non_spec_state_indices_tensor

    prepare chunk metadata from prefill_query_start_loc only
    CPU -> GPU copy uses async_tensor_h2d(...)
```

Line-by-line:
- L330-L332: new prefill-only metadata fields.
- L336-L339: comment moi: mixed non-spec batch peel decodes off to recurrent
  kernel, nen chunk metadata phai build tu prefill-only cu_seqlens.
- L340-L350: neu batch co decode + prefill va khong spec, rebase prefill start loc
  bang cach bo front decode slice.
- L356-L368: cutedsl dung `prefill_query_start_loc`.
- L379-L387: Triton/FLA chunk metadata CPU->GPU copy dung `async_tensor_h2d`,
  thay vi `.to(device, non_blocking=True)` cu.
- L400-L403: `prefill_has_initial_state` cung peel theo decode count.
- L413-L416: cudagraph metadata batch size doi thanh `m.num_reqs`, vi state/query
  metadata index theo request, khong phai padded token count.
- L485-L510: pack extra fields vao `GDNAttentionMetadata`.

Tac dong TTFT/TPOT:
- Mixed decode+prefill batch la core workload khi concurrency cao. 0.24 tach
  prefill-only GDN metadata sach hon, tranh de decode tokens lam lech chunk
  metadata.
- `async_tensor_h2d` giam kha nang sync CPU/GPU trong metadata path.
- Đây là candidate mạnh để giải thích v0.24 tốt hơn v0.22.1 ở TTFT/passed_slo
  mà không cần scheduler defaults đổi.

---

## 3. Qwen3.5 `mamba-cache-mode`

Baseline section:
- `docs/vllm-v0.22.1-line-by-line-traces.md`, section 3

### 3.1 Cache mode rewrite: khong doi semantic

v0.22.1:
- `model_executor/models/config.py`, L337-L395

v0.24.0:
- `model_executor/models/config.py`, L408-L462

Same:
- prefix caching enabled + mode `none` -> `all` neu model supports Mamba prefix
  caching, nguoc lai `align`
- mode `all` + model khong support -> fallback `align`
- `align` requires chunked prefill
- mamba block size unset -> `cache_config.block_size`
- prefix caching disabled -> mode forced `none`, block size -> max model len

### 3.2 Qwen3.5 hard guard: van chua support `all`

v0.22.1:
- `model_executor/models/qwen3_5.py`, L459-L463

v0.24.0:
- `model_executor/models/qwen3_5.py`, L469-L473

Same:

```text
if cache_config.mamba_cache_mode == "all":
    raise NotImplementedError(
        "Qwen3.5 currently does not support 'all' prefix caching, "
        "please use '--mamba-cache-mode=align' instead"
    )
```

Tac dong:
- v0.24.0 khong mo khoa `mamba-cache-mode=all` cho Qwen3.5.
- A/B dung cho Qwen3.5 van la `none` vs `align`, khong phai `all`.
- Neu compose set `all`, engine hoac fallback truoc do ve `align`, hoac crash
  neu `all` toi constructor.

### 3.3 Align validation: khong doi

v0.22.1:
- `config/vllm.py`, L2100-L2118

v0.24.0:
- `config/vllm.py`, L2140-L2158

Same:
- `block_size <= max_num_batched_tokens`
- `long_prefill_token_threshold >= block_size` neu threshold > 0
- `disable_chunked_mm_input` phai false
- V2 model runner chua support `mamba_cache_mode='align'`

### 3.4 `_align_hybrid_block_size`: formula chinh van giong

v0.22.1:
- `platforms/interface.py`, around L509-L670

v0.24.0:
- `platforms/interface.py`, L633-L785

Same important formula:

```text
if mamba_cache_mode == "all":
    base_chunk_size = user_mamba_block_size or model_config.get_mamba_chunk_size()
    attn_tokens_per_mamba_state = ceil(mamba_page_size / attn_page_size_1_token)
    chunk_size = lcm(base_chunk_size, kernel_block_alignment_size)
    attn_block_size = chunk_size * ceil(attn_tokens_per_mamba_state / chunk_size)
    cache_config.mamba_block_size = attn_block_size
else:
    attn_block_size = kernel_block_alignment_size * ceil(
        mamba_page_size / (kernel_block_alignment_size * attn_page_size_1_token)
    )

if block_size < attn_block_size:
    block_size = attn_block_size
if mamba_cache_mode == "align":
    mamba_block_size = block_size
```

Tac dong:
- Nhan xet cu van dung: effective attention block size co the bi Mamba page/kernel
  alignment nang len.
- Rieng Qwen3.5 khong chay `all`, nen nhanh/cham lien quan `align` va effective
  `block_size`, khong phai `all`.

### 3.5 Qwen3.5 Mamba SSM dtype rewrite: khong doi

v0.22.1:
- `model_executor/models/config.py`, L536-L560

v0.24.0:
- `model_executor/models/config.py`, L603-L626

Same:
- `mamba_ssm_cache_dtype=auto` -> lay `hf_text_config.mamba_ssm_dtype` neu co
- user override khac HF config -> warning, dung user value

---

## 4. Cudagraph / Memory / Other Flag Conclusions

### 4.1 GPU memory profiling: khong thay doi logic lon

Spot-checked:
- v0.22.1 `v1/worker/gpu_worker.py::determine_available_memory`, around L358+
- v0.24.0 same function, around L400+

Same:
- `kv_cache_memory_bytes` explicit van override KV reservation
- van run `profile_run()`
- `gpu_memory_utilization` van la capacity lever, khong phai kernel-speed lever

### 4.2 Cudagraph sizing: core logic gan nhu giong

Spot-checked:
- v0.22.1 `config/vllm.py::_set_cudagraph_sizes`, around L1597+
- v0.24.0 `config/vllm.py::_set_cudagraph_sizes`, around L1635+

Same:
- `performance_mode=interactivity` capture every size 1..min(max,32)
- otherwise capture `[1,2,4]`, then multiples of 8/16
- user `cudagraph_capture_sizes` dedup/sort/filter theo max tokens
- final `max_cudagraph_capture_size` set tu valid capture sizes

Tac dong:
- v0.24 speed gain trong results khong nen quy thang cho cudagraph default neu
  workload/flags khong doi.

---

## 5. What likely explains v0.24.0 better results

Based on code diff only, not benchmark measurement:

1. Scheduler runtime got smarter:
   - DP prefill balancing can protect decode steps.
   - Mamba hybrid connector path can reason per KV group.
   - Marconi common-prefix admission can schedule cache-friendly chunks.
   - async KV load has reserved-block accounting.

2. GDN metadata path got cleaner:
   - mixed decode+prefill batch peels decode slice before chunk metadata.
   - CPU->GPU metadata copy uses `async_tensor_h2d`.
   - cudagraph metadata uses request count for request-indexed tensors.

3. What did not explain the gain:
   - `mamba-cache-mode=all` is still unsupported for Qwen3.5.
   - Scheduler default batch/seq validation is unchanged.
   - GDN backend selector is unchanged.
   - GPU memory profiling logic is materially unchanged.

---

## 6. Updates needed to existing docs

For `docs/vllm-v0.22.1-line-by-line-traces.md`, if creating v0.24 overlay:

- Section 1.4: add note about `max_num_scheduled_tokens is not None`.
- Section 1.5: add 0.24 branches `throttle_prefills`, `current_step`,
  `next_decode_eligible_step`, `defer_prefills`.
- Section 1.6: add hybrid connector/per-group prefix hit path, async KV load
  reservation, skip Mamba align on load async.
- Section 1.7: add Marconi common-prefix admission optimization.
- Section 2.5: update GDN metadata path with prefill-only metadata and
  `async_tensor_h2d`.
- Section 3: keep Qwen3.5 `all` unsupported conclusion unchanged.

---

## 7. Section 10-14 Overlay: flag-by-flag v0.24.0 trace

This section traces the flags already documented in
`docs/vllm-v0.22.1-latency-flag-trace.md` against the real v0.24.0 image.
Goal: not benchmark, but check whether source-level behavior changed.

### 7.1 Parser/config funnel

v0.24.0 still uses the same overall route:

```text
CLI arg -> EngineArgs field -> create_engine_config()
        -> ModelConfig / CacheConfig / SchedulerConfig / ParallelConfig /
           CompilationConfig / KernelConfig / ObservabilityConfig / VllmConfig
        -> runtime scheduler / worker / OpenAI serving
```

Important v0.24.0 line references:

- Cache flags are registered in `engine/arg_utils.py`, L1146-L1193:
  `--block-size`, `--gpu-memory-utilization`, `--kv-cache-memory-bytes`,
  `--kv-cache-dtype`, `--num-gpu-blocks-override`,
  `--enable-prefix-caching`, `--prefix-caching-hash-algo`,
  `--calculate-kv-scales`, `--kv-cache-dtype-skip-layers`,
  `--kv-sharing-fast-prefill`, `--mamba-cache-dtype`,
  `--mamba-ssm-cache-dtype`, `--mamba-block-size`,
  `--mamba-cache-mode`, `--kv-offloading-size`,
  `--kv-offloading-backend`.
- Observability KV metrics are registered in `engine/arg_utils.py`,
  L1358-L1364.
- Scheduler flags are registered in `engine/arg_utils.py`, L1392-L1453:
  old flags plus new v0.24 scheduler flags `--watermark` and
  `--prefill-schedule-interval`.
- Cudagraph flags are registered in `engine/arg_utils.py`, L1461-L1467.
- Kernel flags are registered in `engine/arg_utils.py`, L1475-L1485.
- `--optimization-level` and `--performance-mode` are registered in
  `engine/arg_utils.py`, L1533-L1535.
- `--gdn-prefill-backend` is registered in `engine/arg_utils.py`,
  L1570-L1575, with choices unchanged: `flashinfer`, `triton`, `cutedsl`.

Conclusion:
- Most old flags still enter the same config fields.
- The v0.24-only additions found in this trace are `--watermark` and
  `--prefill-schedule-interval`.
- Valid configs mostly preserve behavior; invalid `block_size/hash_block_size`
  can now be rejected earlier due to pydantic `Field(..., gt=0)`.

### 7.2 Scheduler flags: unchanged defaults, new admission controls

Unchanged scheduler fields from v0.22.1 to v0.24.0:

- `max_num_batched_tokens`: v0.22.1 `config/scheduler.py` L49-L54,
  v0.24.0 L49-L54.
- `max_num_scheduled_tokens`: v0.22.1 L56-L61, v0.24.0 L56-L61.
  Field validation changed from plain optional to `Field(default=None, ge=0)`,
  but valid positive values are semantically same.
- `max_num_seqs`: v0.22.1 L63-L68, v0.24.0 L63-L68.
- `max_num_partial_prefills`: v0.22.1 L70-L72, v0.24.0 L70-L72.
- `max_long_partial_prefills`: v0.22.1 L74-L78, v0.24.0 L74-L78.
- `long_prefill_token_threshold`: v0.22.1 L80-L82, v0.24.0 L80-L82.
- `enable_chunked_prefill`: v0.22.1 L84-L90, v0.24.0 L84-L90.
- `scheduler_reserve_full_isl`: v0.22.1 L140-L144, v0.24.0 L140-L144.
- `async_scheduling`: v0.22.1 L146-L149, v0.24.0 L158-L161.
- `stream_interval`: v0.22.1 L151-L155, v0.24.0 L163-L167.

New v0.24.0 scheduler fields:

- `watermark`: `config/scheduler.py` L146-L151.
- `prefill_schedule_interval`: `config/scheduler.py` L153-L156.

Validation still matches v0.22.1 for the important safety checks:

- Encoder-decoder disables chunked prefill: v0.24.0
  `config/scheduler.py` L237-L246.
- If `max_num_partial_prefills > 1` and threshold is zero, threshold becomes
  `int(max_model_len * 0.04)`: v0.24.0 L257-L260.
- If chunked prefill is disabled and `max_num_batched_tokens < max_model_len`,
  fail: v0.24.0 L272-L284.
- `max_num_batched_tokens >= max_num_seqs` required: v0.24.0 L286-L291.
- `max_long_partial_prefills <= max_num_partial_prefills` required:
  v0.24.0 L315-L318.

TTFT/TPOT implication:
- Old knobs keep same validity semantics.
- New `watermark` can intentionally leave free KV blocks when admitting
  waiting/preempted requests, reducing preemption churn but also lowering
  immediate admission capacity.
- New `prefill_schedule_interval` can defer prefill admission in DP/MoE mode,
  smoothing TPOT but potentially raising TTFT for new prompts.

### 7.3 Runtime scheduler: where v0.24 actually differs

This is the biggest source-level difference.

#### `max_num_scheduled_tokens`

v0.22.1:

```python
self.max_num_scheduled_tokens = (
    self.scheduler_config.max_num_scheduled_tokens
    if self.scheduler_config.max_num_scheduled_tokens
    else self.scheduler_config.max_num_batched_tokens
)
```

Source: `v1/core/sched/scheduler.py`, L104-L108.

v0.24.0:

```python
self.max_num_scheduled_tokens = (
    self.scheduler_config.max_num_scheduled_tokens
    if self.scheduler_config.max_num_scheduled_tokens is not None
    else self.scheduler_config.max_num_batched_tokens
)
```

Source: `v1/core/sched/scheduler.py`, L109-L112.

Meaning:
- v0.22.1 treats `0` as false and falls back.
- v0.24.0 treats `0` as a real value if it somehow reaches scheduler.
- Usually not a tuning lever for our submit because CLI does not directly expose
  `max_num_scheduled_tokens`, but it matters for programmatic config.

#### DP prefill throttling

v0.24.0 `schedule(self, throttle_prefills: bool = False)`:

- L388: scheduler accepts `throttle_prefills`.
- L389: increments `current_step`.
- L424-L428: computes `defer_prefills`.
- L457-L461: skips already-running prefill chunks on deferred steps.
- L782-L790: for new waiting requests, async KV loads may still start, but new
  local prefill compute is deferred.
- L984-L987: records whether the release step was still capacity-bound.

v0.22.1 has no equivalent branch in `schedule()`.

TTFT/TPOT implication:
- This is a fairness/smoothing mechanism, not a raw prefill speedup.
- It can improve TPOT and passed-SLO under high concurrency by protecting decode
  steps from prefill-heavy waves.
- It can increase TTFT for some new requests because admission can be held until
  a cadence-aligned step.

#### Decode cadence

v0.24.0:

- L451-L455 checks `request.next_decode_eligible_step`.
- Comment says V2+PP+async needs `pp_size` steps between same-request decodes.

v0.22.1 has no equivalent check in the RUNNING loop.

TTFT/TPOT implication:
- Mostly affects async + PP configurations.
- For single PP rank it is less likely to matter, but the branch is still part
  of the scheduler state machine in v0.24.0.

#### Max model length cap

v0.22.1:

```python
num_new_tokens = min(
    num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
)
```

Source: `v1/core/sched/scheduler.py`, L394-L398.

v0.24.0:

```python
num_new_tokens = min(
    num_new_tokens,
    self.max_model_len
    - request.num_computed_tokens
    - self.num_sampled_tokens_per_step,
)
```

Source: `v1/core/sched/scheduler.py`, L472-L479.

Meaning:
- For normal generate, `num_sampled_tokens_per_step` is 1, so behavior is same.
- For diffusion path, v0.24.0 sets sampled tokens per step differently, so the
  cap is more general.

### 7.4 `watermark`: new v0.24 KV-cache admission headroom

Flag path:

- CLI: `engine/arg_utils.py`, L1439.
- Config field: `config/scheduler.py`, L146-L151.
- Passed into scheduler config: `engine/arg_utils.py`, L2132.
- Scheduler creates KV manager with `watermark=self.scheduler_config.watermark`:
  `v1/core/sched/scheduler.py`, L267.
- KV manager stores `watermark_blocks = int(watermark * num_blocks)`:
  `v1/core/kv_cache_manager.py`, L160-L163.

Runtime:

- `allocate_slots()` starts with `watermark_blocks = 0`:
  `v1/core/kv_cache_manager.py`, L363.
- It applies watermark only if there is already scheduled work and request
  status is `WAITING` or `PREEMPTED`: L364-L370.
- In full-sequence admission, required blocks become
  `num_blocks_to_allocate + watermark_blocks`: L372-L387.
- In normal allocation, available blocks subtract `reserved_blocks`, then
  required blocks add watermark: L414-L420.

Line-by-line meaning:
- L363: default no headroom.
- L366-L370: headroom is not charged to the first request; only later
  waiting/preempted admissions pay it.
- L385-L387: if full prompt would fit only by consuming headroom, reject now.
- L416-L420: even after skipped blocks are freed, admission still respects
  reserved async-load blocks and watermark.

TTFT/TPOT implication:
- Higher `--watermark` can lower preemption/recompute and smooth TPOT.
- Too high `--watermark` reduces effective KV capacity, increasing queue wait
  and TTFT.
- Default `0.0` means v0.24 behaves like v0.22 on this dimension unless set.

### 7.5 `prefill_schedule_interval`: new v0.24 DP/MoE prefill cadence

Flag path:

- CLI: `engine/arg_utils.py`, L1440-L1442.
- Config field: `config/scheduler.py`, L153-L156.
- Passed into SchedulerConfig: `engine/arg_utils.py`, L2133.
- DPEngineCore stores it: `v1/engine/core.py`, L1763-L1765.

Runtime:

- `_should_throttle_prefills()` returns true when
  `prefill_schedule_interval > 1` and `step_counter % interval != 0`:
  `v1/engine/core.py`, L1916-L1923.
- Scheduler receives `throttle_prefills=True`, then enters the defer logic
  described in section 7.3.

Line-by-line meaning:
- L1916-L1919: comment says this exists for DP balancing.
- L1921: disabled unless interval is greater than 1.
- L1922: only cadence-aligned steps admit prefills.

TTFT/TPOT implication:
- This is useful only when DP/MoE imbalance or prefill waves hurt decode.
- Not a general single-GPU speed lever.
- For race-style SLO, it can improve p95/p99 TPOT at the cost of some TTFT.

### 7.6 Cache/memory flags

#### `--block-size` and `--hash-block-size`

v0.22.1:
- `CacheConfig.block_size`: `config/cache.py`, L47.
- `hash_block_size`: L54.
- Both use `SkipValidation`.

v0.24.0:
- `block_size`: `config/cache.py`, L48, `Field(default=None, gt=0)`.
- `hash_block_size`: L55, `Field(default=None, gt=0)`.

Meaning:
- Valid positive values behave the same.
- Invalid non-positive values should fail earlier in v0.24.0.
- Runtime block alignment conclusions from v0.22.1 still stand:
  Mamba/hybrid code can align effective physical block size beyond the raw CLI
  `--block-size`.

#### `--gpu-memory-utilization` and `--kv-cache-memory-bytes`

v0.24.0 memory profiling:

- `determine_available_memory()` starts at `v1/worker/gpu_worker.py`, L400.
- If `kv_cache_memory_bytes` is set, vLLM still runs `profile_run()` then
  returns explicit KV bytes: L412-L430.
- Otherwise it enters memory profiling and `profile_run()`: L432-L439.

Same conclusion as v0.22.1:
- `--gpu-memory-utilization` is a capacity lever.
- `--kv-cache-memory-bytes` overrides automatic KV reservation.
- Neither directly accelerates kernels; they affect queue/preemption/fit.

#### `--kv-cache-dtype`, `--calculate-kv-scales`,
`--kv-cache-dtype-skip-layers`

v0.24.0 config:

- `cache_dtype`: `config/cache.py`, L75-L82.
- `calculate_kv_scales`: L110-L114.
- `kv_cache_dtype_skip_layers`: L115-L117.

v0.22.1 has the same dtype list and comments at `config/cache.py`, L74-L116.

Runtime quant mode:

- v0.24.0 `KVQuantMode` maps `fp8*`, `int8_per_token_head`,
  `fp8_per_token_head`, and `nvfp4`: `v1/kv_cache_interface.py`, L33-L80.
- v0.22.1 has materially the same mapping at `v1/kv_cache_interface.py`,
  L32-L78.

TTFT/TPOT implication:
- No meaningful semantic drift for fp8 KV from v0.22.1 to v0.24.0.
- `calculate_kv_scales=True` still means runtime scale work for fp8 cache.
- `kv_cache_dtype_skip_layers` can reduce quantization on selected layers, but
  is not the main v0.24 scheduler delta.

### 7.7 Prefix/Mamba flags

Unchanged config fields:

- `enable_prefix_caching`: v0.24.0 `config/cache.py`, L92-L93.
- `prefix_caching_hash_algo`: L94-L109.
- `mamba_block_size`: L121-L124.
- `mamba_cache_dtype`: L125-L128.
- `mamba_ssm_cache_dtype`: L129-L132.
- `mamba_cache_mode`: L133-L141.

Important unchanged Qwen3.5 conclusion:

- `qwen3_5.py`, L469-L473 still rejects `mamba_cache_mode == "all"`.
- Therefore v0.24.0 does not make `--mamba-cache-mode=all` usable for Qwen3.5.

Changed runtime around Mamba/prefix:

- v0.24.0 WAITING loop can ask hybrid coordinator for per-group prefix hits:
  `v1/core/sched/scheduler.py`, L675-L712.
- It records `num_uncached_common_prefix_tokens` for Mamba hybrid admission:
  L714-L720.
- It skips Mamba block alignment during async KV load: L832-L840.
- `_mamba_block_aligned_split()` adds Marconi common-prefix admission,
  described earlier in section 1.7.

TTFT/TPOT implication:
- The flag meaning is stable; runtime surrounding prefix/Mamba is smarter.
- For Qwen3.5, tune around `align`, not `all`.
- Gains from v0.24.0 are more likely from better hybrid scheduler behavior than
  a new allowed Mamba cache mode.

### 7.8 DBO / ubatching flags

Fields are effectively unchanged:

- v0.22.1 `config/parallel.py`, L196-L206:
  `enable_dbo`, `ubatch_size`, `dbo_decode_token_threshold`,
  `dbo_prefill_token_threshold`.
- v0.24.0 `config/parallel.py`, L208-L218 has the same defaults, with
  pydantic `Field(..., ge=0)` added on numeric thresholds.
- `use_ubatching` remains `enable_dbo or ubatch_size > 1`:
  v0.22.1 L497-L499, v0.24.0 L522-L524.
- `num_ubatches` remains `2 if enable_dbo else ubatch_size`:
  v0.22.1 L501-L503, v0.24.0 L526-L528.

Runtime unchanged:

- `check_ubatch_thresholds()` returns false if ubatching disabled, then checks
  decode threshold for uniform decode and prefill threshold otherwise:
  v0.24.0 `v1/worker/ubatch_utils.py`, L38-L46.
- `maybe_create_ubatch_slices()` still splits token ranges and maps them back
  to request slices using cumulative token counts: L63-L114.

Important runtime interaction:

- v0.24.0 disables cascade attention when microbatching is active:
  `v1/worker/gpu_model_runner.py`, L4145-L4148.

TTFT/TPOT implication:
- DBO behavior did not materially change from v0.22.1.
- It can help overlap compute on large batches but may lose cascade-attn path.
- Do not attribute v0.24 improvement to DBO unless compose flags changed.

### 7.9 Async scheduling and DP synchronization

v0.24.0 default resolver:

- Explicit `async_scheduling=True` hard-fails incompatible speculative configs
  or unsupported executor backend: `config/vllm.py`, L940-L963.
- If `async_scheduling is None`, vLLM enables it unless pooling,
  incompatible speculative decoding, disabled padded drafter batch, or executor
  unsupported: L964-L1004.
- It logs final state: L1006-L1009.
- If DP sync backend is unset, async scheduling defaults DP sync away from NCCL:
  L1011-L1022.

v0.22.1:
- Same high-level logic around `config/vllm.py`, L952-L993 for the fallback
  and DP sync default.

v0.24 difference:
- Not the default resolver itself, but scheduler runtime behind async has more
  branches: decode cadence, output placeholders, async KV load reservation, and
  DP prefill throttling.

TTFT/TPOT implication:
- `--async-scheduling` is still a real latency/throughput lever.
- The v0.24 implementation makes the consequences more nuanced, especially
  under PP/DP/spec/KV-transfer.

### 7.10 API residual / streaming / usage flags

Frontend fields:

- `enable_prompt_tokens_details`: `entrypoints/openai/cli_args.py`, L135-L136.
- `enable_server_load_tracking`: L137-L138.
- `enable_force_include_usage`: L139-L140.
- `host`, `port`, and server args start at `FrontendArgs`, L224-L252.

Runtime:

- Server load counter increments before handler and decrements via background
  task for JSON/streaming responses:
  `entrypoints/serve/utils/api_utils.py`, L111-L143.
- Disconnect handling decrements the counter if tracking is enabled:
  L38-L49.
- `enable_force_include_usage=True` makes usage included continuously:
  `entrypoints/serve/utils/api_utils.py`, L277-L286.
- Completion serving checks usage flags at stream start:
  `entrypoints/openai/completion/serving.py`, L300-L303.
- Prompt token details are added only when enabled and cached-token count exists:
  L447-L450 and L584-L591.
- `stream_interval > 1` buffers output until finish, first token, or enough
  delta tokens: `v1/engine/output_processor.py`, L287-L300.

TTFT/TPOT implication:
- These flags affect residual HTTP/serialization/output cadence, not GPU
  prefill/decode compute.
- `stream_interval=1` minimizes first visible token buffering.
- Higher `stream_interval` can lower CPU/network overhead but worsens perceived
  streaming granularity.
- `enable_server_load_tracking` adds small request-state accounting overhead.

### 7.11 Kernel / graph / GDN flags

Unchanged registration:

- `--cudagraph-capture-sizes`: `engine/arg_utils.py`, L1461-L1463.
- `--max-cudagraph-capture-size`: L1464-L1467.
- `--enable-flashinfer-autotune`: L1476-L1478.
- `--linear-backend`: L1483-L1485.
- `--gdn-prefill-backend`: L1570-L1575.

Already traced material runtime change:

- GDN selector unchanged.
- GDN metadata builder changed materially in v0.24.0:
  prefill-only metadata for mixed decode+prefill, async host-to-device metadata
  copy, and request-count-based cudagraph metadata sizing.

TTFT/TPOT implication:
- If `--gdn-prefill-backend=flashinfer` is already set in both versions,
  v0.24 gain is not from selector change.
- It can come from lower metadata overhead and fewer bad mixed-batch metadata
  shapes in GDN-heavy prefill.

---

## 8. Updated v0.24.0 tuning conclusions

If tuning v0.24.0 specifically:

1. Keep Qwen3.5 on `--mamba-cache-mode=align`; `all` is still unsupported.
2. Treat `--watermark` as a KV headroom/preemption knob:
   start at default `0.0`, increase only if logs/trace show preemption or
   KV thrash.
3. Treat `--prefill-schedule-interval` as DP/MoE balancing:
   useful for smoothing decode under multi-rank prefill waves, risky for TTFT.
4. Keep existing scheduler budget reasoning:
   `max_num_batched_tokens`, `max_num_seqs`, partial-prefill counts, and
   `long_prefill_token_threshold` retain the same core semantics.
5. Do not credit v0.24 gains to DBO, cudagraph default, GDN selector, or
   `mamba-cache-mode=all` unless the actual compose changed those flags.
6. Most likely v0.24 improvements come from runtime scheduler changes plus GDN
   metadata path changes, not from old flag meaning changing.
