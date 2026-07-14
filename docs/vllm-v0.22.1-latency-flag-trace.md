# vLLM v0.22.1 Latency + Flag Trace for Qwen3.5 Hybrid

Muc tieu cua file nay:
- bo tach TTFT / TPOT thanh cac component nho hon muc `queue + prefill + residual`
- noi tung component do voi code path that su trong image `duc0811/qwen35-2b-race:v1`
- lap `flag -> resolve path -> runtime consumer -> latency component -> cach tune`

Nguon trace:
- image local: `duc0811/qwen35-2b-race:v1`
- base tag trong image: `vllm/vllm-openai:v0.22.1`
- source da copy tu image: `/private/tmp/vllm_0_22_1_src`
- line-by-line deep trace: `docs/vllm-v0.22.1-line-by-line-traces.md`
- v0.24.0 overlay diff: `docs/vllm-v0.24.0-vs-v0.22.1-line-by-line-diff.md`

Quy uoc:
- `EXACT`: da thay trong code implementation cua image.
- `INFERRED`: duoc suy ra tu request flow / serving flow, khong co timestamp san.

---

## 1. TTFT / TPOT sau khi bo tach sau hon

### 1.1 TTFT phia client

```text
TTFT_client
= http_in
+ request_validation
+ frontend_prep
+ enqueue_handoff
+ queue_wait
+ prefill_runtime
+ first_output_build
+ first_chunk_flush
+ client_parse
```

### 1.2 TPOT phia client

```text
TPOT_client
= next_schedule_wait
+ decode_step_runtime
+ stream_buffering
+ output_build
+ chunk_flush
+ client_parse_jitter
```

### 1.3 Mapping sang cac cum lon trong repo

```text
frontend_prep
= model_check
+ chat_template_merge
+ chat_render
+ tokenization
+ multimodal_preprocess
+ sampling_param_build
+ trace_header_build
+ request_logging

queue_wait
= scheduler_tick_wait
+ admission_wait
+ kv_capacity_wait
+ fairness_wait

prefill_runtime
= prefix_hash_build
+ cache_lookup
+ chunk_split
+ prefill_attention_layers
+ prefill_gdn_layers
+ kv_write
+ mamba_state_write_or_copy

decode_step_runtime
= scheduler_overhead
+ graph_or_eager_launch
+ decode_attention_layers
+ decode_gdn_layers
+ lm_head_sampling
+ speculative_verify_or_reject
```

---

## 2. Component nho hon cua TTFT

Bang nay tra loi cau hoi "residual time con co gi nua?".

| Component | Code path chinh | Loai | EXACT/INFERRED | Flag lien quan |
|---|---|---|---|---|
| `http_in` | FastAPI / Starlette route handling truoc `create_chat_completion()` | residual | INFERRED | `disable-uvicorn-access-log`, `uvicorn-log-level`, `h11-*` |
| `request_validation` | Pydantic request parse truoc handler | residual | INFERRED | khong co lever perf rieng ro rang trong nhom 50 |
| `model_check` | `chat_completion/serving.py::render_chat_request -> _check_model` | frontend | EXACT | `served-model-name`, LoRA args |
| `chat_template_merge` | `engine/serving.py::_validate_chat_template`, `_prepare_extra_chat_template_kwargs`, `chat_completion/serving.py::_effective_chat_template_kwargs` | frontend | EXACT | `chat-template`, `chat-template-content-format`, `trust-request-chat-template`, `default-chat-template-kwargs` |
| `chat_render` | `openai_serving_render.render_chat(request)` | frontend | EXACT | `chat-template*`, `response-role` |
| `tokenization` | `renderers/base.py::_tokenize_singleton_prompt(_async)` | frontend | EXACT | `tokenizer`, `tokenizer-mode`, `skip-tokenizer-init`, `renderer-num-workers` |
| `multimodal_preprocess` | `renderers/base.py::_process_multimodal(_async)` | frontend | EXACT | `language-model-only`, `skip-mm-profiling`, `limit-mm-per-prompt`, `renderer-num-workers` |
| `sampling_param_build` | `chat_completion/serving.py::_create_chat_completion -> request.to_sampling_params` | frontend | EXACT | `max-model-len`, generation config flags |
| `trace_header_build` | `chat_completion/serving.py::_get_trace_headers` | frontend | EXACT | tracing flags |
| `request_logging` | `engine/serving.py::_log_inputs -> RequestLogger.log_inputs` | frontend | EXACT | `enable-log-requests`, `max-log-len` |
| `enqueue_handoff` | `engine_client.generate(...)` | handoff | EXACT | tat ca scheduler/cache/compute flags |
| `scheduler_tick_wait` | doi engine den luc request duoc scheduler xet | queue | INFERRED | `async-scheduling`, `max-num-seqs` |
| `admission_wait` | `v1/core/sched/scheduler.py` + `allocate_slots(... full_sequence_must_fit=...)` | queue | EXACT | `scheduler-reserve-full-isl`, `gpu-memory-utilization`, `kv-cache-memory-bytes` |
| `fairness_wait` | long prompt split / partial prefill admission | queue | EXACT | `enable-chunked-prefill`, `max-num-partial-prefills`, `max-long-partial-prefills`, `long-prefill-token-threshold`, `scheduling-policy` |
| `prefix_hash_build` | `v1/engine/core.py` + `resolve_kv_cache_block_sizes` + request block hasher | prefill | EXACT | `enable-prefix-caching`, `prefix-caching-hash-algo`, `block-size`, `mamba-cache-mode` |
| `cache_lookup` | `v1/core/block_pool.py`, `kv_cache_manager.py` | prefill | EXACT | `enable-prefix-caching`, `hash_block_size`, `disable-hybrid-kv-cache-manager` |
| `chunk_split` | `scheduler.py` threshold logic | prefill | EXACT | `enable-chunked-prefill`, `long-prefill-token-threshold` |
| `prefill_attention_layers` | full-attention backend path | prefill | EXACT | `attention-backend`, `kv-cache-dtype`, `calculate-kv-scales` |
| `prefill_gdn_layers` | `qwen_gdn_linear_attn.py`, `gdn_attn.py` | prefill | EXACT | `gdn-prefill-backend`, `mamba-cache-mode`, `quantization`, `linear-backend` |
| `kv_write` | attention layer KV cache update | prefill | EXACT | `kv-cache-dtype`, `kv-cache-dtype-skip-layers`, `calculate-kv-scales` |
| `mamba_state_write_or_copy` | `platforms/interface.py`, `mamba_utils.py`, `gpu_model_runner.py` | prefill | EXACT | `mamba-cache-mode`, `mamba-block-size`, `mamba-cache-dtype`, `mamba-ssm-cache-dtype` |
| `first_output_build` | `output_processor.py::make_request_output` + OpenAI serving usage build | output | EXACT | `stream-interval`, `enable-prompt-tokens-details`, `enable-force-include-usage` |
| `first_chunk_flush` | SSE / StreamingResponse flush | transport | INFERRED | `stream-interval`, `disable-uvicorn-access-log` |
| `client_parse` | benchmark client doc chunk parse | transport | INFERRED | ngoai server |

### 2.1 Component nho hon cua TPOT

| Component | Code path chinh | Loai | EXACT/INFERRED | Flag lien quan |
|---|---|---|---|---|
| `next_schedule_wait` | scheduler loop tick sau token truoc | queue | INFERRED | `async-scheduling`, `max-num-seqs` |
| `scheduler_overhead` | `Scheduler` / `AsyncScheduler` lap batch moi | decode | EXACT | `async-scheduling`, `max-num-seqs`, `max-num-batched-tokens` |
| `graph_or_eager_launch` | `config/vllm.py::_set_cudagraph_sizes`, `gpu_model_runner` | decode | EXACT | `enforce-eager`, `cudagraph-capture-sizes`, `max-cudagraph-capture-size`, `performance-mode` |
| `decode_attention_layers` | attention backend decode kernels | decode | EXACT | `attention-backend`, `disable-cascade-attn`, `kv-cache-dtype` |
| `decode_gdn_layers` | GDN/mamba decode path | decode | EXACT | `mamba-cache-mode`, `mamba-cache-dtype`, `mamba-ssm-cache-dtype` |
| `lm_head_sampling` | logits processor + sampler | decode | INFERRED | `quantization`, `linear-backend` |
| `speculative_verify_or_reject` | rejection sampler / spec decode placeholders | decode | EXACT | `spec-method`, `spec-tokens`, `speculative-config` |
| `stream_buffering` | `output_processor.py::make_request_output` | output | EXACT | `stream-interval` |
| `output_build` | OpenAI delta / usage / logging build | output | EXACT | `enable-log-outputs`, `enable-log-deltas`, `enable-prompt-tokens-details` |
| `chunk_flush` | server stream send | transport | INFERRED | `stream-interval` |
| `client_parse_jitter` | benchmark client doc SSE parse | transport | INFERRED | ngoai server |

---

## 3. Nhung ket luan implementation quan trong nhat

### 3.1 Prefix caching hybrid khong phai duong mac dinh

`EXACT`

- `engine/arg_utils.py::_set_default_chunked_prefill_and_prefix_caching_args`
- `config/model.py::is_prefix_caching_supported`

Voi `attn_type == "hybrid"`, default support tra ve `False`, nen neu khong ep
`--enable-prefix-caching` thi attention prefix cache khong tu bat.

### 3.2 `mamba-cache-mode` thay doi block layout, hash granularity, va state-copy path

`EXACT`

- `platforms/interface.py::_align_hybrid_block_size`
- `v1/core/kv_cache_utils.py::resolve_kv_cache_block_sizes`
- `v1/attention/backends/utils.py::mamba_get_block_table_tensor`
- `v1/worker/gpu_model_runner.py`
- `v1/worker/mamba_utils.py`

`all`:
- co the tang `cache_config.block_size`
- `mamba_block_size` duoc canh theo `chunk/kernel alignment`
- prefix hash co the mat granularity min hon vi block mamba va attn lech nhau

`align`:
- ep `mamba_block_size = cache_config.block_size`
- co preprocess/postprocess state-copy path rieng

`none`:
- khong co Mamba prefix-state cache theo block

### 3.3 `performance-mode=throughput` chi nhan doi default, khong override gia tri user set tay

`EXACT`

- `engine/arg_utils.py::_set_default_max_num_seqs_and_batched_tokens_args`

Neu submit file da set tay `max-num-batched-tokens` va `max-num-seqs`,
`performance-mode` khong con tac dung "preset all-in-one" nua.

### 3.4 `kv-sharing-fast-prefill` hien tai chua phai lever perf that

`EXACT`

- `config/cache.py` comment trong class
- `config/vllm.py` con canh bao them ve correctness

Code ghi ro hien tai chua co prefill optimization that su.

### 3.5 `renderer-num-workers` la lever frontend CPU that su

`EXACT`

- `renderers/base.py::__init__`
- `tokenizers/registry.py`

No tao `ThreadPoolExecutor(max_workers=renderer_num_workers)` de offload:
- tokenizer encode
- chat render
- multimodal preprocess

---

## 4. Trace matrix cho toan bo nhom co y nghia

Cot:
- `Resolve`: noi CLI di vao config/runtime object.
- `Consume`: noi runtime su dung that su.
- `Latency`: startup / frontend / queue / prefill / decode / output.
- `Qwen3.5 tune`: cach doc flag trong boi canh hybrid 6 attn + 18 GDN.

### 4.1 Frontend / API / residual CPU

| Flag | Resolve | Consume | Latency | Qwen3.5 tune |
|---|---|---|---|---|
| `chat-template` | OpenAI args | `api_server.py`, `chat_completion/serving.py`, renderer | frontend | Neu template phuc tap, TTFT residual tang that |
| `chat-template-content-format` | OpenAI args | `responses_parser.py`, `chat_completion/serving.py` | frontend | Sai format co the tang parse/render work |
| `trust-request-chat-template` | OpenAI args | `engine/serving.py` check request template | frontend | Perf khong la lever chinh, chu yeu safety/path choice |
| `default-chat-template-kwargs` | OpenAI args | `engine/serving.py`, `chat_completion/serving.py` | frontend | Doi render behavior, co the them CPU work |
| `response-role` | OpenAI args | `chat_completion/serving.py` response builder | output | Giam/tang perf khong dang ke |
| `tokenizer` | `create_model_config -> ModelConfig.tokenizer` | `tokenizers/registry.py::cached_tokenizer_from_config` | frontend | Doi tokenizer source co the doi TTFT residual |
| `tokenizer-mode` | `ModelConfig.tokenizer_mode` | `config/model.py` auto resolve, `tokenizers/registry.py` load cls | frontend | `slow` la anti-perf ro rang |
| `skip-tokenizer-init` | `ModelConfig.skip_tokenizer_init` | `tokenizers/registry.py`, `renderers/base.py`, serving errors | frontend | Chi dung khi request dua token ids san |
| `renderer-num-workers` | `ModelConfig.renderer_num_workers` | `renderers/base.py` thread pool | frontend | La lever residual CPU hop le nhat |
| `enable-log-requests` | API args | `api_server.py` tao `RequestLogger` | frontend | Nen tat khi bench |
| `max-log-len` | API args | `entrypoints/logger.py` cat input/output truoc log | frontend/output | Chi co nghia khi bat log |
| `enable-log-outputs` | API args | chat/responses serving log output | output | Tat khi bench |
| `enable-log-deltas` | API args | chat serving chi log delta parts | output | Chi giam log volume, khong giai quyet goc |
| `disable-uvicorn-access-log` | API args | `serve_http(... access_log=...)` | residual | Nen tat khi bench |
| `uvicorn-log-level` | API args | `serve_http(log_level=...)` | residual | `debug`/`trace` co them log IO |
| `enable-server-load-tracking` | API args | `entrypoints/utils.py::load_aware_call` | residual | Them counter update moi request |
| `enable-prompt-tokens-details` | API args | chat/completion serving them `cached_tokens` usage | output | Nen bat khi can debug prefix hit |
| `enable-force-include-usage` | API args | `entrypoints/utils.py::should_include_usage` | output | Huu ich do bench, khong phai perf lever |
| `stream-interval` | `SchedulerConfig.stream_interval` | `output_processor.py::make_request_output` | TPOT/output | >1 se lam token gap theo cum |

### 4.2 Startup / model / multimodal

| Flag | Resolve | Consume | Latency | Qwen3.5 tune |
|---|---|---|---|---|
| `model` | `create_model_config -> ModelConfig.model` | model load / HF config / weights | startup+all | model path local trong submit |
| `served-model-name` | `ModelConfig.served_model_name` | `_check_model`, response `model` field | frontend | Chu yeu dung API target cho dung |
| `max-model-len` | `ModelConfig.max_model_len` | `SchedulerConfig.verify_max_model_len`, `get_max_tokens`, mm budgets | startup+queue | Rat quan trong, de qua cao se lam du tru vo ich |
| `load-format` | `create_load_config -> LoadConfig.load_format` | default loader | startup | `auto`/`safetensors` la duong chinh |
| `safetensors-load-strategy` | `LoadConfig` | `default_loader.py` -> `weight_utils.py` | startup | `prefetch` co co che thuc su |
| `safetensors-prefetch-num-threads` | `LoadConfig` | `_prefetch_all_checkpoints(...)` | startup | Chi co y nghia khi prefetch bat |
| `safetensors-prefetch-block-size` | `LoadConfig` | `_prefetch_all_checkpoints(...)` | startup | Tinh IO pattern luc warmup load |
| `language-model-only` | `ModelConfig -> MultiModalConfig.language_model_only` | `multimodal.py::get_limit_per_prompt` | startup+frontend | Text-only nen rat hop ly |
| `skip-mm-profiling` | `MultiModalConfig.skip_mm_profiling` | multimodal startup path | startup | Giam startup neu khong dung image |
| `limit-mm-per-prompt` | `MultiModalConfig.limit_per_prompt` | `multimodal.py::get_limit_per_prompt`, processing context | frontend | Set `image=0` de khoa hinh anh |

### 4.3 Scheduler / queue / batching

| Flag | Resolve | Consume | Latency | Qwen3.5 tune |
|---|---|---|---|---|
| `max-num-batched-tokens` | `EngineArgs._set_default_max_num_seqs...` | `SchedulerConfig`, warmup, scheduler token budget | queue+prefill | Lever TTFT lon nhat |
| `max-num-seqs` | same | `SchedulerConfig`, `gpu_model_runner.max_num_reqs` | queue+decode | Trade-off truc tiep voi TPOT |
| `enable-chunked-prefill` | default tu `ModelConfig.is_chunked_prefill_supported` | scheduler chunk logic | queue+prefill | Mac dinh generate support = True |
| `max-num-partial-prefills` | `SchedulerConfig` | scheduler concurrent partial prefill logic | queue | Chi dung sau khi da on dinh crash/rui ro |
| `max-long-partial-prefills` | `SchedulerConfig` | scheduler fairness logic | queue | Cho short prompt nhay len truoc long prompt |
| `long-prefill-token-threshold` | `SchedulerConfig` | scheduler split `num_new_tokens` | queue+prefill | Danh dau prompt "dai" |
| `scheduling-policy` | `SchedulerConfig.policy` | scheduler request ordering | queue | `fcfs` phu hop trace nhat |
| `scheduler-reserve-full-isl` | `SchedulerConfig.scheduler_reserve_full_isl` | `allocate_slots(... full_sequence_must_fit=...)` | queue | Giam over-admission, co the tang cho doi |
| `async-scheduling` | `SchedulerConfig.async_scheduling` | `get_scheduler_cls -> AsyncScheduler` | queue+decode | Thuong la free win nho |
| `disable-hybrid-kv-cache-manager` | `SchedulerConfig.disable_hybrid_kv_cache_manager` | `config/vllm.py` auto force / `kv_cache_utils.py::get_kv_cache_groups` | queue+cache | Worth A/B neu nghi HMA gay regression |
| `performance-mode` | `VllmConfig.performance_mode` | batch default + cudagraph sizing | queue+decode | Khong overwrite gia tri user set tay |

### 4.4 Cache / prefix / hybrid state

| Flag | Resolve | Consume | Latency | Qwen3.5 tune |
|---|---|---|---|---|
| `gpu-memory-utilization` | `CacheConfig.gpu_memory_utilization` | KV sizing / worker memory profiling | startup+queue | Lever suc chua, khong phai compute win truc tiep |
| `kv-cache-memory-bytes` | `CacheConfig.kv_cache_memory_bytes` | KV sizing override | startup+queue | Override truc tiep tot hon util khi can reproducible |
| `block-size` | `CacheConfig.block_size` | backend block align, block hash, scheduler invariant | queue+prefill | Hybrid co the bi ep tang them |
| `enable-prefix-caching` | default resolution -> `CacheConfig.enable_prefix_caching` | engine core / block pool / request hasher | prefill | Can bat tay tren hybrid |
| `prefix-caching-hash-algo` | `CacheConfig.prefix_caching_hash_algo` | `v1/engine/core.py` hash fn choice | prefill CPU | `xxhash` nhanh hon, `sha256` an toan hon |
| `kv-cache-dtype` | `resolve_kv_cache_dtype_string -> CacheConfig.cache_dtype` | attention layers / kv specs / worker dtype | prefill+decode | Doi bo nho va kernel path that su |
| `calculate-kv-scales` | `CacheConfig.calculate_kv_scales` | attention layer KV quant scale logic | prefill+startup | Da deprecated; hybrid con co path tat no |
| `kv-cache-dtype-skip-layers` | `CacheConfig` | attention layer per-layer skip logic | prefill+decode | Huu ich khi chi muon quant mot phan layer |
| `kv-sharing-fast-prefill` | `CacheConfig` | model metadata override / warnings | almost none | Hien tai khong nen ky vong perf |
| `mamba-block-size` | `CacheConfig.mamba_block_size` | hybrid block alignment | prefill+cache | Rat nhay voi hash granularity |
| `mamba-cache-mode` | `CacheConfig.mamba_cache_mode` | platform align + kv utils + runner + mamba utils | prefill+decode | Lever hybrid quan trong nhat |
| `mamba-cache-dtype` | `CacheConfig.mamba_cache_dtype` | mamba state dtype resolution | prefill+decode | Memory win nho hon so voi weight quant |
| `mamba-ssm-cache-dtype` | `CacheConfig.mamba_ssm_cache_dtype` | Qwen3.5/Nemotron config updates + mamba utils | prefill+decode | Can than accuracy/state dynamics |

### 4.5 Compute / kernels / graph / speculative

| Flag | Resolve | Consume | Latency | Qwen3.5 tune |
|---|---|---|---|---|
| `quantization` | `resolve_quantization_config`, `ModelConfig._verify_quantization` | model loader + layer quant methods | prefill+decode | Weight quant la lever compute lon nhat |
| `quantization-config` | `resolve_quantization_config` | online quant / per-layer override | prefill+decode | Dung khi can override chi tiet |
| `attention-backend` | arg override -> `AttentionConfig.backend` | runner build attn groups / backend builders | prefill+decode | Chi tac dong 6 full-attn layers |
| `gdn-prefill-backend` | `additional_config["gdn_prefill_backend"]` | `_resolve_gdn_prefill_backend`, `ChunkGatedDeltaRule` | prefill | Tac dong 18 GDN layers, rat dang ke |
| `linear-backend` | arg override -> `KernelConfig.linear_backend` | `model_executor/kernels/linear/__init__.py` filter kernels | prefill+decode | Rat quan trong sau quantization |
| `enable-flashinfer-autotune` | `KernelConfig.enable_flashinfer_autotune` | `kernel_warmup.py::flashinfer_autotune` | startup+compute | Trade startup de lay kernel tot hon |
| `disable-cascade-attn` | `ModelConfig.disable_cascade_attn` | `gpu_model_runner.cascade_attn_enabled` | decode+prefill | Mac dinh dang tat |
| `enforce-eager` | `ModelConfig.enforce_eager` | `config/vllm.py` tat cudagraph | decode+startup | Bat = thuong xau TPOT |
| `cudagraph-capture-sizes` | `CompilationConfig.cudagraph_capture_sizes` | `config/vllm.py::_set_cudagraph_sizes`, runner | decode+startup | Bao phu dung batch size thuc te |
| `max-cudagraph-capture-size` | `CompilationConfig.max_cudagraph_capture_size` | same | decode+startup | Cap cho decode graph |
| `spec-method` | `create_speculative_config` | speculative config + runner + rejection sampler | decode | Dung khi TPOT chua dat tran |
| `spec-tokens` | `create_speculative_config` | same | decode | Tang placeholder / verify cost |

### 4.6 Observability / benchmark support

| Flag | Resolve | Consume | Latency | Qwen3.5 tune |
|---|---|---|---|---|
| `disable-log-stats` | app state | `api_server.py`, engine periodic logging | residual | Nen tat khi bench |
| `kv-cache-metrics` | observability config | metrics path | residual | Bat khi debug cache, tat khi do perf |
| `kv-cache-metrics-sample` | observability config | metrics sampler | residual | Trade detail/overhead |
| `collect-detailed-traces` | observability config | tracing exporters | residual | Chi dung luc mo xac |

---

## 5. Runtime logic nho nhat cua nhung lever quan trong

### 5.1 `max-num-batched-tokens`

`EXACT`

Resolve:
- `engine/arg_utils.py::get_batch_defaults`
- `engine/arg_utils.py::_set_default_max_num_seqs_and_batched_tokens_args`

Consume:
- `config/scheduler.py`
- `model_executor/warmup/kernel_warmup.py`
- `config/vllm.py::_set_cudagraph_sizes`

Anh huong:
- TTFT: giam so lan prefill phai bi chia iteration
- TPOT: batch decode to hon, co the tang mixed-prefill penalty

### 5.2 `max-num-seqs`

`EXACT`

Resolve:
- same nhu tren

Consume:
- `SchedulerConfig.max_num_seqs`
- `gpu_model_runner.max_num_reqs`
- `config/vllm.py::_set_cudagraph_sizes`

Anh huong:
- queue_wait giam khi co nhieu user song song
- decode batch width tang -> TPOT co the xau di

### 5.3 `enable-chunked-prefill` + partial prefills

`EXACT`

Resolve:
- default tu `ModelConfig.is_chunked_prefill_supported`

Consume:
- `scheduler.py`: neu prompt dai hon threshold thi cat `num_new_tokens`
- `config/scheduler.py`: validate `max_num_partial_prefills`, `max_long_partial_prefills`

Anh huong:
- queue/fairness path, khong phai chi "bat chunk" chung chung
- co the giup TTFT p95 khi prompt dai tranh GPU voi decode

### 5.4 `scheduler-reserve-full-isl`

`EXACT`

Consume:
- `v1/core/sched/scheduler.py::allocate_slots(... full_sequence_must_fit=...)`

Anh huong:
- bat: admission can than hon, tranh thrash KV
- tat: co the nhan request vao som hon, nhung de bi over-admission

### 5.5 `mamba-cache-mode`

`EXACT`

Consume:
- `platforms/interface.py` canh block
- `kv_cache_utils.py` chon `hash_block_size`
- `attention/backends/utils.py` bien doi block table
- `mamba_utils.py` preprocess/postprocess align path
- `gpu_model_runner.py` spec decode handling cho `all`

Anh huong:
- TTFT vong sau sau prefix hit phu thuoc rat manh vao mode nay
- day la lever hybrid-specific, khong the doc nhu model attention thuong

### 5.6 `gdn-prefill-backend`

`EXACT`

Consume:
- `qwen_gdn_linear_attn.py::_resolve_gdn_prefill_backend`
- `ChunkGatedDeltaRule.__init__`
- `v1/attention/backends/gdn_attn.py`

Logic:
- `auto/flashinfer` chi vao FlashInfer neu dung platform / capability
- neu khong thi roi ve `triton`
- `cutedsl` chi hop mot so Blackwell path

Anh huong:
- vi 18/24 layer cua Qwen3.5 la GDN, day la prefill lever lon

### 5.7 `attention-backend`

`EXACT`

Consume:
- `engine/arg_utils.py` override `AttentionConfig.backend`
- `gpu_model_runner.py` build attn groups theo backend tung layer

Anh huong:
- chu yeu len 6 full-attention layer
- khong duoc nham no voi GDN backend

### 5.8 `linear-backend`

`EXACT`

Consume:
- `KernelConfig.linear_backend`
- `model_executor/kernels/linear/__init__.py`

Logic:
- neu `auto`, vLLM tu loc kernel hop hardware/layer type
- neu chi dinh tay, no filter tap kernel; khong co kernel hop le se error

Anh huong:
- dac biet quan trong sau khi dung quantized weights

### 5.9 `enforce-eager` + `cudagraph-*`

`EXACT`

Consume:
- `config/vllm.py`

Logic:
- neu `enforce_eager=True`, cudagraph mode -> `NONE`
- `max_cudagraph_capture_size = 0`
- `cudagraph_capture_sizes = []`
- nguoc lai, vLLM tu sinh list capture size dua tren `max_num_seqs`,
  `spec_tokens`, `max_num_batched_tokens`, `performance_mode`

Anh huong:
- lever TPOT decode rat ro

### 5.10 `quantization`

`EXACT`

Resolve:
- `config/model.py::_verify_quantization`
- `config/quantization.py::resolve_quantization_config`

Consume:
- `model_loader/weight_utils.py::get_quant_config`
- tung layer lay `quant_method` rieng

Anh huong:
- compute prefill/decode thay doi that su
- la lever lon hon `kv-cache-dtype` neu muc tieu la giam compute GEMM

---

## 6. Tuning map theo tung component nho

### 6.1 Neu `frontend_prep` cao

Uu tien:
- `renderer-num-workers`
- `tokenizer-mode=hf`
- tat `enable-log-requests`
- tat `enable-log-outputs`
- `language-model-only`
- `limit-mm-per-prompt image=0`

### 6.2 Neu `queue_wait` cao

Uu tien:
- tang `max-num-seqs`
- tang `max-num-batched-tokens`
- giu `enable-chunked-prefill`
- can than `scheduler-reserve-full-isl`
- chi A/B `max-num-partial-prefills` sau khi da chac on dinh

### 6.3 Neu `prefill_runtime` cao

Uu tien:
- `enable-prefix-caching`
- `mamba-cache-mode`
- `gdn-prefill-backend`
- `quantization`
- `linear-backend`
- `attention-backend`

### 6.4 Neu `decode_step_runtime` cao

Uu tien:
- khong bat `enforce-eager`
- chinh `cudagraph-capture-sizes` / `max-cudagraph-capture-size`
- giam `max-num-seqs` neu dang qua tham
- chi dung `spec-method` khi TPOT chua duoi nguong diem

### 6.5 Neu `output/transport` cao

Uu tien:
- giu `stream-interval=1` neu metric tinh theo client token gap
- tat log outputs / access log
- dung `enable-prompt-tokens-details` chi khi can debug

---

## 7. Qwen3.5-specific takeaways

1. Qwen3.5 hybrid khong nen duoc doc nhu model full-attention thuong.
   `attention-backend` chi phu 6 layer; `gdn-prefill-backend` va
   `mamba-cache-mode` moi la hybrid levers thuc chat.

2. `mamba-cache-mode=all` khong chi la "cache nhieu hon".
   No co the doi `block_size`, doi `hash_block_size`, va doi ca state path.

3. Prefix caching round 2+ co the van "khong an manh" du da bat flag, vi:
   - hybrid default support = false
   - Mamba block co the lech attn block
   - `hash_block_size` co the bi fallback len scheduler block size

4. `performance-mode=throughput` chi con y nghia khi ban chua set tay batch/seqs.

5. `kv-sharing-fast-prefill` khong nen dua vao shortlist toi uu hien tai.

---

## 8. Thu tu uu tien tune thuc dung

Neu muc tieu la diem Track 3 cho Qwen3.5 hybrid, thu tu uu tien nen la:

1. do tach `frontend_prep` vs `queue_wait` vs `prefill_runtime`
2. khoa text-only path: `language-model-only`, `skip-mm-profiling`, `limit-mm-per-prompt`
3. A/B `max-num-batched-tokens` va `max-num-seqs`
4. A/B `enable-prefix-caching` + `mamba-cache-mode`
5. A/B `gdn-prefill-backend`
6. neu can them compute win: `quantization` + `linear-backend`
7. sau cung moi den `cudagraph-*`, `spec-method`, partial prefill nang cao

---

## 9. Deep implementation audit: logic trace, khong can benchmark

Phan nay chi trace logic implementation. Khong su dung benchmark runtime.

### 9.1 Thu tu resolve config that su

`EXACT`

Entry flow cua `api_server`:

```text
make_arg_parser()
-> FrontendArgs.add_cli_args()
-> AsyncEngineArgs.add_cli_args()
-> EngineArgs.from_cli_args()
-> EngineArgs.__post_init__()
-> EngineArgs.create_engine_config()
```

Ben trong `create_engine_config()`:

```text
current_platform.pre_register_and_update()
-> DeviceConfig
-> maybe_override_with_speculators()
-> create_model_config()
-> _check_feature_supported()
-> _set_default_chunked_prefill_and_prefix_caching_args()
-> resolve_kv_cache_dtype_string()
-> CacheConfig(...)
-> ParallelConfig(...)
-> create_speculative_config()
-> _set_default_max_num_seqs_and_batched_tokens_args()
-> SchedulerConfig(...)
-> attention_config override
-> mamba_config override
-> kernel_config override
-> create_load_config()
-> ObservabilityConfig(...)
-> additional_config["gdn_prefill_backend"]
-> VllmConfig(...)
```

Sau khi `VllmConfig(...)` duoc tao, `VllmConfig.__post_init__()` tiep tuc rewrite:

```text
current_platform.apply_config_platform_defaults()
-> kernel_config.set_platform_defaults()
-> _apply_optimization_level_defaults()
-> cudagraph compatibility rewrite
-> enforce_eager rewrite
-> _set_cudagraph_sizes()
-> kv_sharing_fast_prefill warning/compat
-> model/config specific verify_and_update_config()
-> hybrid kv cache manager auto decision
-> validate_block_size()
-> validate_mamba_block_size()
```

He qua:
- Nhieu flag khong chi "set xong la xong"; no co the bi rewrite sau khi
  `VllmConfig` biet model architecture, platform, compilation mode, va cache groups.
- Cac flag hay bi hieu sai nhat la `performance-mode`, `mamba-cache-mode`,
  `kv-cache-dtype`, `enforce-eager`, `cudagraph-*`, `disable-hybrid-kv-cache-manager`.

---

## 10. Submit-specific flag trace

Day la nhom flag dang xuat hien trong cac file submit hien tai:
- `docker-compose.submit-v25.yml`
- `docker-compose.submit-v26.yml`
- `docker-compose.submit-v28.yml`
- `docker-compose.submit-v30.yml`

### 10.1 `--model=/model`

`EXACT`

Resolve:
- `EngineArgs.create_model_config()`
- `ModelConfig.__post_init__()`
- `get_config(self.hf_config_path or self.model, ...)`
- `registry.inspect_model_cls(architectures, self)`

Logic:
- Neu `--tokenizer` khong set, tokenizer mac dinh bang `model`.
- `maybe_model_redirect()` co the rewrite model/tokenizer path.
- Architecture resolve xong moi biet model la text, multimodal, hybrid, attention-free.

Latency components:
- startup: HF config load, model class inspect, weight path
- frontend: tokenizer path
- prefill/decode: architecture quyet dinh backend/layer path

Qwen3.5 note:
- `model=/model` la nguon de detect `Qwen3_5ForConditionalGenerationConfig`,
  hybrid attn type, GDN layers, mamba state shape/dtype.

### 10.2 `--served-model-name=Qwen3.5-2B`

`EXACT`

Resolve:
- `ModelConfig.served_model_name`
- `get_served_model_name(model, served_model_name)`

Consume:
- OpenAI serving `_check_model(...)`
- response `model` field
- metrics label `model_name`

Logic:
- Neu list nhieu ten, response/metrics dung ten dau tien.
- Khong doi engine compute.

Latency components:
- `model_check`: tiny frontend CPU
- observability label only

### 10.3 `--host`, `--port`

`EXACT`

Resolve:
- OpenAI API args

Consume:
- `api_server.py::serve_http(host=args.host, port=args.port, ...)`

Logic:
- Khong vao `VllmConfig`.
- Chi anh huong listen socket.

Latency components:
- network accept path only

### 10.4 `--max-model-len`

`EXACT`

Resolve:
- `ModelConfig.max_model_len`
- `ModelConfig.get_and_verify_max_len(...)`
- `SchedulerConfig(max_model_len=...)`

Consume:
- `SchedulerConfig.verify_max_model_len(max_model_len)`
- OpenAI `get_max_tokens(max_model_len, input_length, ...)`
- multimodal budget: `encoder_budget.py`, processor dummy inputs
- KV memory check: `_check_enough_kv_cache_memory(...)`
- cudagraph sizing indirectly via `max_num_batched_tokens` and scheduling limits

Validation/rewrite:
- Neu unset, model config max len duoc derive tu HF config/tokenizer config.
- Neu chunked prefill tat va `max_num_batched_tokens < max_model_len`, raise.
- Neu KV memory khong du cho at least one max-length request, raise va goi y
  giam `max_model_len` hoac tang memory.

Latency components:
- startup: KV sizing / dummy profile / MM dummy generation
- queue: admission and max sequence validity
- frontend: request max_tokens cap

Qwen3.5 tune:
- `48000` vs `32768` khong chi la "context max"; no doi KV feasibility,
  concurrency estimate, admission pressure, and MM dummy budgets.

### 10.5 `--gpu-memory-utilization`

`EXACT`

Resolve:
- `CacheConfig.gpu_memory_utilization`

Consume:
- `gpu_worker.py::init_device()` via `request_memory(init_snapshot, cache_config)`
- `gpu_worker.py::determine_available_memory()`
- KV cache sizing utilities

Logic:
- vLLM lay memory snapshot sau distributed init.
- Requested memory = fraction cua total/free memory theo platform helper.
- Available KV memory = requested memory - weights - activation/profile - optional cudagraph estimate.
- Neu `kv_cache_memory_bytes` set, flag nay bi bo qua cho KV size.

Latency components:
- startup: memory profiling
- queue: KV pool capacity

Qwen3.5 tune:
- Tang util co the tang KV capacity, nhung neu cudagraph/activation/mamba state
  lon thi van co OOM risk. No khong tang raw kernel speed.

### 10.6 `--tensor-parallel-size=1`

`EXACT`

Resolve:
- `ParallelConfig.tensor_parallel_size`

Consume:
- world size defaults for batching
- model runner parallel groups
- attention heads/KV heads per rank
- cudagraph sequence-parallel rewrite disabled when TP=1

Logic:
- `world_size = pipeline_parallel_size * tensor_parallel_size`.
- `get_batch_defaults(world_size)` dung world size cho CPU defaults; GPU OpenAI defaults khong nhan theo TP trong fast path nay.
- `pass_config.enable_sp` bi disable neu TP=1.

Latency components:
- startup: distributed init
- prefill/decode: per-rank compute shape

Qwen3.5 tune:
- Single GPU thi TP=1 la dung; cac flag sequence-parallel/async TP khong con la lever.

### 10.7 `--enable-prefix-caching`

`EXACT`

Resolve:
- `EngineArgs.enable_prefix_caching`
- `_set_default_chunked_prefill_and_prefix_caching_args(model_config)`
- `CacheConfig.enable_prefix_caching`

Default logic:
- Neu user khong set, vLLM lay `model_config.is_prefix_caching_supported`.
- Voi generative hybrid, `is_prefix_caching_supported` tra `False`.
- User ep `--enable-prefix-caching` thi co the bat du hybrid default false.

Consume:
- `v1/engine/core.py`: chon caching hash fn
- `v1/core/kv_cache_utils.py`: `hash_block_size`
- `v1/core/block_pool.py`: cache lookup by block hashes
- `v1/core/kv_cache_manager.py`: cached block admission

Latency components:
- prefill: cache hit, hash build, lookup
- queue: fewer tokens can reduce scheduled prefill work

Qwen3.5 caveat:
- Chi bat flag chua dam bao "full hybrid cache"; `mamba-cache-mode` va block/hash
  alignment quyet dinh Mamba/GDN state co duoc reuse min khong.

### 10.8 `--language-model-only`

`EXACT`

Resolve:
- `ModelConfig(... language_model_only=...)`
- `MultiModalConfig.language_model_only`

Consume:
- `MultiModalConfig.get_limit_per_prompt(modality)`
- multimodal registry / processing context

Logic:
- Neu `language_model_only=True`, `get_limit_per_prompt()` tra `0` cho moi modality.
- Voi model support multimodal, van co `MultiModalConfig`, nhung input modality bi gioi han ve 0.

Latency components:
- startup: bot MM warm/profile path tuy model/config
- frontend: avoid image/video preprocess
- memory: avoid or reduce MM processor path

Qwen3.5 tune:
- Text-only race nen day la flag tot; no la semantic guard va perf guard.

### 10.9 `--default-chat-template-kwargs={"enable_thinking": false}`

`EXACT`

Resolve:
- OpenAI frontend args `default_chat_template_kwargs`
- json loaded in `cli_args.py`

Consume:
- `OpenAIServingChat.warmup()`
- `_effective_chat_template_kwargs()`
- `request.build_chat_params(...).with_defaults(...)`
- renderer `render_chat(...)`

Logic:
- Server defaults duoc merge vao request chat template kwargs.
- Request kwargs co the override server defaults neu duoc pass.
- Khong vao model graph; no doi prompt text/tokens truoc engine.

Latency components:
- frontend: chat render
- prefill: prompt token count can change
- decode: reasoning/thinking path co the change output behavior

Qwen3.5 tune:
- Day la flag co the anh huong diem rat lon neu no lam prompt/output ngan hon
  hoac tat reasoning branch. No khong phai "frontend tiny only"; no doi token stream.

### 10.10 `--quantization=fp8`

`EXACT`

Resolve:
- `EngineArgs.__post_init__()`: `resolve_quantization_config(...)`
- `ModelConfig._verify_quantization()`
- platform `verify_quantization(...)`

Consume:
- `model_loader/weight_utils.py::get_quant_config(...)`
- layer init: `quant_config.get_quant_method(...)`
- linear/MoE/attention layer quant methods
- `linear-backend` kernel filtering

Logic:
- Neu checkpoint da co `quantization_config`, vLLM verify CLI method khop config method.
- Neu `quantization` la online shorthand, `quantization_config` co the merge vao base config.
- Neu quantization la non-online method ma van pass `quantization-config`, raise.

Latency components:
- startup: quant config loading / possibly online transform
- prefill/decode: GEMM kernel path, weight memory bandwidth

Qwen3.5 tune:
- Weight FP8 la compute lever lon hon `kv-cache-dtype=fp8`.
- Can phan biet weight quantization voi KV cache quantization.

### 10.11 `--kv-cache-dtype=fp8`

`EXACT`

Resolve:
- `resolve_kv_cache_dtype_string(self.kv_cache_dtype, model_config)`
- `CacheConfig.cache_dtype`
- `CacheConfig._validate_cache_dtype`

Consume:
- attention layer init: `Attention(... cache_config.cache_dtype ...)`
- `FullAttentionSpec(... kv_quant_mode=...)`
- `platforms/interface.py::_align_hybrid_block_size`
- KV cache shape/layout and dtype

Logic:
- `auto` resolve theo model/cache scheme.
- Explicit `fp8` wins over checkpoint KV cache scheme.
- Attention layers may force `cache_config.cache_dtype = "fp8"` if quant config declares KV scheme and cache dtype is auto.
- `kv-cache-dtype-skip-layers` can revert selected attention layers to `auto`.
- Hybrid model config can disable `calculate_kv_scales` for some cases.

Latency components:
- prefill/decode: KV write/read bandwidth, quant/dequant overhead
- startup: KV block sizing/profile
- queue: KV capacity

Qwen3.5 caveat:
- With hybrid, FP8 KV changes attention page size; `_align_hybrid_block_size`
  can then change `block_size`, which changes hash granularity and scheduler block invariants.

### 10.12 `--calculate-kv-scales`

`EXACT`

Resolve:
- `CacheConfig.calculate_kv_scales`
- `CacheConfig._warn_deprecated_calculate_kv_scales`

Consume:
- `Attention.__init__`: stores `calculate_kv_scales`
- `model_executor/layers/quantization/kv_cache.py`: loads/calculates scales
- `model_executor/models/config.py`: can disable for hybrid models

Logic:
- Deprecated in code comment.
- Only meaningful for quantized KV cache variants that use scales.
- Per-token-head scale dtypes do not use loaded fixed scales in same way.
- If selected layer is skipped via `kv-cache-dtype-skip-layers`, scales disabled for that layer.

Latency components:
- startup / prefill tiny
- correctness/accuracy more than speed

Qwen3.5 tune:
- Current submit comments saying "inert" are plausible because hybrid-specific
  config path can disable it; verify by reading startup logs/config dump, not by timing.

### 10.13 `--max-num-seqs`

`EXACT`

Resolve:
- `_set_default_max_num_seqs_and_batched_tokens_args`
- `SchedulerConfig.max_num_seqs`

Consume:
- `gpu_model_runner.max_num_reqs`
- scheduler running/waiting admission
- cudagraph default max capture size: `min(max_num_seqs * decode_query_len * 2, 512)`

Validation:
- `max_num_batched_tokens >= max_num_seqs`, else raise.

Latency components:
- queue: concurrency cap
- decode: batch width and graph shape
- startup: graph capture/profile sizes

Qwen3.5 tune:
- `max-num-seqs=10` caps simultaneous in-engine sequences; if workload has 20 users,
  queue can dominate. But increasing it grows decode batch and graph memory.

### 10.14 `--max-num-batched-tokens`

`EXACT`

Resolve:
- `_set_default_max_num_seqs_and_batched_tokens_args`
- `SchedulerConfig.max_num_batched_tokens`

Consume:
- scheduler token budget each step
- FlashInfer autotune max token dummy run
- cudagraph max size cap
- multimodal encoder budget
- GDN/attention prefill shapes

Validation:
- If chunked prefill disabled, must be >= `max_model_len`.
- Must be >= `max_num_seqs`.
- If `> max_num_seqs * max_model_len`, warn.

Latency components:
- queue: more prefill tokens per engine step
- prefill: fewer chunks
- decode: mixed batch can become heavier

Qwen3.5 tune:
- The observed small sweeps `2164`, `2174` are in the zone where block alignment
  and `mamba_cache_mode` can matter. If resolved block size becomes ~1072, only
  about two blocks fit in a 2164-token step.

### 10.15 `--gdn-prefill-backend=flashinfer`

`EXACT`

Resolve:
- `EngineArgs.gdn_prefill_backend`
- `additional_config["gdn_prefill_backend"]`

Consume:
- `qwen_gdn_linear_attn.py::_resolve_gdn_prefill_backend`
- `ChunkGatedDeltaRule.__init__`
- `gdn_attn.py`

Fallback logic:
- Non-CUDA -> active backend `triton`.
- Hopper SM90 + requested `auto/flashinfer` -> FlashInfer supported.
- Blackwell path requires `head_k_dim == 128`, CUDA runtime >= 13, and intact
  `nvidia-cutlass-dsl-libs-cu13` install for FlashInfer; else fallback warning.
- `cutedsl` is opt-in and only supported in specific Blackwell path.

Latency components:
- prefill_gdn_layers

Qwen3.5 tune:
- This is more important than `attention-backend` for Qwen3.5 because 18/24
  layers are GDN/linear attention.

### 10.16 `--no-enable-log-requests`

`EXACT`

Resolve:
- API arg boolean pair around `enable_log_requests`

Consume:
- `api_server.py`: if false, `request_logger = None`
- `OpenAIServing...::_log_inputs()` returns without logging

Logic:
- `enable-log-outputs` validation requires `enable-log-requests`; otherwise CLI check fails/warns depending parser path.
- `max-log-len` only matters if logger exists.

Latency components:
- frontend residual CPU and log IO

Qwen3.5 tune:
- Good for benchmark. It removes prompt string/token logging work.

### 10.17 `--async-scheduling`

`EXACT`

Resolve:
- `SchedulerConfig.async_scheduling`

Consume:
- `SchedulerConfig.get_scheduler_cls()`
- if true -> `vllm.v1.core.sched.async_scheduler.AsyncScheduler`

Logic:
- Not just a bool inside same scheduler; it swaps scheduler class.
- Spec decode has additional placeholder logic in `AsyncScheduler`.

Latency components:
- scheduler_overhead
- next_schedule_wait

Qwen3.5 tune:
- Useful when CPU scheduling bubbles show up, but it can interact with spec decode.

### 10.18 commented `--reasoning-parser=qwen3`

`EXACT`

Resolve:
- OpenAI serving reasoning parser args

Consume:
- `OpenAIServingChat._create_chat_completion()`
- constructs parser if `reasoning_parser_cls` set
- stream generator uses reasoning state/delta parse

Logic:
- Parser is response parsing/output processing, not model compute directly.
- If `enable_thinking=false`, parser may be unnecessary depending request/format.

Latency components:
- output_build
- frontend parser setup

Qwen3.5 tune:
- If thinking is disabled at template level, leaving reasoning parser off avoids
  parser work and failure surface.

### 10.19 commented `--speculative-config={"method":"mtp",...}`

`EXACT`

Resolve:
- `EngineArgs.create_speculative_config(...)`
- CLI shorthands `--spec-method`, `--spec-tokens` merge into `speculative_config`
- `SpeculativeConfig.__post_init__()`

Consume:
- scheduler spec placeholders
- `gpu_model_runner.num_spec_tokens`
- rejection sampler
- Mamba/GDN spec state paths
- cudagraph sizes adjusted by `decode_query_len = 1 + num_spec_tokens`

Logic:
- If method `mtp` and no draft model, vLLM uses target model as draft model.
- Draft quantization aligns to target quantization if not explicitly set.
- If config has `n_predict`, `num_speculative_tokens` must obey divisibility constraints.
- Cudagraph capture sizes are rounded/adjusted for spec decode.
- With `mamba_cache_mode=all`, runner has special spec decode postprocess path.

Latency components:
- decode_step_runtime
- graph_or_eager_launch
- mamba_state_write_or_copy

Qwen3.5 tune:
- Spec decode is not automatically a win for GDN hybrid; it increases state and
  verification complexity. Treat as decode lever only after TTFT/prefill is solved.

---

## 11. Extra important flags not always in current submit

### 11.1 `--mamba-cache-mode`

`EXACT`

Resolve:
- `CacheConfig.mamba_cache_mode`, default `"none"`

Consume and rewrite:
- `platforms/interface.py::_align_hybrid_block_size`
- `VllmConfig.validate_block_size`
- `kv_cache_utils.resolve_kv_cache_block_sizes`
- `attention/backends/utils.py::mamba_get_block_table_tensor`
- `gpu_model_runner.py`, `mamba_utils.py`

Mode logic:
- `none`: no Mamba state block cache. Block table shape is compact.
- `all`: cache Mamba state for all block-aligned positions. It does not simply
  "enable more cache"; it recalculates attention block size from Mamba page size,
  Mamba chunk size, and backend kernel alignment.
- `align`: `mamba_block_size = cache_config.block_size`; block table output only
  gathers last `1 + num_speculative_blocks`; has preprocess/postprocess state copy path.

`all` block formula from `platforms/interface.py::_align_hybrid_block_size`:

```text
kv_cache_dtype = cache_dtype == auto ? model dtype : explicit cache dtype
attn_page_size_1_token = FullAttentionSpec/MLA/TurboQuant page bytes for 1 token
mamba_page_size = MambaSpec(state_shapes, state_dtypes, block_size=-1).page_size_bytes
mamba_block_size = user mamba_block_size if explicitly set else None
kernel_block_alignment_size =
  max(min_supported_backend_kernel_block_size, cache_config.block_size)

base_chunk_size = mamba_block_size or model_config.get_mamba_chunk_size()
attn_tokens_per_mamba_state = ceil(mamba_page_size / attn_page_size_1_token)
chunk_size = lcm(base_chunk_size, kernel_block_alignment_size)
attn_block_size = chunk_size * ceil(attn_tokens_per_mamba_state / chunk_size)
cache_config.mamba_block_size = attn_block_size
if cache_config.block_size < attn_block_size:
  cache_config.block_size = attn_block_size
```

Implication:
- `mamba_cache_mode == "all"` can make attention block size much larger than the
  CLI/default block size.
- This changes scheduler block invariants, prefix hash granularity, and how many
  chunks fit inside a `max-num-batched-tokens` budget.
- For Qwen3.5, this is exactly the "block size attention bi align theo
  chunk/kernel alignment cua Mamba" behavior.

Validation:
- `align` requires `block_size <= max_num_batched_tokens`.
- If `long_prefill_token_threshold > 0`, threshold must be >= block_size.
- `align` requires chunked MM flexibility and is not supported by V2 model runner.

Critical interaction:
- If Mamba group block size diverges from attention block size, hash granularity
  falls back to scheduler block size. This can reduce prefix-cache usefulness.

### 11.2 `--mamba-block-size`

`EXACT`

Resolve:
- `CacheConfig.mamba_block_size`
- `user_specified_mamba_block_size`

Validation:
- `VllmConfig.validate_mamba_block_size` raises if set while prefix caching disabled.
- Must be positive; code comment says multiple of 8 for causal_conv1d alignment.

Consume:
- `_align_hybrid_block_size`: used as base chunk size when user-specified.
- Mamba KV group specs and block table logic.

Qwen3.5 note:
- This is a dangerous but real lever. It can change Mamba page/block relation and
  therefore prefix hash granularity.

### 11.3 `--block-size`

`EXACT`

Resolve:
- `CacheConfig.block_size`
- if not user-specified, backend preferred block size may overwrite default.

Consume:
- attention backend supported block size check
- platform hybrid block alignment
- scheduler block invariant
- block hash sizing

Rewrite:
- `current_platform.update_block_size_for_backend()`
- if hybrid, `_align_hybrid_block_size()` may increase it even after backend default.

Qwen3.5 note:
- User setting `--block-size` does not fully protect against hybrid alignment logic;
  `_align_hybrid_block_size` may still need to raise effective block size to make
  attention page >= Mamba page.

### 11.4 `--prefix-caching-hash-algo`

`EXACT`

Resolve:
- `CacheConfig.prefix_caching_hash_algo`

Consume:
- `v1/engine/core.py`: creates caching hash function
- request block hashes via `get_request_block_hasher(hash_block_size, fn)`

Logic:
- Hash is computed per `hash_block_size`, not necessarily physical `block_size`.
- In multi-group hybrid, if Mamba block sizes are incompatible, hash block size
  can fallback coarser.

Qwen3.5 note:
- `xxhash` may reduce CPU hash overhead, but if hash granularity is already coarse,
  the bigger issue is block layout, not hash function speed.

### 11.5 `--disable-hybrid-kv-cache-manager`

`EXACT`

Resolve:
- `SchedulerConfig.disable_hybrid_kv_cache_manager`

Auto decision:
- `VllmConfig.__post_init__()` computes `need_disable_hybrid_kv_cache_manager`.
- Non-GPU platform disables hybrid manager.
- Chunked local attention and some KV connectors can force disable.
- If user explicitly enables while config requires disable, raise.
- If still None after checks, default false.

Consume:
- `kv_cache_utils.get_kv_cache_groups()`
- if true, `unify_hybrid_kv_cache_specs(kv_cache_spec)`

Qwen3.5 note:
- This flag changes cache grouping/allocation strategy. It is not a tiny debug flag.

### 11.6 `--attention-backend`

`EXACT`

Resolve:
- `AttentionConfig.validate_backend_before`
- top-level `attention_backend` and nested `attention_config.backend` are mutually exclusive.

Consume:
- attention backend registry and runner attention groups

Logic:
- `"auto"` maps to `None`.
- String becomes `AttentionBackendEnum[value.upper()]`.
- TurboQuant KV cache can force `flash_attn_version=2` if incompatible with FA3+.

Qwen3.5 note:
- Applies to full-attention layers. It does not select GDN prefill kernels.

### 11.7 `--linear-backend`

`EXACT`

Resolve:
- `KernelConfig.linear_backend`, normalized by lowercase and `-` -> `_`

Consume:
- `model_executor/kernels/linear/__init__.py`

Logic:
- If not `auto`, kernel candidate set is filtered by backend.
- If no kernel exists for the layer/quant type, vLLM raises.

Qwen3.5 note:
- Especially relevant after `--quantization=fp8`; wrong manual backend can be worse
  than auto or simply unsupported.

### 11.8 `--enable-flashinfer-autotune`

`EXACT`

Resolve:
- top-level arg overrides `KernelConfig.enable_flashinfer_autotune`
- optimization-level defaults fill it if still None

Consume:
- `model_executor/warmup/kernel_warmup.py`

Logic:
- If false, logs skip.
- If FlashInfer installed and device capability >= SM90, runs autotune.
- Autotune dummy run uses `max_num_batched_tokens`.

Latency components:
- startup increases
- prefill/decode kernels may choose better tactics

### 11.9 `--enforce-eager`

`EXACT`

Resolve:
- `ModelConfig.enforce_eager`

Consume/rewrite:
- `VllmConfig.__post_init__()`

Logic:
- Sets cudagraph mode to `NONE`.
- Sets `max_cudagraph_capture_size=0`.
- Sets `cudagraph_capture_sizes=[]`.
- ModelConfig docs say eager disables CUDA graph and uses eager PyTorch.

Qwen3.5 note:
- Useful for debugging only. For TPOT it usually removes an important optimization.

### 11.10 `--cudagraph-capture-sizes`

`EXACT`

Resolve:
- `CompilationConfig.cudagraph_capture_sizes`
- mutually exclusive with nested `compilation_config.cudagraph_capture_sizes`

Consume:
- `VllmConfig._set_cudagraph_sizes()`
- `CompilationConfig.post_init_cudagraph_sizes()`
- `gpu_model_runner.py` dispatch/capture logic

Logic:
- If user supplies list, vLLM dedupes, drops sizes > `max_num_batched_tokens`,
  sorts ascending.
- If paired with inconsistent `max_cudagraph_capture_size`, raises.
- Spec decode can round sizes to multiples of `1 + num_spec_tokens`.

Qwen3.5 note:
- Useful only if you know actual decode batch widths. It is easy to capture sizes
  that do not match workload.

### 11.11 `--max-cudagraph-capture-size`

`EXACT`

Resolve:
- `CompilationConfig.max_cudagraph_capture_size`

Default logic:
- If unset and cudagraph enabled:
  `min(max_num_seqs * (1 + num_spec_tokens) * 2, 512)`.
- Then capped by `max_num_batched_tokens`.
- Final value becomes max of actual capture sizes.

Qwen3.5 note:
- This links scheduler concurrency, spec decode, and graph memory. It is not only
  a decode flag.

### 11.12 `--performance-mode`

`EXACT`

Resolve:
- `VllmConfig.performance_mode`

Consume:
- `_set_default_max_num_seqs_and_batched_tokens_args`
- `_set_cudagraph_sizes`
- logs if not balanced

Logic:
- `throughput` doubles default `max_num_batched_tokens` and `max_num_seqs` only
  if the original values were None.
- `interactivity` changes cudagraph default sizes: captures every size from 1
  up to min(max capture, 32), reducing padding at small decode batch sizes.

Qwen3.5 note:
- If compose explicitly sets batch/seqs, throughput mode only affects remaining
  defaults like cudagraph posture, not the manually-set scheduler knobs.

### 11.13 `--safetensors-load-strategy`

`EXACT`

Resolve:
- `LoadConfig.safetensors_load_strategy`

Consume:
- `weight_utils.py::safetensors_weights_iterator`

Logic:
- `"eager"` reads whole safetensors file bytes then loads.
- `"prefetch"` calls `_prefetch_all_checkpoints(...)` before iterating.
- `None` auto-prefetches only if filesystem is recognized network FS
  (`nfs`, `nfs4`, `lustre`) and checkpoint fits within 90% available RAM.
- Local FS with enough RAM still logs auto-prefetch disabled unless user forces it.

Latency components:
- startup only

### 11.14 `--limit-mm-per-prompt`

`EXACT`

Resolve:
- `MultiModalConfig._validate_limit_per_prompt`

Logic:
- Legacy int form becomes `{"count": int}`.
- `image`, `video`, `audio` get modality-specific dummy option classes.
- Unspecified modality defaults to limit 999.
- If `language_model_only=True`, effective limit is always 0.

Latency components:
- frontend validation/preprocess
- startup dummy budget

### 11.15 `--skip-mm-profiling`

`EXACT`

Resolve:
- `MultiModalConfig.skip_mm_profiling`

Consume:
- multimodal registry/profiling paths
- renderer warmup and MM processor behavior

Logic:
- Relevant only if model is considered multimodal and MM processor path exists.
- With `language_model_only=True`, practical importance is lower but still worth
  tracing because some startup paths inspect MM config before request time.

### 11.16 `--renderer-num-workers`

`EXACT`

Resolve:
- `ModelConfig.renderer_num_workers`

Validation:
- If model is multimodal and `renderer_num_workers > 1` while
  `mm_processor_cache_gb > 0`, vLLM raises because cache is not thread-safe.

Consume:
- `BaseRenderer.__init__`: `ThreadPoolExecutor(max_workers=pool_workers)`
- async tokenizer and MM preprocess use that executor
- HF renderer may create async tokenizer with `renderer_num_workers + 1`

Latency components:
- frontend CPU parallelism

### 11.17 `--mamba-cache-dtype` / `--mamba-ssm-cache-dtype`

`EXACT`

Resolve:
- `CacheConfig.mamba_cache_dtype`
- `CacheConfig.mamba_ssm_cache_dtype`

Consume:
- Mamba/GDN state shape dtype helpers
- `model_executor/layers/mamba/mamba_utils.py`
- Qwen3.5 verify update config

Qwen3.5-specific rewrite:
- If `mamba_ssm_cache_dtype == "auto"` and HF config has `mamba_ssm_dtype`,
  Qwen3.5 config update sets cache dtype to that value.
- If user overrides to different value, vLLM logs warning and uses user value.

Latency components:
- mamba state memory bandwidth/capacity
- numerical behavior

### 11.18 `--mamba-backend`

`EXACT`

Resolve:
- `MambaConfig.backend`
- string -> `MambaBackendEnum[value.upper()]`

Validation:
- stochastic rounding with Triton backend requires CUDA Blackwell family.
- stochastic rounding generally requires CUDA.

Consume:
- Mamba selective state update backend choice

Qwen3.5 note:
- Separate from `gdn-prefill-backend`. One selects SSM/state update backend, the
  other selects GDN prefill kernel.

### 11.19 `--kv-cache-memory-bytes`

`EXACT`

Resolve:
- `CacheConfig.kv_cache_memory_bytes`

Consume:
- `gpu_worker.py::determine_available_memory`

Logic:
- If set, vLLM still runs `profile_run()` for compilation/max token shapes.
- Then returns the explicit KV bytes and logs that it ignores
  `gpu_memory_utilization`.

Latency components:
- startup/profile
- queue capacity

Qwen3.5 note:
- Good for reproducibility. Dangerous if container free memory changes between runs.

### 11.20 `--num-gpu-blocks-override`

`EXACT`

Resolve:
- `CacheConfig.num_gpu_blocks_override`

Consume:
- `kv_cache_utils.py`: override computed `num_blocks`
- `gpu_model_runner.profile_run()` temporarily sets minimal override for profiling

Logic:
- Meant for testing/preemption/reproducibility.
- Changes actual cache capacity independent of memory profiler result.

### 11.21 `--scheduler-cls`

`EXACT`

Resolve:
- `SchedulerConfig.scheduler_cls`

Consume:
- `SchedulerConfig.get_scheduler_cls()`

Logic:
- If custom path, resolved by `resolve_obj_by_qualname`.
- vLLM warns interface is not public/stable.
- If unset and `async_scheduling=True`, uses `AsyncScheduler`; else `Scheduler`.

### 11.22 `--scheduling-policy`

`EXACT`

Resolve:
- `SchedulerConfig.policy`

Consume:
- scheduler waiting queue ordering

Logic:
- `fcfs`: arrival order.
- `priority`: priority field with arrival tie-break.

Qwen3.5 note:
- For benchmark traces with fixed arrival, `fcfs` is easier to reason about.

### 11.23 `--max-num-partial-prefills` / `--max-long-partial-prefills`

`EXACT`

Resolve:
- `SchedulerConfig`

Validation:
- If `max_num_partial_prefills > 1`, chunked prefill must be enabled.
- If `long_prefill_token_threshold == 0`, vLLM sets it to `int(max_model_len * 0.04)`.
- `max_long_partial_prefills <= max_num_partial_prefills`.

Consume:
- scheduler limits how many partial prefills and long partial prefills run concurrently.

Qwen3.5 note:
- This is a fairness/queue lever, not direct kernel speed. It can change tail TTFT
  and TPOT by allowing more prefill chunks to coexist with decode.

### 11.24 `--long-prefill-token-threshold`

`EXACT`

Resolve:
- `SchedulerConfig.long_prefill_token_threshold`

Consume:
- scheduler clamps `num_new_tokens` to threshold for long requests.
- align mode validation requires threshold >= block_size if threshold > 0.

Qwen3.5 note:
- If resolved `block_size` becomes large due to Mamba alignment, a too-small
  threshold is invalid or counterproductive.

### 11.25 `--stream-interval`

`EXACT`

Resolve:
- `SchedulerConfig.stream_interval`

Consume:
- `output_processor.py::make_request_output`

Logic:
- If `stream_interval > 1`, output is emitted only on finish, first token, or
  enough new tokens since previous emission.

Latency components:
- client-visible TPOT/output cadence
- not engine decode speed directly

### 11.26 `--enable-prompt-tokens-details`

`EXACT`

Resolve:
- OpenAI API args

Consume:
- chat/completion serving final usage
- sets `PromptTokenUsageInfo(cached_tokens=...)` when cached tokens exist

Logic:
- Useful to observe prefix caching from API response.
- Adds response construction work and response bytes.

### 11.27 `--enable-force-include-usage`

`EXACT`

Resolve:
- OpenAI API args

Consume:
- `entrypoints/utils.py::should_include_usage`

Logic:
- If enabled, always returns `(True, True)` irrespective of request stream options.
- This is observability/benchmark support, not engine perf.

### 11.28 `--enable-server-load-tracking`

`EXACT`

Resolve:
- API args -> app state

Consume:
- `load_aware_call`
- increments/decrements `server_load_metrics`
- disconnect listener decrements on disconnect

Latency components:
- tiny residual CPU per request

### 11.29 `--disable-log-stats`

`EXACT`

Resolve:
- API/engine args

Consume:
- `api_server.py`: `state.log_stats = not args.disable_log_stats`
- engine/stat loggers

Logic:
- Periodic stats logging can be useful for diagnostics, but should not be
  confused with request logging.

### 11.30 `--kv-cache-metrics` / `--kv-cache-metrics-sample`

`EXACT`

Resolve:
- `ObservabilityConfig`

Consume:
- `Scheduler.__init__`: `KVCacheMetricsCollector(sample_rate)`
- metrics loggers expose sampled KV cache events

Logic:
- Provides cache observability at overhead cost.
- Sampling controls event volume, not cache behavior.

---

## 12. Cross-flag interactions that matter most

### 12.1 `max-model-len` x `max-num-batched-tokens`

`EXACT`

- If chunked prefill disabled, `max_num_batched_tokens` must cover `max_model_len`.
- If chunked prefill enabled, `max_num_batched_tokens` can be smaller and acts as
  per-step token budget.
- Lowering `max_model_len` can reduce KV feasibility pressure even if batch tokens
  stay the same.

### 12.2 `kv-cache-dtype=fp8` x `mamba-cache-mode`

`EXACT`

- FP8 KV changes attention page size.
- Hybrid block alignment tries to make attention page >= Mamba page.
- `mamba_cache_mode=all` aligns to `lcm(mamba_chunk_size, kernel_block_alignment)`.
- Result can be a much larger effective attention `block_size`.

### 12.3 `mamba-cache-mode` x `prefix-caching-hash-algo`

`EXACT`

- Hash algo controls hash function.
- `hash_block_size` controls granularity.
- Hybrid Mamba block divergence can force `hash_block_size = scheduler_block_size`.
- In that case switching `sha256` to `xxhash` may reduce CPU hash cost, but does
  not recover lost granularity.

### 12.4 `spec-tokens` x `cudagraph-capture-sizes`

`EXACT`

- `decode_query_len = 1 + num_spec_tokens`.
- Default max capture size uses `max_num_seqs * decode_query_len * 2`.
- Spec decode may round capture sizes to multiples of `decode_query_len`.
- Mamba spec decode has extra all/align state handling.

### 12.5 `renderer-num-workers` x multimodal cache

`EXACT`

- `renderer_num_workers > 1` is rejected when multimodal processor cache is enabled.
- To use parallel renderer workers with multimodal-capable model, MM processor cache
  may need to be disabled.
- With `language-model-only=True`, effective MM prompt limit is 0, lowering relevance
  of MM cache.

### 12.6 `performance-mode` x explicit scheduler flags

`EXACT`

- `throughput` doubles only missing defaults.
- If `max-num-batched-tokens` and `max-num-seqs` are explicit, no doubling.
- `interactivity` still changes cudagraph size generation if cudagraph sizes are not
  manually specified.

### 12.7 `gdn-prefill-backend` x hardware

`EXACT`

- `flashinfer` request is not a guarantee.
- Active backend can be `triton` if platform support checks fail.
- For Qwen3.5, this fallback is significant because GDN layers dominate count.

### 12.8 `kv-cache-memory-bytes` x `gpu-memory-utilization`

`EXACT`

- Explicit KV bytes ignore GPU utilization for KV reservation.
- vLLM still runs profile for compilation/shape warmup.
- This is deterministic capacity control, not speed control.

---

## 13. Coverage index: da trace nhung gi

Full parser cua image co 261 CLI flags. Khong phai flag nao cung la TTFT/TPOT
lever. Phan nay danh dau nhom da trace den implementation logic, de tranh nham
"trace vai flag trong compose" voi "trace latency-relevant flags".

### 13.1 Traced deeply: submit flags

Da trace toi resolve path, runtime consumer, rewrite/validation, latency component:

```text
model
served-model-name
host
port
max-model-len
gpu-memory-utilization
tensor-parallel-size
enable-prefix-caching
language-model-only
default-chat-template-kwargs
quantization
kv-cache-dtype
calculate-kv-scales
max-num-seqs
max-num-batched-tokens
gdn-prefill-backend
enable-log-requests/no-enable-log-requests
async-scheduling
reasoning-parser
speculative-config
```

### 13.2 Traced deeply: high-impact non-submit flags

Da trace toi config object va consumer that su:

```text
mamba-cache-mode
mamba-block-size
block-size
prefix-caching-hash-algo
disable-hybrid-kv-cache-manager
attention-backend
linear-backend
enable-flashinfer-autotune
enforce-eager
cudagraph-capture-sizes
max-cudagraph-capture-size
performance-mode
safetensors-load-strategy
limit-mm-per-prompt
skip-mm-profiling
renderer-num-workers
mamba-cache-dtype
mamba-ssm-cache-dtype
mamba-backend
kv-cache-memory-bytes
num-gpu-blocks-override
scheduler-cls
scheduling-policy
max-num-partial-prefills
max-long-partial-prefills
long-prefill-token-threshold
stream-interval
enable-prompt-tokens-details
enable-force-include-usage
enable-server-load-tracking
disable-log-stats
kv-cache-metrics
kv-cache-metrics-sample
```

### 13.3 Traced shallow but latency-relevant enough

Nhung flag nay co anh huong latency, nhung voi current Qwen3.5 text-only
single-GPU submit, chung thuong khong phai shortlist dau tien:

```text
chat-template
chat-template-content-format
trust-request-chat-template
response-role
tokenizer
tokenizer-mode
skip-tokenizer-init
disable-uvicorn-access-log
uvicorn-log-level
max-log-len
enable-log-outputs
enable-log-deltas
collect-detailed-traces
kv-sharing-fast-prefill
disable-cascade-attn
load-format
safetensors-prefetch-num-threads
safetensors-prefetch-block-size
spec-method
spec-model
spec-tokens
```

### 13.4 Parser flags deliberately not treated as tuning levers here

Nhung flag nhu TLS/CORS/API key/docs endpoint, DP supervisor port, request ID
headers, fingerprint, shutdown timeout, middleware, SSL refresh, FastAPI docs
toggle co the them residual CPU rat nho hoac thay doi deployment behavior. Chung
khong doi scheduler, KV cache, model kernels, graph capture, GDN/Mamba state, hay
tokenization path trong benchmark text-only hien tai.

---

## 14. Extra parser flags: implementation logic and TTFT/TPOT relevance

Phan nay gom cac flag quan trong trong `docs/vllm-v0.22.1-flags-reference.md`
nhung chua xuat hien trong compose hien tai.

### 14.1 `--dtype`

`EXACT`

Resolve:
- `ModelConfig.dtype`
- model HF config dtype resolution
- platform verification

Consume:
- weight load dtype
- activation dtype
- attention page size when `kv-cache-dtype=auto`
- Mamba alignment because `attn_page_size_1_token` uses model dtype if KV dtype is auto

Latency components:
- startup load/cast
- prefill/decode compute bandwidth
- cache block alignment if KV cache follows model dtype

Qwen3.5 note:
- If explicit `kv-cache-dtype=fp8`, `dtype` no longer decides attention KV dtype,
  but it still affects weights/activations and some fallback dtype paths.

### 14.2 `--runner` / `--convert` / `--model-impl`

`EXACT`

Resolve:
- `ModelConfig.runner`
- `ModelConfig.convert`
- `ModelConfig.model_impl`
- `registry.inspect_model_cls(...)`

Consume:
- model class selection
- generate vs pooling vs draft behavior
- supported runner checks before engine starts

Latency components:
- startup and whole engine path selection

Qwen3.5 note:
- For this race, `runner=generate`/`auto` is the relevant path. Wrong runner can
  disable the intended generation engine rather than tune latency.

### 14.3 `--trust-remote-code`, `--hf-config-path`, `--revision`, `--code-revision`, `--tokenizer-revision`

`EXACT`

Resolve:
- `get_config(...)`
- tokenizer/config/model code loading
- HF/local checkpoint resolution

Consume:
- model architecture inspect
- tokenizer class loading
- possible remote/custom model code path

Latency components:
- startup only for local fixed image
- frontend tokenization can change if tokenizer revision/path changes

Qwen3.5 note:
- With `/model` baked in image, these are reproducibility/source-selection flags,
  not runtime TTFT/TPOT levers.

### 14.4 `--hf-overrides`, `--generation-config`, `--override-generation-config`

`EXACT`

Resolve:
- model config after HF config load
- generation config fetch/override

Consume:
- default sampling/generation parameter construction
- model metadata such as layer config if `hf-overrides` changes architecture fields

Latency components:
- frontend sampling param build
- prefill/decode indirectly if max tokens, stop behavior, or architecture metadata changes

Qwen3.5 note:
- `hf-overrides` is powerful and dangerous: it can change what vLLM believes the
  model architecture is. It should not be used as a casual perf knob.

### 14.5 `--disable-sliding-window`

`EXACT`

Resolve:
- `ModelConfig.disable_sliding_window`

Consume:
- attention layer/window metadata
- KV cache and attention mask behavior for sliding-window-capable models

Latency components:
- prefill/decode attention work and KV retention for models with sliding window

Qwen3.5 note:
- Only meaningful if the model config has sliding-window attention. It is not the
  main hybrid GDN/Mamba lever.

### 14.6 `--override-attention-dtype`

`EXACT`

Resolve:
- `ModelConfig.override_attention_dtype`

Consume:
- attention backend dtype path

Latency components:
- attention kernel dtype, numerical behavior

Qwen3.5 note:
- Affects full-attention layers, not GDN prefill backend selection.

### 14.7 `--max-logprobs`, `--logprobs-mode`, `--use-fp64-gumbel`

`EXACT`

Resolve:
- `ModelConfig.max_logprobs`
- `ModelConfig.logprobs_mode`
- sampler config path

Consume:
- output logprob collection
- logits/logprobs processing
- sampling randomness path

Latency components:
- decode sampler/output build

Qwen3.5 note:
- If benchmark does not request logprobs, these should not matter much. If it
  requests logprobs, output build and sampler overhead can become visible.

### 14.8 `--enable-prompt-embeds`

`EXACT`

Resolve:
- `ModelConfig.enable_prompt_embeds`

Consume:
- serving/input processor allows prompt embedding requests
- V2 model runner unsupported-feature checks include prompt embeds

Latency components:
- frontend validation/input handling

Qwen3.5 note:
- Not relevant for normal text prompts; enabling increases supported surface area
  rather than speeding tokenized prompts.

### 14.9 `--decode-context-parallel-size`, `--prefill-context-parallel-size`, `--cp-kv-cache-interleave-size`

`EXACT`

Resolve:
- `ParallelConfig.decode_context_parallel_size`
- `ParallelConfig.prefill_context_parallel_size`
- `ParallelConfig.cp_kv_cache_interleave_size`

Consume:
- parallel group layout
- block-size validation:
  `block_size >= cp_kv_cache_interleave_size` and divisible by it

Latency components:
- distributed prefill/decode communication
- block-size constraints

Qwen3.5 note:
- With `tensor-parallel-size=1` and single GPU, these are not practical levers.
  They become important only when splitting context across devices.

### 14.10 `--data-parallel-size` and DP external-LB flags

`EXACT`

Resolve:
- `ParallelConfig.data_parallel_size`
- API server supervisor/external load balancer args

Consume:
- multi-engine process layout
- external load-balancer rank/rpc setup
- aggregate engine logging

Latency components:
- queue at deployment level, not per-engine kernel speed

Qwen3.5 note:
- Helps throughput/queue only if you actually run multiple replicas. It does not
  reduce single-request prefill/decode inside one engine.

### 14.11 `--enable-dbo`, `--ubatch-size`, `--dbo-decode-token-threshold`, `--dbo-prefill-token-threshold`

`EXACT`

Resolve:
- parallel/scheduler config path for dual batch overlap

Consume:
- microbatch split/overlap logic when DBO is supported
- unsupported-feature checks in some runner paths

Latency components:
- decode/prefill overlap
- scheduler/launch overhead

Qwen3.5 note:
- Potentially relevant only after stable baseline. It changes execution overlap
  behavior and can make traces harder to interpret.

### 14.12 `--cpu-offload-gb`, `--offload-backend`, `--offload-*`, `--kv-offloading-*`

`EXACT`

Resolve:
- cache/offload config
- KV transfer/offloading config

Consume:
- weight or KV movement between CPU/GPU or offload backend
- offload prefetch/group scheduling

Latency components:
- startup and per-step memory movement
- queue capacity if KV offload changes effective cache

Qwen3.5 note:
- Usually anti-latency unless GPU memory is the bottleneck. It can improve
  feasibility/capacity, not raw TPOT.

### 14.13 `--mm-processor-cache-gb`, `--mm-processor-cache-type`, `--enable-mm-embeds`, `--media-io-kwargs`, `--mm-processor-kwargs`

`EXACT`

Resolve:
- `MultiModalConfig`
- renderer/multimodal processing context

Consume:
- MM processor cache
- media loader and MM input preprocessing
- renderer worker validation

Latency components:
- frontend and startup for multimodal models

Qwen3.5 note:
- With `language-model-only=True`, effective modality limit becomes 0, so these
  are mostly guarded away for text-only benchmark.

### 14.14 `--profiler-config`, `--cudagraph-metrics`, `--enable-layerwise-nvtx-tracing`, `--enable-mfu-metrics`, `--enable-logging-iteration-details`

`EXACT`

Resolve:
- `ObservabilityConfig`
- profiler config

Consume:
- engine stats/tracing/profiler hooks
- NVTX and detailed iteration logging

Latency components:
- residual CPU/GPU instrumentation overhead

Qwen3.5 note:
- Good for diagnosis, bad for clean benchmark. Use only during trace runs.

### 14.15 `--optimization-level`

`EXACT`

Resolve:
- `VllmConfig.optimization_level`

Consume:
- `_apply_optimization_level_defaults()`
- compilation/kernel defaults including cudagraph and FlashInfer autotune posture

Latency components:
- startup compile/warmup
- decode graph path
- kernel warmup/default selection

Qwen3.5 note:
- This is a preset layer. If you manually set cudagraph/kernel flags, some of its
  defaulting power disappears, similar to `performance-mode`.
