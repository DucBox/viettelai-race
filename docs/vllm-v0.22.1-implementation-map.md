# vLLM v0.22.1 Implementation Map for Qwen3.5-2B

Muc tieu cua file nay la di tu **flag -> runtime config -> code path -> latency component**
thay vi chi mo ta y nghia be mat cua CLI.

Nguon trace:
- Image local: `duc0811/qwen35-2b-race:v1`
- Base image tag: `vllm/vllm-openai:v0.22.1`
- Build commit trong image: `0decac0d96c42b49572498019f0a0e3600f50398`

Phuong phap:
- Doc toan bo CLI parser thuc te trong image.
- Trace tiep tu `EngineArgs.create_engine_config()` den `ModelConfig`, `CacheConfig`,
  `SchedulerConfig`, `VllmConfig`, `gpu_model_runner`, `output_processor`,
  `openai api_server`, `renderer`, va `weight_loader`.
- Khong ket luan tu docs online neu chua thay trong code cua image.

Companion doc moi:
- `docs/vllm-v0.22.1-latency-flag-trace.md`
  - tap trung vao bo tach TTFT/TPOT sau hon
  - co ma tran `flag -> resolve -> consume -> latency component`
  - bao phu toan bo nhom co y nghia thay vi chi mot vai leverage lon

Luu y:
- May dev hien tai khong co CUDA runtime thuc, nen phan nay la **trace implementation**
  trong image, khong phai full benchmark runtime tren GPU.
- Nhung gi duoc danh dau `EXACT` la da thay trong code.
- Nhung gi duoc danh dau `OBSERVED IN REPO` la ket qua chay/submit co san trong repo.

---

## 1. Cay phan ra TTFT / TPOT sau khi trace code

### 1.1 TTFT

```
client_ttft
= frontend_prep
+ server_queue
+ server_prefill
+ client_transport
```

Trong do:

- `frontend_prep`
  - `model_check`
  - `chat_template_resolution`
  - `chat_render`
  - `tokenization`
  - `multimodal_preprocess`
  - `sampling_param_build`
  - `request_id / metadata / trace_header prep`
  - `engine enqueue handoff`

- `server_queue`
  - `wait_prefill`
  - `wait_decode`
  - `scheduler_overhead`
  - `kv admission / reserve-full-isl gating`

- `server_prefill`
  - `prefix hash / cache lookup`
  - `chunk sizing`
  - `full-attention prefill compute`
  - `GDN linear-attention prefill compute`
  - `KV write / mamba state write`
  - `interleave with other requests`

- `client_transport`
  - `first SSE chunk flush`
  - `HTTP / socket buffering`
  - `client-side chunk parse`

### 1.2 TPOT

```
client_tpot
~ pure_decode_step
+ mixed_prefill_penalty
+ streaming_buffer_penalty
+ client_transport_jitter
```

Trong do:
- `pure_decode_step`: 1 iteration decode-only trong engine.
- `mixed_prefill_penalty`: decode iteration bi chen boi prefill chunk.
- `streaming_buffer_penalty`: do `stream_interval > 1`.
- `client_transport_jitter`: network / SSE flush / parser.

---

## 2. Co che runtime quan trong nhat da xac nhan

### 2.1 Prefix caching mac dinh KHONG tu bat cho hybrid model

`EXACT`

Code path:
- `EngineArgs._set_default_chunked_prefill_and_prefix_caching_args`
- `ModelConfig.is_prefix_caching_supported`

Logic:
- Neu `enable_prefix_caching is None`, runtime lay default tu
  `model_config.is_prefix_caching_supported`.
- Voi generative model co `attn_type == "hybrid"`, `is_prefix_caching_supported`
  tra ve `False`.

He qua cho Qwen3.5:
- Qwen3.5 hybrid khong tu bat prefix caching.
- Muon dung thi phai ep `--enable-prefix-caching`.
- Path nay la path experimental tren hybrid, khong phai path mac dinh an toan.

### 2.2 Chunked prefill mac dinh duoc xem la supported cho generative model

`EXACT`

Code path:
- `EngineArgs._set_default_chunked_prefill_and_prefix_caching_args`
- `ModelConfig.is_chunked_prefill_supported`

Logic:
- Voi generative model khong phai encoder-decoder, `is_chunked_prefill_supported`
  tra ve `True`.
- Neu `enable_chunked_prefill is None`, runtime tu bat.

He qua:
- "Khong set flag" khong dong nghia "chunked prefill tat".

### 2.3 `performance_mode=throughput` khong magic nhu tuong tuong

`EXACT`

Code path:
- `EngineArgs._set_default_max_num_seqs_and_batched_tokens_args`

Logic:
- Chi khi user **khong tu set tay** `max_num_batched_tokens` va `max_num_seqs`,
  throughput mode moi nhan doi default.
- Neu user da set tay, no khong overwrite.

He qua:
- Tren config submit da set tay batch/seqs, `performance_mode` khong con la mot
  "preset thay tat ca".

### 2.4 `mamba_cache_mode` anh huong khong chi cache ma con doi block layout

`EXACT`

Code path:
- `vllm/platforms/interface.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/attention/backends/utils.py`

Logic:
- `all`: canh block attention theo mamba chunk/kernel alignment, co the lam
  `attention block_size` tang len.
- `align`: ep `mamba_block_size = attention block_size`, giu alignment cho hash.
- `none`: khong giu state cachable theo block.

He qua:
- `mamba_cache_mode` co the doi:
  - granularity cua prefix hash
  - scheduler block size
  - layout cua Mamba state
  - overhead preprocess/postprocess state copy

### 2.5 `hash_block_size` bi fallback neu hybrid block sizes lech nhau

`EXACT`

Code path:
- `vllm/v1/core/kv_cache_utils.py::get_kv_cache_config_block_sizes`

Logic:
- Neu nhieu KV groups va Mamba group co `block_size != cache_config.block_size`
  (thuong xay ra khi `mamba_cache_mode != "align"`), `hash_block_size` bi fallback
  len `scheduler_block_size`.

He qua:
- Prefix hashing mat finer granularity.
- Day la mot trong cac ly do hybrid prefix caching co the "an khong min" du
  da bat cờ.

### 2.6 `kv_sharing_fast_prefill` hien tai chua co toi uu prefill thuc

`EXACT`

Code path:
- `CacheConfig.kv_sharing_fast_prefill` comment trong code

Logic:
- Code ghi ro: `no prefill optimization takes place with this flag enabled currently`.

He qua:
- Tren Qwen3.5, flag nay khong nen duoc xem la lever toi uu chinh.

### 2.7 `renderer_num_workers` thuc su tac dong vao frontend CPU

`EXACT`

Code path:
- `vllm/renderers/base.py::BaseRenderer.__init__`

Logic:
- API server tao `ThreadPoolExecutor(max_workers=renderer_num_workers)`.
- Pool nay dung cho:
  - tokenization
  - chat template rendering
  - multimodal preprocessing

He qua:
- Day la lever hop le de tach sau `frontend_prep`.
- Repo cu chua do nhanh nay rieng.

### 2.8 `disable_cascade_attn` mac dinh la `True`

`EXACT`

Code path:
- `ModelConfig.disable_cascade_attn`
- `gpu_model_runner.cascade_attn_enabled = not self.model_config.disable_cascade_attn`

Logic:
- Muon cho phep heuristic dung cascade attention thi phai pass
  `--no-disable-cascade-attn`.

He qua:
- "Khong set gi" = cascade attention dang tat.

---

## 3. Frontend request path thuc te

Cho OpenAI chat:

Code path chinh:
- `vllm.entrypoints.openai.chat_completion.serving.OpenAIServingChat`

Flow:

1. `create_chat_completion()`
2. `_create_chat_completion()`
3. `render_chat_request()`
4. `openai_serving_render.render_chat(request)`
5. Renderer lam:
   - resolve chat template
   - parse messages
   - render chat
   - tokenize
   - multimodal preprocess neu can
6. Quay lai `_create_chat_completion()`
7. Build `SamplingParams`
8. Goi engine/generator

Ket luan:
- `chat template + tokenize + multimodal preprocess` xay ra **truoc queue engine**.
- Tuc la nhung cờ frontend/model sau co the an vao TTFT that:
  - `tokenizer`
  - `tokenizer-mode`
  - `renderer-num-workers`
  - `chat-template*`
  - `language-model-only`
  - `skip-mm-profiling`

---

## 4. Flag map: ~50 flag co y nghia

Bang duoi day tap trung vao nhung flag co kha nang:
- doi runtime behavior that su
- doi path latency
- hoac rat quan trong cho benchmarking / giai thich metric

Cot:
- `Runtime object`: config object thuc su nhan gia tri.
- `Code path`: class/hàm tieu thu chinh.
- `Tac dong`: frontend / queue / prefill / decode / startup / observability.
- `Qwen3.5 note`: ghi chu rieng cho hybrid Qwen3.5.

### 4.1 Frontend / API / residual CPU

| Flag | Runtime object | Code path | Tac dong | Qwen3.5 note |
|---|---|---|---|---|
| `--chat-template` | frontend args | `OpenAIServingChat`, renderer | frontend_prep | Anh huong render path |
| `--chat-template-content-format` | frontend args | renderer / chat utils | frontend_prep | Sai format co the tang parse work |
| `--trust-request-chat-template` | frontend args | OpenAI serving | frontend_prep / safety | Khong phai lever perf chinh |
| `--default-chat-template-kwargs` | frontend args | OpenAI serving | frontend_prep | Co the doi logic render |
| `--tokenizer` | `ModelConfig` | tokenizer registry | frontend_prep | Cho phep doi tokenizer source |
| `--tokenizer-mode` | `ModelConfig` | tokenizer registry | frontend_prep | `hf`/`slow` tac dong CPU tokenization |
| `--skip-tokenizer-init` | `ModelConfig` | renderer/tokenizer access | frontend_prep | Chi hop khi request dua token ids san |
| `--renderer-num-workers` | `ModelConfig` | `BaseRenderer` thread pool | frontend_prep | Lever CPU hop le, chua duoc do trong repo cu |
| `--enable-log-requests` | API server args | `api_server.py` tao `RequestLogger` | frontend CPU overhead | Nen tat khi chay bench |
| `--max-log-len` | API server args | request logger | frontend CPU / log IO | Giam chi phi logging neu bat log |
| `--disable-uvicorn-access-log` | API server args | uvicorn launch | frontend CPU / log IO | Nen tat khi bench |
| `--uvicorn-log-level` | API server args | uvicorn | frontend CPU / log IO | Muc `info`/`debug` co the nhieu IO hon |
| `--enable-prompt-tokens-details` | frontend serving | OpenAI serving response | observability | Can bat de thay `cached_tokens` |
| `--enable-force-include-usage` | frontend serving | `entrypoints/utils.py::should_include_usage` | observability / response size | Ep usage luon duoc tra |
| `--stream-interval` | `SchedulerConfig` | `output_processor.py` | TPOT / client_transport | >1 se buffer token, co the lam TTFT/TPOT client xau |
| `--response-role` | frontend args | OpenAI response builder | none / tiny | Khong phai lever perf |
| `--enable-log-outputs` | frontend serving | OpenAI serving | CPU/log IO | Khong nen bat luc bench |
| `--enable-log-deltas` | frontend serving | OpenAI serving | CPU/log IO | Nho hon output full nhung van la log work |

### 4.2 Startup / model load / CPU warm path

| Flag | Runtime object | Code path | Tac dong | Qwen3.5 note |
|---|---|---|---|---|
| `--model` | `ModelConfig` | model load path | startup / all | Model local path trong submit |
| `--hf-config-path` | `ModelConfig` | HF config load | startup | It khi can |
| `--max-model-len` | `ModelConfig` | config -> scheduler validation | KV sizing / queue / startup | Rat quan trong cho trace nay |
| `--load-format` | `LoadConfig` | default loader | startup | `auto`/`safetensors` la duong chinh |
| `--safetensors-load-strategy` | `LoadConfig` | `weight_utils.py` | startup | `prefetch` co co che that su |
| `--safetensors-prefetch-num-threads` | `LoadConfig` | `_prefetch_all_checkpoints` | startup / CPU IO | Chi co y nghia neu prefetch bat |
| `--safetensors-prefetch-block-size` | `LoadConfig` | `_prefetch_checkpoint` | startup / IO pattern | Tinh chinh page-cache warming |
| `--use-tqdm-on-load` | `LoadConfig` | load path | startup / logging | Chu yeu UI/log |
| `--language-model-only` | `ModelConfig` / multimodal | multimodal config infer | startup + frontend_prep + VRAM | Rat hop ly cho text-only Qwen3.5 |
| `--skip-mm-profiling` | multimodal | multimodal startup | startup | Giam startup neu khong dung vision |
| `--limit-mm-per-prompt` | multimodal | multimodal config | frontend validation | Co the chan image path |
| `--mm-processor-cache-gb` | multimodal | mm processor | frontend/multimodal | Khong quan trong cho text-only |

### 4.3 Scheduler / queue / batching

| Flag | Runtime object | Code path | Tac dong | Qwen3.5 note |
|---|---|---|---|---|
| `--max-num-batched-tokens` | `SchedulerConfig` | `EngineArgs._set_default_max_num_seqs_and_batched_tokens_args` | queue + prefill chunking | Lever batch/TTFT chinh |
| `--max-num-seqs` | `SchedulerConfig` | same | queue + TPOT | Lever TPOT chinh |
| `--enable-chunked-prefill` | `SchedulerConfig` | default resolution + scheduler | queue / prefill interleave | Mac dinh path generate la supported |
| `--max-num-partial-prefills` | `SchedulerConfig` | scheduler validation | queue / fairness | OBSERVED IN REPO: de crash/rui ro tren v0.22.1 hybrid |
| `--max-long-partial-prefills` | `SchedulerConfig` | scheduler validation | queue / fairness | Chi co y nghia khi partial prefill > 1 |
| `--long-prefill-token-threshold` | `SchedulerConfig` | scheduler validation | queue / fairness | Repo da gap crash/rui ro voi hybrid |
| `--scheduling-policy` | `SchedulerConfig` | scheduler | queue ordering | `fcfs` khop trace nhat |
| `--scheduler-reserve-full-isl` | `SchedulerConfig` | scheduler admission | queue / KV admission | Co the doi latency tail do over-admission |
| `--async-scheduling` | `SchedulerConfig` | `SchedulerConfig.get_scheduler_cls` | queue / sched overhead / TPOT nho | Doi class `AsyncScheduler` |
| `--performance-mode` | `VllmConfig` + EngineArgs | auto-default batch/seqs | queue + decode (indirect) | Throughput chi nhan doi default neu chua set tay |
| `--stream-interval` | `SchedulerConfig` | output processor | TPOT client / stream jitter | 1 la tot nhat cho metric client |
| `--scheduler-cls` | `SchedulerConfig` | custom scheduler class | queue | Nang cao |

### 4.4 Cache / prefix / hybrid Mamba-GDN

| Flag | Runtime object | Code path | Tac dong | Qwen3.5 note |
|---|---|---|---|---|
| `--gpu-memory-utilization` | `CacheConfig` | cache sizing | KV pool / startup | Co y nghia, nhung khong phai lever diem lon nhat |
| `--kv-cache-memory-bytes` | `CacheConfig` | cache sizing | KV pool | Override truc tiep thay vi util |
| `--block-size` | `CacheConfig` | cache config + platform align | queue / hash / cache granularity | Hybrid co the bi nang len boi Mamba |
| `--num-gpu-blocks-override` | `CacheConfig` | cache manager | KV pool | Chu yeu diagnostic |
| `--enable-prefix-caching` | `CacheConfig` | default resolution + engine core | prefill / cache hit | Hybrid khong tu bat |
| `--prefix-caching-hash-algo` | `CacheConfig` | engine core / block hasher | prefill CPU overhead / cache correctness | `xxhash` nhanh hon, `sha256` mac dinh |
| `--kv-cache-dtype` | `CacheConfig` | cache config / runner | KV memory + attention path | Tren hybrid co the doi block size va TTFT |
| `--calculate-kv-scales` | `CacheConfig` | FP8 KV scale handling | accuracy / tiny startup | Deprecated; khong phai lever perf lon |
| `--kv-cache-dtype-skip-layers` | `CacheConfig` | cache config | KV quant scope | Nang cao |
| `--kv-sharing-fast-prefill` | `CacheConfig` | runner/attention metadata | almost none | WIP, chua co prefill opt that su |
| `--disable-hybrid-kv-cache-manager` | `SchedulerConfig` | scheduler/cache manager | queue / allocation | Worth A/B neu nghi hybrid manager co bug |
| `--mamba-block-size` | `CacheConfig` | platform alignment | hash granularity / state layout | Rat nhay tren hybrid |
| `--mamba-cache-mode` | `CacheConfig` | platform/runner/kv utils | prefill cache / block layout / hash | Lever hybrid quan trong nhat |
| `--mamba-cache-dtype` | `CacheConfig` | mamba cache dtype | memory / accuracy | It memory win hon KV quant |
| `--mamba-ssm-cache-dtype` | `CacheConfig` | mamba ssm state dtype | memory / accuracy | Nang cao |
| `--kv-offloading-size` | `CacheConfig` | KV offload | memory vs latency | Khong hop cho box RAM yeu |
| `--kv-offloading-backend` | `CacheConfig` | KV offload | memory vs latency | Khong phai huong toi uu cho bai nay |

### 4.5 Compute / kernel / prefill / decode

| Flag | Runtime object | Code path | Tac dong | Qwen3.5 note |
|---|---|---|---|---|
| `--quantization` | `ModelConfig` / quant config | quant resolution + model load | prefill + decode GEMM | FP8 weight la lever compute lon |
| `--quantization-config` | quant config | quant resolution | same | Dung khi can override chi tiet |
| `--dtype` | `ModelConfig` | model load | compute precision | BF16 goc cua model |
| `--attention-backend` | `AttentionConfig` | attention backend selection | prefill/decode attention | Chi tac dong 6 attention layers |
| `--gdn-prefill-backend` | arg -> runner/backend | `gdn_attn.py`, qwen GDN layer | prefill GDN | Rat quan trong vi 18/24 layer la GDN |
| `--linear-backend` | `KernelConfig` | kernel selection | GEMM perf | Huu ich sau quantization |
| `--enable-flashinfer-autotune` | `KernelConfig` | backend tuning | prefill/decode kernels | Trade startup lay kernel tot hon |
| `--disable-cascade-attn` | `ModelConfig` -> runner flag | `gpu_model_runner` | prefill/decode attention heuristic | Mac dinh dang tat |
| `--enforce-eager` | `ModelConfig` | compilation/runner | decode overhead / startup | Bat = tat CUDA graph |
| `--cudagraph-capture-sizes` | `CompilationConfig` | compile/cudagraph | decode / startup | Quan trong khi muon batch sizes duoc capture |
| `--max-cudagraph-capture-size` | `CompilationConfig` | compile/cudagraph | decode / startup | Cap cho graph capture |
| `--optimization-level` | `VllmConfig` | compile config | startup vs perf | O2 mac dinh |
| `--performance-mode` | `VllmConfig` | default runtime posture | queue/decode indirect | Khong overwrite value user set tay |
| `--spec-method` | speculative config | draft path | decode / TPOT | OBSERVED IN REPO: MTP co regression tren GDN |
| `--spec-tokens` | speculative config | draft path | decode / TPOT | Chi co y nghia khi spec decode bat |
| `--speculative-config` | speculative config | draft path | decode / TPOT | Tinh chinh tong |

### 4.6 Observability / benchmark support

| Flag | Runtime object | Code path | Tac dong | Qwen3.5 note |
|---|---|---|---|---|
| `--disable-log-stats` | engine/api server | `api_server.py`, `async_llm.py`, `llm_engine.py` | CPU/log overhead | Nen tat khi chay bench |
| `--kv-cache-metrics` | observability | metrics path | observability | Huu ich khi debug cache |
| `--kv-cache-metrics-sample` | observability | metrics path | observability | Trade overhead/chi tiet |
| `--enable-logging-iteration-details` | observability | iteration logging | CPU/log overhead | Diagnostic only |
| `--enable-mfu-metrics` | observability | metrics path | observability | Chuan doan compute utilization |
| `--collect-detailed-traces` | observability | tracing | observability / overhead | Chi bat khi debug sau |
| `--otlp-traces-endpoint` | observability | tracing export | observability | Khong phai lever perf |

---

## 5. Logic runtime chi tiet cua cac flag quan trong nhat

### 5.1 `max-num-batched-tokens`

Code:
- `EngineArgs.get_batch_defaults`
- `EngineArgs._set_default_max_num_seqs_and_batched_tokens_args`
- `SchedulerConfig`

Logic:
- OpenAI API server tren GPU thuong:
  - default batch tokens = `2048`
  - default seqs = `256`
- GPU RAM >= 70 GiB va khong phai A100:
  - default batch tokens = `8192`
  - default seqs = `1024`
- `performance_mode=throughput`:
  - neu khong user override, nhan doi ca 2.

Tac dong latency:
- Tăng batch token budget:
  - giam so iteration prefill can dung
  - giam queue tail
  - co the tang mixed-prefill penalty len decode

### 5.2 `max-num-seqs`

Code:
- same path nhu tren
- `gpu_model_runner.max_num_reqs = scheduler_config.max_num_seqs`

Tac dong latency:
- Giam `max_num_seqs`:
  - giam ap luc decode batch
  - TPOT thuong tot len
  - nhung queue co the tang rat manh

### 5.3 `stream-interval`

Code:
- `output_processor.py::make_request_output`

Logic:
- Neu `stream_interval > 1`, output chi duoc gui khi:
  - request finished, hoac
  - day la token dau, hoac
  - so token moi tu lan gui truoc >= stream_interval

He qua:
- Day la mot co che buffer stream that su.
- No co the lam:
  - TTFT client van on cho token dau
  - nhung TPOT client bi "gap" lon hon
  - giam host overhead mot chut

### 5.4 `safetensors-load-strategy`

Code:
- `weight_utils.py`

Logic:
- `prefetch`: chu dong page-cache warm file weights.
- `None`:
  - chi auto-prefetch khi la network FS (`nfs`, `nfs4`, `lustre`)
  - va checkpoint fit trong RAM.
- Neu local FS thi auto-prefetch mac dinh tat.

He qua:
- Tren moi truong giong repo cua ban, log "Auto-prefetch disabled" co co che ro rang.
- Muon ep prefetch phai set tay.

### 5.5 `enable-force-include-usage`

Code:
- `entrypoints/utils.py::should_include_usage`

Logic:
- Neu bat cờ nay, function tra `(True, True)` bat ke `stream_options`.

He qua:
- Huu ich cho bench/telemetry.
- Khong phai lever latency engine chinh, nhung co the tang kich thuoc response va
  mot it frontend work.

### 5.6 `disable-log-stats` va `enable-log-requests`

Code:
- `api_server.py`
- `async_llm.py`
- `llm_engine.py`

Logic:
- `enable_log_requests` tao `RequestLogger`.
- `disable_log_stats` tat periodic engine stats logging.

He qua:
- Tren box CPU it core, day la overhead that su cua residual CPU/log IO.

### 5.7 `mamba_cache_mode`

Code:
- `platforms/interface.py`
- `kv_cache_utils.py`
- `gpu_model_runner.py`
- `attention/backends/utils.py`

Logic:
- `none`
  - mamba block table shape ngan nhat
  - khong prefix cache state theo block
- `all`
  - cache state tai moi vi tri `i * block_size`
  - co the doi attention block size theo mamba chunk alignment
  - spec decode co postprocess rieng
- `align`
  - lay block table "last block only" theo seq len
  - co preprocess/postprocess state-copy path rieng
  - co xu huong giu hash/block alignment dep hon

He qua:
- Khong chi la `cache on/off`.
- No can thiep truc tiep vao:
  - block layout
  - block hash granularity
  - preprocessing state copy
  - speculative decode handling

---

## 6. Ket luan rieng cho Qwen3.5 hybrid

### 6.1 Nhom flag chac chan dang y nghia

- `max-num-batched-tokens`
- `max-num-seqs`
- `enable-chunked-prefill`
- `enable-prefix-caching`
- `mamba-cache-mode`
- `mamba-block-size`
- `gdn-prefill-backend`
- `quantization`
- `kv-cache-dtype`
- `gpu-memory-utilization`
- `max-model-len`
- `language-model-only`
- `renderer-num-workers`
- `stream-interval`
- `disable-log-stats`
- `enable-log-requests`
- `disable-uvicorn-access-log`
- `safetensors-load-strategy`
- `attention-backend`
- `disable-cascade-attn`
- `async-scheduling`
- `cudagraph-capture-sizes`
- `max-cudagraph-capture-size`
- `performance-mode`

### 6.2 Nhom flag co y nghia nhung can rat than trong

- `mamba-cache-mode=all`
  - co the tang block size va doi hash granularity
  - code khong cam rieng Qwen3.5 o tang parser/config
  - nhung hybrid prefix-caching mac dinh la unsupported/experimental path
  - khong duoc xem la "free win"
- `kv-cache-dtype=fp8`
  - co tac dong that vao block/page layout
  - tren head_dim=256 can A/B TTFT that su
- `max-num-partial-prefills`
  - repo da quan sat crash/rui ro
- `speculative-config`
  - repo da quan sat regression tren GDN

### 6.3 Nhom flag nhin co ve ngon nhung hien tai khong nen ky vong nhieu

- `kv-sharing-fast-prefill`
  - WIP, chua co prefill optimization thuc su.
- `enable-force-include-usage`
  - bench flag, khong phai perf lever.
- `enable-prompt-tokens-details`
  - observability flag, khong phai perf lever.

---

## 7. De xuat bo tach TTFT/TPOT sau hon nua cho repo

Repo hien da co:
- `queue`
- `prefill`
- `decode`
- `residual`

Sau khi trace code, de bo tach sau hon nua nen them timestamp o:

### Frontend
- truoc `render_chat_request`
- sau `render_chat_request`
- sau `tokenization`
- sau `sampling_params build`
- ngay truoc `engine generate`

De co:

```
frontend_prep
= model_check
+ chat_render
+ tokenization
+ mm_preprocess
+ request_build
+ enqueue_handoff
```

### Streaming
- timestamp tai luc engine tao first `RequestOutput`
- timestamp tai luc OpenAI serving tao first SSE chunk
- timestamp tai luc chunk duoc flush

De tach:

```
client_transport
= server_chunk_build
+ sse_flush
+ network_or_local_socket
+ client_parse
```

### Hybrid cache
- log them `resolved block_size`
- `resolved hash_block_size`
- `resolved mamba_block_size`
- `mamba_cache_mode`
- `num_cached_tokens` tach cho attention vs mamba neu co the

De giai thich:
- vi sao prefix hit co the thay doi khi doi `mamba_cache_mode`
- vi sao block hash fine-grain bi mat

---

## 8. Tom tat 1 dong cho tung cum

- Frontend flags co the cuu TTFT that su vi chat render/tokenize xay ra truoc queue.
- Scheduler flags la lever lon nhat cho `queue` va trade-off chinh giua TTFT/TPOT.
- Hybrid cache flags khong chi cache, ma con doi block layout va hash granularity.
- Kernel flags phai phan biet ro:
  - 6 layer attention
  - 18 layer GDN
  - GEMM path cua quantization
- Startup flags nhu `safetensors-load-strategy` co co che implementation ro rang,
  khong phai meo vo can cu.
