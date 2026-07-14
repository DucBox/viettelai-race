# vLLM v0.22.1 Line-by-line Trace Notes

Nguon source trace:
- local image source: `/private/tmp/vllm_0_22_1_src`
- base: `vllm/vllm-openai:v0.22.1`
- v0.24.0 overlay diff: `docs/vllm-v0.24.0-vs-v0.22.1-line-by-line-diff.md`

Muc tieu cua file nay:
- trace theo dung style code-walkthrough: line number, if/else, comment trong code
- khong do benchmark
- khong chi noi "flag co y nghia gi", ma noi "flag di qua nhung nhanh code nao"
- noi tung nhanh code voi TTFT / TPOT / queue / prefill / decode

Quy uoc doc:
- `Lx-Ly` la line number trong source local `/private/tmp/vllm_0_22_1_src`.
- `TTFT` bi anh huong khi request chua co token dau tien.
- `TPOT` bi anh huong khi engine dang decode/stream cac token tiep theo.

---

## 1. Chunked Prefill & Batching

Flags lien quan:
- `--enable-chunked-prefill`
- `--max-num-batched-tokens`
- `--max-num-seqs`
- `--max-num-partial-prefills`
- `--max-long-partial-prefills`
- `--long-prefill-token-threshold`
- `--mamba-cache-mode=align`
- `--block-size`

Source chinh:
- `engine/arg_utils.py`
- `config/scheduler.py`
- `v1/core/sched/scheduler.py`
- `config/vllm.py`

### 1.1 Default resolve: user khong set flag thi vLLM tu quyet

File: `engine/arg_utils.py`, L2365-L2433.

Code flow:

```text
default_chunked_prefill = model_config.is_chunked_prefill_supported
default_prefix_caching = model_config.is_prefix_caching_supported

if self.enable_chunked_prefill is None:
    self.enable_chunked_prefill = default_chunked_prefill
elif generate model + user disables + default says supported:
    warn
elif pooling model + user enables + default says unsupported:
    warn

if self.enable_prefix_caching is None:
    self.enable_prefix_caching = default_prefix_caching
...
if current platform is RISCV CPU:
    force disable chunked prefill and prefix caching
```

Line-by-line meaning:
- L2368-L2369: vLLM khong lay default tu CLI parser raw. No hoi `ModelConfig`.
- L2371-L2377: neu user khong truyen `--enable/--no-enable-chunked-prefill`,
  `model_config.is_chunked_prefill_supported` la nguon su that.
- L2378-L2387: neu model generate support chunked prefill ma user tat, vLLM khong
  chan ngay, nhung warning "may crash or produce incorrect outputs".
- L2388-L2397: voi pooling model, bat chunked prefill khi model khong support cung
  chi warning. Nghia la co flag khong dong nghia voi safe path.
- L2399-L2405: prefix caching cung resolve tu model support neu user khong set.
- L2417-L2433: platform RISCV CPU force tat ca hai, du user co the da de default.

Tac dong:
- TTFT: neu chunked prefill bat, prompt dai co the duoc cat thanh nhieu scheduler
  step, request khac co the chen vao; TTFT p50/p95 co the doi manh.
- TPOT: chunked prefill bat cho phep prefill va decode cung ton tai trong loop,
  nen decode token co the bi canh tranh voi prefill chunk.
- Ket luan tune: dung docs/CLI parser default la chua du. Phai trace sau
  `ModelConfig` va platform rewrite.

### 1.2 `max_num_batched_tokens` khong chi la "batch size"

File: `engine/arg_utils.py`, L2475-L2564.

Code flow:

```text
world_size = pipeline_parallel_size * tensor_parallel_size
defaults = get_batch_defaults(world_size)
orig_max_num_batched_tokens = self.max_num_batched_tokens
orig_max_num_seqs = self.max_num_seqs

if max_num_batched_tokens is None:
    set default by usage context or batched-DP-MoE
if max_num_seqs is None:
    set default by usage context

if performance_mode == "throughput":
    double only values that were originally None

if max_num_batched_tokens originally None:
    if not enable_chunked_prefill:
        max(max_model_len, default)
    if multimodal prefix-LM needs bigger single item:
        raise floor
    cap at max_num_seqs * max_model_len

if max_num_seqs originally None:
    max_num_seqs = min(max_num_seqs, max_num_batched_tokens)
```

Line-by-line meaning:
- L2481: `world_size` co tham gia default. Neu TP/PP doi, default co the doi.
- L2487-L2488: code giu lai "user co set tay khong". Cai nay quan trong.
- L2490-L2505: chi khi `None` moi default. Neu compose da set tay, cac preset
  sau khong override truc tiep.
- L2507-L2512: `performance-mode=throughput` chi double gia tri ban dau la `None`.
  Neu submit da set `--max-num-batched-tokens` va `--max-num-seqs`, throughput
  khong con la nut "tang batch" nua.
- L2518-L2523: neu chunked prefill tat, vLLM bat buoc token budget phai cover
  `max_model_len`, vi khong duoc cat prompt.
- L2525-L2540: multimodal prefix-LM co the nang floor cua token budget.
- L2542-L2548: default budget bi cap boi `max_num_seqs * max_model_len`.
- L2556-L2558: neu user khong set `max_num_seqs`, no bi cap theo token budget.

Tac dong:
- Queue: `max_num_batched_tokens` la per-step token budget cua scheduler.
- Prefill: prompt dai hon budget se thanh partial prefill neu chunked prefill bat.
- Decode: budget lon hon cho phep nhieu decode tokens/prefill tokens cung step,
  nhung co the lam mixed prefill+decode nang hon.

### 1.3 SchedulerConfig validate: nhung cau lenh raise/warn quan trong

File: `config/scheduler.py`, L42-L90 va L224-L308.

Important fields:
- L49-L54: `max_num_batched_tokens` = max tokens processed in one iteration.
- L56-L61: `max_num_scheduled_tokens` co the nho hon batched tokens khi model
  append token vao batch, vi du speculative decoding.
- L70-L78: partial prefill concurrency va long prefill concurrency.
- L80-L82: threshold de coi prompt la "long".
- L84-L90: chunked prefill cat request theo remaining `max_num_batched_tokens`.

Post-init:
- L225-L233: encoder-decoder force tat chunked prefill va prefix caching.
- L235-L236: encoder compute/cache budget mac dinh bang `max_num_batched_tokens`.
- L238-L242: log "Chunked prefill is enabled..." neu bat.
- L244-L247: neu `max_num_partial_prefills > 1` va threshold = 0, vLLM tu set
  threshold = `int(max_model_len * 0.04)`.
- L260-L271: neu chunked prefill tat ma `max_num_batched_tokens < max_model_len`,
  raise. Day la hard invariant.
- L273-L278: `max_num_batched_tokens >= max_num_seqs`; neu khong raise.
- L280-L286: neu budget > `max_num_seqs * max_model_len`, warning vi vo ly.
- L288-L293: partial prefill > 1 yeu cau chunked prefill bat.
- L295-L300: long threshold khong duoc > max model len.
- L302-L306: `max_long_partial_prefills <= max_num_partial_prefills`.

Tac dong:
- TTFT: partial prefill co the giam tail cho prompt ngan khi prompt dai dang doi.
- Queue: validation nay quyet dinh request co duoc engine start khong.
- TPOT: partial prefill concurrency qua cao co the chen nhieu prefill work vao
  decode loop, lam TPOT xau di.

### 1.4 Scheduler init: token_budget lay tu dau

File: `v1/core/sched/scheduler.py`, L102-L108.

Code flow:

```text
self.max_num_running_reqs = scheduler_config.max_num_seqs
self.max_num_scheduled_tokens =
    scheduler_config.max_num_scheduled_tokens
    if set
    else scheduler_config.max_num_batched_tokens
```

Line-by-line meaning:
- L103: `max_num_seqs` la cap so request RUNNING cung luc.
- L104-L108: scheduler khong truc tiep doc CLI flag. No doc resolved
  `scheduler_config.max_num_scheduled_tokens`, fallback ve `max_num_batched_tokens`.
- Neu spec decode hoac config khac lam `max_num_scheduled_tokens` nho hon,
  `max_num_batched_tokens` khong con la budget thuc te duy nhat.

Tac dong:
- Queue wait = request co vao RUNNING duoc khong phu thuoc ca req cap va token cap.
- TTFT = lan dau request duoc schedule phu thuoc `token_budget`.

### 1.5 RUNNING loop: decode va chunked prefill khong tach phase

File: `v1/core/sched/scheduler.py`, L329-L423.

Comment quan trong:
- L330-L339 noi thang: scheduler khong co "decode phase" hay "prefill phase".
  Moi request chi co `num_computed_tokens` va `num_tokens_with_spec`.

Code flow:

```text
token_budget = self.max_num_scheduled_tokens
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
    num_new_tokens =
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens
    if 0 < long_prefill_token_threshold < num_new_tokens:
        num_new_tokens = long_prefill_token_threshold
    num_new_tokens = min(num_new_tokens, token_budget)
    num_new_tokens = min(num_new_tokens, max_model_len - 1 - computed)
    if encoder inputs:
        maybe reduce num_new_tokens
    if need_mamba_block_aligned_split:
        num_new_tokens = _mamba_block_aligned_split(...)
    if num_new_tokens == 0:
        continue
```

Line-by-line meaning:
- L348: `token_budget` set mot lan dau scheduler step.
- L364-L366: RUNNING requests duoc schedule truoc WAITING. Decode dang chay co
  uu tien truoc request moi.
- L385-L389: so token can tinh tiep = target tokens including spec/output
  placeholders - computed tokens.
- L390-L391: long prefill threshold cat chunk truoc khi token budget cat.
- L392: `max_num_batched_tokens`/`max_num_scheduled_tokens` cat tiep.
- L396-L398: spec decode safety, khong cho input position vuot `max_model_len - 1`.
- L404-L416: encoder/MM input co the giam num_new_tokens nua.
- L418-L421: neu hybrid Mamba align path bat, chunk co the bi san xuong block-size.
- L423-L439: neu thanh 0, scheduler skip request nay va tiep tuc request sau.
  Comment L433-L434 ghi ro mot ly do: insufficient budget for block-aligned chunk
  in hybrid models with Mamba cache mode `align`.

Tac dong:
- TPOT: RUNNING loop la noi decode token duoc cap budget. Neu prefill chunk cua
  request running con dai, no cung dung chung loop voi decode.
- TTFT round sau: request bi skip vi aligned chunk = 0 se doi scheduler step sau,
  tang queue_wait.
- Nuance: voi RUNNING request, num_new_tokens = 0 thi `continue`, khong `break`,
  nen FCFS khong strict tuyet doi.

### 1.6 WAITING loop: request moi vao engine nhu the nao

File: `v1/core/sched/scheduler.py`, L544-L699 va L720-L804.

Code flow:

```text
if no preempted requests and unpaused:
    while waiting/skipped_waiting and token_budget > 0:
        if len(running) == max_num_running_reqs:
            break
        request = selected_queue.peek_request()
        handle blocked status / LoRA cap

        if first schedule:
            local prefix cache lookup
            optional external KV lookup
            num_computed_tokens = local + external hits

        if load_kv_async:
            num_new_tokens = 0
        else:
            num_new_tokens = request.num_tokens - num_computed_tokens
            threshold cut
            if not enable_chunked_prefill and num_new_tokens > token_budget:
                break
            num_new_tokens = min(num_new_tokens, token_budget)

        encoder handling
        mamba block aligned split
        allocate_slots(... full_sequence_must_fit=...)
        move request to running
        token_budget -= num_new_tokens
```

Line-by-line meaning:
- L548: WAITING request chi duoc xet khi con token budget.
- L549-L550: neu RUNNING da bang `max_num_seqs`, dung nhan request moi.
- L590-L624: prefix cache lookup chi khi `request.num_computed_tokens == 0`.
  Cached tokens lam giam `num_new_tokens`, tuc giam prefill_runtime va queue.
- L654: raw prefill work = request tokens - computed/cache hit tokens.
- L655-L657: long threshold cat truoc.
- L659-L667: day la nhanh cua `--enable-chunked-prefill`.
  Neu chunked prefill tat va prompt khong fit token_budget, scheduler `break`.
  Nghia la khong schedule request nay, va cung khong tiep tuc request sau trong
  waiting queue o pass nay.
- L669-L670: khi chunked prefill bat, prompt bi cat ve token_budget va assert > 0.
- L690-L698: Mamba align co the bien chunk thanh 0; voi WAITING thi `break`.
- L721-L731: KV block allocation dung `full_sequence_must_fit` neu
  `scheduler_reserve_full_isl` bat.
- L733-L740: neu KV khong du, request khong vao RUNNING.
- L784-L804: request vao RUNNING, record scheduled, tru token_budget, set
  `request.num_computed_tokens = num_computed_tokens`.

Tac dong:
- TTFT = request phai qua: waiting queue -> prefix cache lookup -> chunk decision
  -> Mamba align -> KV allocation -> RUNNING.
- `enable_chunked_prefill=false` co the lam request dai block dau hang va lam
  request sau khong duoc xet trong WAITING loop.
- `enable_chunked_prefill=true` cho request dai an mot phan budget, nhung co the
  tang decode interference.

### 1.7 `_mamba_block_aligned_split`: day la chon chunk nguy hiem nhat

File: `v1/core/sched/scheduler.py`, L279-L327.

Code flow:

```text
assert external computed tokens == 0
num_computed_tokens = request.computed + local_cache_hits + external_hits
if still in prefill/resumed-prefill:
    block_size = cache_config.block_size
    last_cache_position = request.num_tokens - request.num_tokens % block_size
    if use_eagle:
        last_cache_position -= block_size
    after = num_computed_tokens + num_new_tokens
    if after < last_cache_position:
        num_new_tokens = num_new_tokens // block_size * block_size
    elif computed < last_cache_position < after:
        num_new_tokens = last_cache_position - computed
    else:
        pass
return num_new_tokens
```

Line-by-line meaning:
- L286-L288: external KV connector chua verify voi path nay. Neu co external hit
  ma vao day se assert.
- L294-L300: split alignment chi ap dung khi dang prefill/resumed prefill, khong
  phai normal decode.
- L301-L304: comment cuc quan trong. Mamba state cache can chunk multiple of
  `block_size`; ngoai le la neu chunk nho hon block_size thi state khong cache,
  khong can special handling.
- L305-L307: Eagle co prune last matching block, nen last chunk phai >= block_size.
- L308-L309: `block_size` la effective cache block size sau tat ca rewrite,
  khong nhat thiet la CLI `--block-size`.
- L314-L316: neu chunk sau schedule van chua cham last cache position, floor ve
  multiple cua block_size.
- L317-L323: neu chunk vuot qua last cache position, ep chunk ket thuc dung
  `last_cache_position` de cache duoc last chunk.
- L324-L326: neu da la last few tokens sau cache boundary, khong can align.

Correction so voi cach hieu don gian:
- Khong phai moi `num_new_tokens < block_size` deu hang vinh vien.
- Voi RUNNING loop, num_new_tokens = 0 thi skip request va co the schedule request
  sau.
- Voi WAITING loop, num_new_tokens = 0 thi `break`, nen request hien tai co the
  chan pass hien tai.
- Hard validation o `config/vllm.py` L2101-L2109 da yeu cau `block_size <=
  max_num_batched_tokens` va neu threshold > 0 thi threshold >= block_size khi
  `mamba_cache_mode=align`. Nen config qua nho thuong bi bat truoc runtime.

Tac dong:
- TTFT: neu effective block_size bi Mamba alignment nang len lon, token budget nho
  se lam chunk prefill it hon hoac bi break.
- TPOT: align lam scheduler chon chunk size it linh hoat hon, co the de lai
  residual prefill tokens cho step sau.
- Tuning implication: voi Qwen3.5, phai doc effective `cache_config.block_size`,
  khong chi doc CLI `--block-size`.

### 1.8 `_update_after_schedule`: request thanh partial prefill khi nao

File: `v1/core/sched/scheduler.py`, L951-L967.

Line-by-line meaning:
- L951-L958: comment giai thich vi sao update computed tokens sau khi output da
  du thong tin goc cua step.
- L961-L964: moi request scheduled trong step duoc cong
  `num_scheduled_token`.
- L965-L967: `request.is_prefill_chunk = computed < tokens + placeholders`.
  Neu con chua bat kip target tokens, request van la prefill chunk.

Tac dong:
- TTFT: request chua ra token neu con dang chunk prefill.
- TPOT: scheduler step sau se tiep tuc tinh request do nhu RUNNING prefill/decode
  mixed state.

---

## 2. GDN Prefill Backend

Flag lien quan:
- `--gdn-prefill-backend`

Source chinh:
- `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
- `v1/attention/backends/gdn_attn.py`

### 2.1 Backend resolve: flashinfer request khong co nghia la flashinfer active

File: `qwen_gdn_linear_attn.py`, L150-L211.

Docstring L153-L167 noi dieu kien:
- FlashInfer duoc chon khi requested in `["flashinfer", "auto"]`, platform CUDA,
  va mot trong cac dieu kien GPU:
  - Hopper SM90: khong can constraint them.
  - Blackwell SM10.x: `head_k_dim == 128`, CUDA runtime >= 13, va install
    `nvidia-cutlass-dsl-libs-cu13` con intact.
- CuteDSL chi khi requested `"cutedsl"` va Blackwell SM10.x voi `head_k_dim == 128`.

Line-by-line meaning:
- L168-L174: backend user request lay tu `vllm_config.additional_config`.
  CLI `--gdn-prefill-backend` khong thanh field rieng trong SchedulerConfig, ma
  nam trong additional config.
- L176-L177: non-CUDA fallback thang `"triton"`.
- L179-L181: doc `linear_key_head_dim` tu HF config. Qwen3.5 layer dimension
  tham gia backend support.
- L183-L184: init support flags false.
- L186-L187: SM90 bat FlashInfer.
- L188-L194: Blackwell can family 100 + head_k_dim 128 + CUDA runtime >= 13.
- L193: FlashInfer Blackwell con can `_is_libs_cu13_install_intact()`.
- L195-L205: neu cutlass-dsl libs hong hash, warning va fallback.
- L207-L208: requested `flashinfer` hoac `auto` + support true thi active
  `"flashinfer"`.
- L209-L210: requested `cutedsl` + support true thi active `"cutedsl"`.
- L211: tat ca con lai la `"triton"`.

Tac dong:
- Prefill runtime: Qwen3.5 co nhieu GDN/linear-attention layers, nen backend nay
  la prefill lever lon hon `attention-backend`.
- Startup/TTFT request dau: L234-L238 warning FlashInfer GDN prefill JIT compiled,
  first run may take a while. Nghia la request dau co the chiu compile latency.
- Tune implication: phai doc log "Using ... GDN prefill kernel", khong duoc tin
  moi CLI request.

### 2.2 CustomOp dispatch: active backend quyet dinh forward method

File: `qwen_gdn_linear_attn.py`, L290-L312.

Line-by-line meaning:
- L290-L291: register custom op `chunk_gated_delta_rule`.
- L294: lay current vLLM config global; backend decision xay ra khi op init trong
  context config.
- L295-L296: resolve requested backend va active backend.
- L298-L303: neu user requested `flashinfer`/`cutedsl` ma platform khong dung duoc,
  warning fallback.
- L304: log backend decision.
- L306-L307: active flashinfer -> `forward_cuda`.
- L308-L309: active cutedsl -> `forward_cutedsl`.
- L310-L311: con lai -> `forward_native`.

Tac dong:
- Prefill path duoc bind mot lan thanh method, khong phai moi token if/else lai
  chon backend.
- Neu active fallback ve native, CLI `--gdn-prefill-backend=flashinfer` khong co
  perf win.

### 2.3 FlashInfer path: data transforms truoc khi goi binding

File: `qwen_gdn_linear_attn.py`, L241-L287 va L313-L343.

Line-by-line meaning:
- L252-L254: import `flashinfer.gdn_prefill.chunk_gated_delta_rule`.
- L256-L258: neu `use_qk_l2norm_in_kernel`, ap L2 norm cho q/k truoc.
- L260-L266: squeeze batch dimension va contiguous q/k/v/g/beta.
- L267-L269: state, gate, beta cast sang float32.
- L270-L279: goi FlashInfer binding, truyen `exp(g)`, beta, initial_state,
  `cu_seqlens`.
- L280-L287: FlashInfer tra `(output, state)` neu can final state; neu khong thi
  chi output. vLLM unsqueeze ve format FLA.
- L328-L338: `forward_cuda` chi wrapper vao `fi_chunk_gated_delta_rule`.
- L339-L343: neu `core_attn_out` duoc truyen, copy flat output vao buffer co san.

Tac dong:
- Prefill: FlashInfer co them data conversion/cast/contiguous overhead truoc
  binding, nhung kernel chinh co the nhanh hon.
- Accuracy/stability: q/k L2 norm va float32 state/gate/beta la behavior that,
  khong phai chi "kernel nhanh".
- TTFT request dau: JIT compile co the lam first prefill cham.

### 2.4 Native/Triton path va CuteDSL path

File: `qwen_gdn_linear_attn.py`, L345-L416.

Line-by-line meaning:
- L345-L373: native path goi `fla_chunk_gated_delta_rule` voi q/k/v/g/beta,
  initial state, chunk metadata, va optional core output buffer.
- L375-L416: cutedsl path import in-tree cutedsl op, L394-L396 cung L2 norm q/k,
  L398-L400 assert bat buoc co `cu_seqlens`, `chunk_indices`, `chunk_offsets`.
- L402-L413: goi `chunk_gated_delta_rule_cutedsl`.
- L414-L416: neu khong can final state, set final_state None.

Tac dong:
- Triton/FLA thuong tuong thich rong hon, nhung co JIT/metadata cost rieng.
- CuteDSL yeu cau metadata day du va hardware rat cu the.

### 2.5 Metadata path: GDN prefill can chunk metadata truoc kernel

File: `v1/attention/backends/gdn_attn.py`, L323-L472.

Line-by-line meaning:
- L323-L324: init `chunk_indices`/`chunk_offsets`.
- L325-L327: chi khi co prefill (`num_prefills > 0`) moi prepare FLA chunk ops.
- L328-L340: neu backend cutedsl, goi `prepare_metadata_cutedsl`.
- L341-L357: con lai prepare chunk indices/offsets tren CPU roi async-copy sang GPU
  de tranh GPU->CPU sync `.tolist()`.
- L359-L369: neu prefill, tinh `has_initial_state` va causal conv metadata.
- L373-L377: assert khong vua non-spec decode vua spec decode trong cung function
  counted path.
- L379-L448: neu full cudagraph decode-only, copy state index tensors vao buffer
  padded san.
- L450-L472: pack tat ca metadata vao `GDNAttentionMetadata`.

Tac dong:
- Prefill runtime khong chi la kernel; con co metadata prepare, CPU/GPU copy,
  causal conv metadata, initial state handling.
- TPOT: full cudagraph path trong comment L475-L481 noi hien tai only decode
  supported for full cudagraphs with Mamba. GDN prefill van la path rieng.

---

## 3. Qwen3.5 `mamba-cache-mode`

Flags lien quan:
- `--enable-prefix-caching`
- `--mamba-cache-mode`
- `--mamba-block-size`
- `--block-size`
- `--max-num-batched-tokens`

Source chinh:
- `config/cache.py`
- `model_executor/models/config.py`
- `model_executor/models/qwen3_5.py`
- `config/vllm.py`
- `platforms/interface.py`

### 3.1 CacheConfig comment: 3 mode khac nhau that su

File: `config/cache.py`, L120-L140.

Line-by-line meaning:
- L120-L123: `mamba_block_size` chi set duoc khi prefix caching bat; comment noi
  multiple of 8 de align causal_conv1d kernel.
- L124-L131: `mamba_cache_dtype` va `mamba_ssm_cache_dtype` co scope khac nhau:
  conv+ssm vs chi ssm.
- L132-L140: `mamba_cache_mode` default `"none"`.
  - `none`: prefix caching disabled.
  - `all`: cache Mamba state tai moi vi tri `i * block_size`; default cho model
    support khi prefix caching enabled.
  - `align`: chi cache Mamba state cua last token cua moi scheduler step va khi
    token o vi tri `i * block_size`.

Tac dong:
- `all` va `align` khac co ban: `all` co tham vong snapshot moi block position,
  `align` an theo boundary cua scheduler step.
- Qwen3.5 khong the doc nhu full attention prefix cache.

### 3.2 MambaModelConfig rewrite: user set `all` co the bi fallback `align`

File: `model_executor/models/config.py`, L337-L395.

Line-by-line meaning:
- L350: chi vao nhanh nay neu `cache_config.enable_prefix_caching` true.
- L351-L354: neu mode van `"none"`, vLLM tu set:
  - `"all"` neu `model_config.supports_mamba_prefix_caching`
  - `"align"` neu khong support
- L361-L370: neu mode la `"all"` nhung model khong support Mamba prefix caching,
  code doi thanh `"align"` va warning fallback.
- L371-L374: neu mode la `"align"`, bat buoc chunked prefill enabled.
- L375-L381: log experimental warning.
- L382-L386: neu `mamba_block_size` chua set, set bang `cache_config.block_size`.
- L387-L394: neu prefix caching disabled, force `mamba_cache_mode="none"` va neu
  block size chua set thi set mamba block size = max_model_len.

Tac dong:
- Neu Qwen3.5 `supports_mamba_prefix_caching=False`, set `--mamba-cache-mode=all`
  co the bi rewrite thanh `align` truoc khi model init.
- Phai xem resolved config/log, khong chi xem compose.
- TTFT round sau phu thuoc mode sau rewrite.

### 3.3 Qwen3.5 hard guard: neu `all` van toi model init thi crash

File: `model_executor/models/qwen3_5.py`, L452-L463.

Line-by-line meaning:
- L452-L456: constructor lay HF text config, vLLM config, model config, cache config.
- L458: lay scheduler_config.
- L459-L463: neu `cache_config.mamba_cache_mode == "all"`, raise
  `NotImplementedError` voi message yeu cau dung `--mamba-cache-mode=align`.

Tac dong:
- Qwen3.5 v0.22.1 khong support mode `all` trong model class.
- Neu config rewrite khong fallback duoc, engine fail at startup, khong co
  TTFT/TPOT de toi uu.
- Tuning implication: voi Qwen3.5, shortlist hop ly la `none` vs `align`, khong
  phai `all` vs `align` neu source nay la source submit.

### 3.4 Align-mode validation: block_size phai fit token budget

File: `config/vllm.py`, L2100-L2118.

Line-by-line meaning:
- L2101: chi validate neu mode = `align`.
- L2102-L2107: assert `block_size <= max_num_batched_tokens`.
- L2108-L2109: neu `long_prefill_token_threshold > 0`, threshold phai >= block_size.
- L2110-L2114: `disable_chunked_mm_input` khong duoc true, vi align can flexibility
  de schedule multiple of block_size token ke ca o giua MM input.
- L2115-L2118: V2 model runner chua support `mamba_cache_mode='align'`.

Tac dong:
- Config sai se fail truoc runtime scheduler.
- Neu effective block_size bi platform/Mamba alignment nang len, budget/threshold
  phai nang theo.
- TTFT: budget qua sat block_size lam scheduler it linh hoat, de chunk prefill
  nho va nhieu step hon.

### 3.5 Qwen3.5 mamba SSM dtype rewrite

File: `model_executor/models/config.py`, L536-L560.

Line-by-line meaning:
- L539-L542: docstring noi Qwen3.5 co field `mamba_ssm_dtype` trong HF config.
- L544-L546: lay cache_config va hf_text_config.
- L547-L550: neu user de `mamba_ssm_cache_dtype=auto`, vLLM set theo
  `hf_text_config.mamba_ssm_dtype` neu co.
- L550-L560: neu user override khac HF config, warning va dung user value.

Tac dong:
- `--mamba-ssm-cache-dtype=auto` khong co nghia la model dtype chung; Qwen3.5 co
  rewrite rieng.
- TPOT/prefill co the bi anh huong qua Mamba/GDN state bandwidth va numerical
  behavior.

---

## 4. Cach tiep tuc trace 50 flags theo dung chuan nay

Voi moi flag/cum flag, format bat buoc:

```text
1. CLI/parser line
2. EngineArgs resolve line
3. Config dataclass field/comment line
4. __post_init__/verify/rewrite line
5. Runtime consumer line
6. Moi if/elif/else quan trong
7. Comment trong source neu no giai thich invariant
8. TTFT/TPOT component bi anh huong
9. Qwen3.5-specific caveat
10. Tuning conclusion
```

Nhom tiep theo nen trace bang format nay:
- `kv-cache-dtype=fp8` + `calculate-kv-scales` + `kv-cache-dtype-skip-layers`
- `block-size` + `_align_hybrid_block_size`
- `enable-prefix-caching` + block hash + hash algo
- `gpu-memory-utilization` + `kv-cache-memory-bytes` + GPUWorker profiling
- `cudagraph-*` + `performance-mode` + `spec-tokens`
- `quantization=fp8` + `linear-backend`
- frontend residual: chat template, tokenizer, renderer workers, logging
