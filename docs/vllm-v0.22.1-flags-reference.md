# vLLM v0.22.1 — Toàn bộ CLI flags của `api_server` (trích xuất từ source thật)

Trích xuất **trực tiếp từ argparse parser thật** bên trong image `duc0811/qwen35-2b-race:v1` (base `vllm/vllm-openai:v0.22.1`), không phải chép từ tài liệu online — vì version pin theo BTC có thể lệch so với vLLM public docs. Cách trích xuất: patch `current_platform` để vượt qua bước auto-detect GPU (máy dev không có CUDA thật), build parser thật qua `make_arg_parser()`, rồi đọc `.default`/`.choices` của từng `argparse.Action`. Xem cuối file để tái tạo.

**Tổng số flag: 261** (không tính `-h/--help`).

---

## ⚠️ Phát hiện quan trọng — đính chính so với `docs/vllm-optimization-guide.md`


Đối chiếu số liệu thật với giả định trong guide cũ, có 2 chỗ **sai lệch đáng kể**:

| Flag | Guide cũ giả định | **Giá trị mặc định THẬT (đo được)** | Hệ quả |
|---|---|---|---|
| `--mamba-cache-mode` | mặc định `align` (có bug đã biết) | **mặc định `none`** | Nghĩa là **toàn bộ baseline/v2/v3 (chưa từng set cờ này) đang chạy với mamba/GDN state cache TẮT HOÀN TOÀN** — không phải "align" như tưởng. 18/24 layer GDN không hề tái sử dụng state giữa các request, dù `--enable-prefix-caching` đã bật (cờ đó chỉ tái dùng KV cache của 6 layer full-attention). Đây có thể là lý do TTFT round 2-6 vẫn cao dù đã bật prefix caching — **phần lớn kiến trúc (18/24 layer) chưa từng được cache**. |
| `--gdn-prefill-backend` | mặc định `triton` | **mặc định `None`** (tự chọn nội bộ, không cố định `triton`) | Việc v5 đổi sang `flashinfer` là so sánh với một baseline nội bộ không rõ ràng (`None`), không hẳn là so với `triton` như giả định — có thể giải thích vì sao đổi sang `flashinfer` không thấy cải thiện rõ (nếu nội bộ vốn đã tự chọn kernel tối ưu, đổi thủ công có khi trùng hoặc tệ hơn lựa chọn tự động). |

**Khuyến nghị ngay:** `v10` (V2 + `--mamba-cache-mode=all`) đang test đúng trọng tâm —
nhưng giờ biết rằng đối chiếu phải là **`none` (mặc định thật) vs `all`**, không phải
"align vs all" như comment cũ trong file v4 giả định. Đây là phép so sánh **quan
trọng nhất** trong toàn bộ sweep vì nó động chạm đến toàn bộ 18/24 layer của kiến
trúc, không chỉ 6 layer full-attention.

---

## Toàn bộ flags (sắp xếp theo thứ tự parser, giữ nhóm Frontend/Engine gốc)

### `--headless`
- **default:** `False`
- **type/nargs:** `bool`
- **help:** Run in headless mode. See multi-node data parallel documentation for more details.

### `--api-server-count, -asc`
- **default:** `None`
- **type/nargs:** `int`
- **help:** How many API server processes to run. Defaults to data_parallel_size if not specified.

### `--config`
- **default:** `None`
- **type/nargs:** `str`
- **help:** Read CLI options from a config file. Must be a YAML with the following options: https://docs.vllm.ai/en/latest/configuration/serve_args.html

### `--grpc`
- **default:** `False`
- **type/nargs:** `bool`
- **help:** Launch a gRPC server instead of the HTTP OpenAI-compatible server. Requires: pip install vllm[grpc].

### `--lora-modules`
- **default:** `None`
- **type/nargs:** `_optional_type` · nargs=`+`

### `--chat-template`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--chat-template-content-format`
- **default:** `auto` · choices: `auto`, `openai`, `string`
- **type/nargs:** `str`

### `--trust-request-chat-template, --no-trust-request-chat-template`
- **default:** `False`
- **type/nargs:** `bool`

### `--default-chat-template-kwargs`
- **default:** `None`
- **type/nargs:** `loads`
- **help:** Should either be a valid JSON string or JSON keys passed individually.

### `--response-role`
- **default:** `assistant`
- **type/nargs:** `str`

### `--return-tokens-as-token-ids, --no-return-tokens-as-token-ids`
- **default:** `False`
- **type/nargs:** `bool`

### `--enable-auto-tool-choice, --no-enable-auto-tool-choice`
- **default:** `False`
- **type/nargs:** `bool`

### `--exclude-tools-when-tool-choice-none, --no-exclude-tools-when-tool-choice-none`
- **default:** `False`
- **type/nargs:** `bool`

### `--tool-call-parser`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--tool-parser-plugin`
- **default:** `""`
- **type/nargs:** `str`

### `--tool-server`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--log-config-file`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--max-log-len`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--enable-prompt-tokens-details, --no-enable-prompt-tokens-details`
- **default:** `False`
- **type/nargs:** `bool`

### `--enable-server-load-tracking, --no-enable-server-load-tracking`
- **default:** `False`
- **type/nargs:** `bool`

### `--enable-force-include-usage, --no-enable-force-include-usage`
- **default:** `False`
- **type/nargs:** `bool`

### `--enable-tokenizer-info-endpoint, --no-enable-tokenizer-info-endpoint`
- **default:** `False`
- **type/nargs:** `bool`

### `--enable-log-outputs, --no-enable-log-outputs`
- **default:** `False`
- **type/nargs:** `bool`

### `--enable-log-deltas, --no-enable-log-deltas`
- **default:** `True`
- **type/nargs:** `bool`

### `--log-error-stack, --no-log-error-stack`
- **default:** `False`
- **type/nargs:** `bool`

### `--tokens-only, --no-tokens-only`
- **default:** `False`
- **type/nargs:** `bool`

### `--fingerprint-mode`
- **default:** `full` · choices: `custom`, `full`, `hash`, `none`
- **type/nargs:** `str`

### `--fingerprint-value`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--host`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--port`
- **default:** `8000`
- **type/nargs:** `int`

### `--data-parallel-supervisor-port`
- **default:** `9256`
- **type/nargs:** `int`

### `--dp-supervisor-probe-interval-s`
- **default:** `5.0`
- **type/nargs:** `float`

### `--dp-supervisor-probe-timeout-s`
- **default:** `5.0`
- **type/nargs:** `float`

### `--dp-supervisor-probe-failure-threshold`
- **default:** `3`
- **type/nargs:** `int`

### `--uds`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--uvicorn-log-level`
- **default:** `info` · choices: `critical`, `debug`, `error`, `info`, `trace`, `warning`
- **type/nargs:** `str`

### `--disable-uvicorn-access-log, --no-disable-uvicorn-access-log`
- **default:** `False`
- **type/nargs:** `bool`

### `--disable-access-log-for-endpoints`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--allow-credentials, --no-allow-credentials`
- **default:** `False`
- **type/nargs:** `bool`

### `--allowed-origins`
- **default:** `['*']`
- **type/nargs:** `loads`

### `--allowed-methods`
- **default:** `['*']`
- **type/nargs:** `loads`

### `--allowed-headers`
- **default:** `['*']`
- **type/nargs:** `loads`

### `--api-key`
- **default:** `None`
- **type/nargs:** `_optional_type` · nargs=`+`

### `--ssl-keyfile`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--ssl-certfile`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--ssl-ca-certs`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--enable-ssl-refresh, --no-enable-ssl-refresh`
- **default:** `False`
- **type/nargs:** `bool`

### `--ssl-cert-reqs`
- **default:** `0`
- **type/nargs:** `int`

### `--ssl-ciphers`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--root-path`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--middleware`
- **default:** `[]`
- **type/nargs:** `str`

### `--enable-request-id-headers, --no-enable-request-id-headers`
- **default:** `False`
- **type/nargs:** `bool`

### `--disable-fastapi-docs, --no-disable-fastapi-docs`
- **default:** `False`
- **type/nargs:** `bool`

### `--h11-max-incomplete-event-size`
- **default:** `4194304`
- **type/nargs:** `int`

### `--h11-max-header-count`
- **default:** `256`
- **type/nargs:** `int`

### `--enable-offline-docs, --no-enable-offline-docs`
- **default:** `False`
- **type/nargs:** `bool`

### `--enable-flash-late-interaction, --no-enable-flash-late-interaction`
- **default:** `True`
- **type/nargs:** `bool`

### `--model`
- **default:** `Qwen/Qwen3-0.6B`
- **type/nargs:** `str`

### `--runner`
- **default:** `auto` · choices: `auto`, `draft`, `generate`, `pooling`
- **type/nargs:** `str`

### `--convert`
- **default:** `auto` · choices: `auto`, `classify`, `embed`, `none`
- **type/nargs:** `str`

### `--tokenizer`
- **default:** `None`
- **type/nargs:** `str`

### `--tokenizer-mode`
- **default:** `auto`
- **type/nargs:** `str`

### `--trust-remote-code, --no-trust-remote-code`
- **default:** `False`
- **type/nargs:** `bool`

### `--dtype`
- **default:** `auto` · choices: `auto`, `bfloat16`, `float`, `float16`, `float32`, `half`
- **type/nargs:** `str`

### `--seed`
- **default:** `0`
- **type/nargs:** `int`

### `--hf-config-path`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--allowed-local-media-path`
- **default:** `""`
- **type/nargs:** `str`

### `--allowed-media-domains`
- **default:** `None`
- **type/nargs:** `_optional_type` · nargs=`+`

### `--revision`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--code-revision`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--tokenizer-revision`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--max-model-len`
- **default:** `None`
- **type/nargs:** `human_readable_int_or_auto`
- **help:** Parse human-readable integers like '1k', '2M', etc.     Including decimal values with decimal multipliers.     Also accepts -1 or 'auto' as a special value for auto-detection.      Examples:     - '

### `--quantization, -q`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--quantization-config`
- **default:** `None`
- **type/nargs:** `_optional_type`
- **help:** API docs: https://docs.vllm.ai/en/v0.22.1/api/vllm/config/#vllm.config.QuantizationConfigArgs  Should either be a valid JSON string or JSON keys passed individually.

### `--allow-deprecated-quantization, --no-allow-deprecated-quantization`
- **default:** `False`
- **type/nargs:** `bool`

### `--enforce-eager, --no-enforce-eager`
- **default:** `False`
- **type/nargs:** `bool`

### `--enable-return-routed-experts, --no-enable-return-routed-experts`
- **default:** `False`
- **type/nargs:** `bool`

### `--max-logprobs`
- **default:** `20`
- **type/nargs:** `int`

### `--logprobs-mode`
- **default:** `raw_logprobs` · choices: `processed_logits`, `processed_logprobs`, `raw_logits`, `raw_logprobs`
- **type/nargs:** `str`

### `--use-fp64-gumbel, --no-use-fp64-gumbel`
- **default:** `False`
- **type/nargs:** `bool`

### `--disable-sliding-window, --no-disable-sliding-window`
- **default:** `False`
- **type/nargs:** `bool`

### `--disable-cascade-attn, --no-disable-cascade-attn`
- **default:** `True`
- **type/nargs:** `bool`

### `--skip-tokenizer-init, --no-skip-tokenizer-init`
- **default:** `False`
- **type/nargs:** `bool`

### `--enable-prompt-embeds, --no-enable-prompt-embeds`
- **default:** `False`
- **type/nargs:** `bool`

### `--served-model-name`
- **default:** `None`
- **type/nargs:** `_optional_type` · nargs=`+`

### `--config-format`
- **default:** `auto`
- **type/nargs:** `str`

### `--hf-token`
- **default:** `None`
- **type/nargs:** `str`

### `--hf-overrides`
- **default:** `{}`
- **type/nargs:** `union_dict_and_str`

### `--pooler-config`
- **default:** `None`
- **type/nargs:** `_optional_type`
- **help:** API docs: https://docs.vllm.ai/en/v0.22.1/api/vllm/config/#vllm.config.PoolerConfig  Should either be a valid JSON string or JSON keys passed individually.

### `--generation-config`
- **default:** `auto`
- **type/nargs:** `str`

### `--override-generation-config`
- **default:** `{}`
- **type/nargs:** `_parse_type`
- **help:** Should either be a valid JSON string or JSON keys passed individually.

### `--enable-sleep-mode, --no-enable-sleep-mode`
- **default:** `False`
- **type/nargs:** `bool`

### `--enable-cumem-allocator, --no-enable-cumem-allocator`
- **default:** `False`
- **type/nargs:** `bool`

### `--model-impl`
- **default:** `auto`
- **type/nargs:** `str`

### `--override-attention-dtype`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--logits-processors`
- **default:** `None`
- **type/nargs:** `_optional_type` · nargs=`+`

### `--io-processor-plugin`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--renderer-num-workers`
- **default:** `1`
- **type/nargs:** `int`

### `--load-format`
- **default:** `auto`
- **type/nargs:** `str`

### `--download-dir`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--safetensors-load-strategy`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--safetensors-prefetch-num-threads`
- **default:** `8`
- **type/nargs:** `int`

### `--safetensors-prefetch-block-size`
- **default:** `16777216`
- **type/nargs:** `human_readable_int`
- **help:** Parse human-readable integers like '1k', '2M', etc.     Including decimal values with decimal multipliers.      Examples:     - '1k' -> 1,000     - '1K' -> 1,024     - '25.6k' -> 25,600

### `--model-loader-extra-config`
- **default:** `{}`
- **type/nargs:** `union_dict_and_str`

### `--ignore-patterns`
- **default:** `['original/**/*']`
- **type/nargs:** `str` · nargs=`+`

### `--use-tqdm-on-load, --no-use-tqdm-on-load`
- **default:** `True`
- **type/nargs:** `bool`

### `--pt-load-map-location`
- **default:** `cpu`
- **type/nargs:** `union_dict_and_str`

### `--attention-backend`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--mamba-backend`
- **default:** `MambaBackendEnum.TRITON`
- **type/nargs:** `str`

### `--enable-mamba-cache-stochastic-rounding, --no-enable-mamba-cache-stochastic-rounding`
- **default:** `False`
- **type/nargs:** `bool`

### `--mamba-cache-philox-rounds`
- **default:** `0`
- **type/nargs:** `int`

### `--reasoning-parser`
- **default:** `""`
- **type/nargs:** `str`

### `--reasoning-parser-plugin`
- **default:** `""`
- **type/nargs:** `str`

### `--distributed-executor-backend`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--pipeline-parallel-size, -pp`
- **default:** `1`
- **type/nargs:** `int`

### `--master-addr`
- **default:** `127.0.0.1`
- **type/nargs:** `str`

### `--master-port`
- **default:** `29501`
- **type/nargs:** `int`

### `--nnodes, -n`
- **default:** `1`
- **type/nargs:** `int`

### `--node-rank, -r`
- **default:** `0`
- **type/nargs:** `int`

### `--distributed-timeout-seconds`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--cpu-distributed-timeout-seconds`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--numa-bind, --no-numa-bind`
- **default:** `False`
- **type/nargs:** `bool`

### `--numa-bind-nodes`
- **default:** `None`
- **type/nargs:** `_optional_type` · nargs=`+`

### `--numa-bind-cpus`
- **default:** `None`
- **type/nargs:** `_optional_type` · nargs=`+`

### `--tensor-parallel-size, -tp`
- **default:** `1`
- **type/nargs:** `int`

### `--decode-context-parallel-size, -dcp`
- **default:** `1`
- **type/nargs:** `int`

### `--dcp-comm-backend`
- **default:** `ag_rs` · choices: `a2a`, `ag_rs`
- **type/nargs:** `str`

### `--dcp-kv-cache-interleave-size`
- **default:** `1`
- **type/nargs:** `int`

### `--cp-kv-cache-interleave-size`
- **default:** `1`
- **type/nargs:** `int`

### `--prefill-context-parallel-size, -pcp`
- **default:** `1`
- **type/nargs:** `int`

### `--data-parallel-size, -dp`
- **default:** `1`
- **type/nargs:** `int`

### `--data-parallel-rank, -dpn`
- **default:** `None`
- **type/nargs:** `int`
- **help:** Data parallel rank of this instance. When set, enables external load balancer mode for MoE data-parallel deployments. Unsupported for non-MoE models; launch independent vLLM instances instead.

### `--data-parallel-start-rank, -dpr`
- **default:** `None`
- **type/nargs:** `int`
- **help:** Starting data parallel rank for secondary nodes.

### `--data-parallel-size-local, -dpl`
- **default:** `None`
- **type/nargs:** `int`
- **help:** Number of data parallel replicas to run on this node.

### `--data-parallel-address, -dpa`
- **default:** `None`
- **type/nargs:** `str`
- **help:** Address of data parallel cluster head-node.

### `--data-parallel-rpc-port, -dpp`
- **default:** `None`
- **type/nargs:** `int`
- **help:** Port for data parallel RPC communication.

### `--data-parallel-backend, -dpb`
- **default:** `mp`
- **type/nargs:** `str`
- **help:** Backend for data parallel, either "mp" or "ray".

### `--data-parallel-hybrid-lb, --no-data-parallel-hybrid-lb, -dph`
- **default:** `False`
- **type/nargs:** `bool`

### `--data-parallel-external-lb, --no-data-parallel-external-lb, -dpe`
- **default:** `False`
- **type/nargs:** `bool`

### `--data-parallel-multi-port-external-lb, -dpm`
- **default:** `False`
- **type/nargs:** `bool`
- **help:** Run a node-local supervisor that launches one external-LB API server per local data parallel rank and exposes aggregated health on a supervisor port.

### `--enable-expert-parallel, --no-enable-expert-parallel, -ep`
- **default:** `False`
- **type/nargs:** `bool`

### `--enable-ep-weight-filter, --no-enable-ep-weight-filter`
- **default:** `False`
- **type/nargs:** `bool`

### `--all2all-backend`
- **default:** `allgather_reducescatter` · choices: `allgather_reducescatter`, `deepep_high_throughput`, `deepep_low_latency`, `flashinfer_all2allv`, `flashinfer_nvlink_one_sided`, `flashinfer_nvlink_two_sided`, `mori`, `naive`, `nixl_ep`, `pplx`
- **type/nargs:** `str`

### `--enable-dbo, --no-enable-dbo`
- **default:** `False`
- **type/nargs:** `bool`

### `--ubatch-size`
- **default:** `0`
- **type/nargs:** `int`

### `--enable-elastic-ep, --no-enable-elastic-ep`
- **default:** `False`
- **type/nargs:** `bool`

### `--dbo-decode-token-threshold`
- **default:** `32`
- **type/nargs:** `int`

### `--dbo-prefill-token-threshold`
- **default:** `512`
- **type/nargs:** `int`

### `--disable-nccl-for-dp-synchronization, --no-disable-nccl-for-dp-synchronization`
- **default:** `None`
- **type/nargs:** `bool`

### `--enable-eplb, --no-enable-eplb`
- **default:** `False`
- **type/nargs:** `bool`

### `--eplb-config`
- **default:** `EPLBConfig(window_size=1000, step_interval=3000, num_redundant_experts=0, log_balancedness=False, log_balancedness_interval=1, use_async=False, policy='default', communicator=None)`
- **type/nargs:** `parse_dataclass`
- **help:** API docs: https://docs.vllm.ai/en/v0.22.1/api/vllm/config/#vllm.config.EPLBConfig  Should either be a valid JSON string or JSON keys passed individually.

### `--expert-placement-strategy`
- **default:** `linear` · choices: `linear`, `round_robin`
- **type/nargs:** `str`

### `--max-parallel-loading-workers`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--ray-workers-use-nsight, --no-ray-workers-use-nsight`
- **default:** `False`
- **type/nargs:** `bool`

### `--disable-custom-all-reduce, --no-disable-custom-all-reduce`
- **default:** `False`
- **type/nargs:** `bool`

### `--worker-cls`
- **default:** `auto`
- **type/nargs:** `str`

### `--worker-extension-cls`
- **default:** `""`
- **type/nargs:** `str`

### `--block-size`
- **default:** `None`
- **type/nargs:** `int`

### `--gpu-memory-utilization`
- **default:** `0.92`
- **type/nargs:** `float`

### `--kv-cache-memory-bytes`
- **default:** `None`
- **type/nargs:** `_optional_type`
- **help:** Parse human-readable integers like '1k', '2M', etc.     Including decimal values with decimal multipliers.      Examples:     - '1k' -> 1,000     - '1K' -> 1,024     - '25.6k' -> 25,600

### `--kv-cache-dtype`
- **default:** `auto` · choices: `auto`, `bfloat16`, `float16`, `fp8`, `fp8_ds_mla`, `fp8_e4m3`, `fp8_e5m2`, `fp8_inc`, `fp8_per_token_head`, `int8_per_token_head`, `nvfp4`, `turboquant_3bit_nc`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_k8v4`
- **type/nargs:** `str`

### `--num-gpu-blocks-override`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--enable-prefix-caching, --no-enable-prefix-caching`
- **default:** `None`
- **type/nargs:** `bool`

### `--prefix-caching-hash-algo`
- **default:** `sha256` · choices: `sha256`, `sha256_cbor`, `xxhash`, `xxhash_cbor`
- **type/nargs:** `str`

### `--calculate-kv-scales, --no-calculate-kv-scales`
- **default:** `False`
- **type/nargs:** `bool`

### `--kv-cache-dtype-skip-layers`
- **default:** `[]`
- **type/nargs:** `str` · nargs=`+`

### `--kv-sharing-fast-prefill, --no-kv-sharing-fast-prefill`
- **default:** `False`
- **type/nargs:** `bool`

### `--mamba-cache-dtype`
- **default:** `auto` · choices: `auto`, `bfloat16`, `float16`, `float32`
- **type/nargs:** `str`

### `--mamba-ssm-cache-dtype`
- **default:** `auto` · choices: `auto`, `bfloat16`, `float16`, `float32`
- **type/nargs:** `str`

### `--mamba-block-size`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--mamba-cache-mode`
- **default:** `none` · choices: `align`, `all`, `none`
- **type/nargs:** `str`

### `--kv-offloading-size`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--kv-offloading-backend`
- **default:** `native` · choices: `lmcache`, `native`
- **type/nargs:** `str`

### `--offload-backend`
- **default:** `auto` · choices: `auto`, `prefetch`, `uva`
- **type/nargs:** `str`

### `--cpu-offload-gb`
- **default:** `0`
- **type/nargs:** `float`

### `--cpu-offload-params`
- **default:** `set()`
- **type/nargs:** `str` · nargs=`+`

### `--offload-group-size`
- **default:** `0`
- **type/nargs:** `int`

### `--offload-num-in-group`
- **default:** `1`
- **type/nargs:** `int`

### `--offload-prefetch-step`
- **default:** `1`
- **type/nargs:** `int`

### `--offload-params`
- **default:** `set()`
- **type/nargs:** `str` · nargs=`+`

### `--language-model-only, --no-language-model-only`
- **default:** `False`
- **type/nargs:** `bool`

### `--limit-mm-per-prompt`
- **default:** `{}`
- **type/nargs:** `_parse_type`
- **help:** Should either be a valid JSON string or JSON keys passed individually.

### `--enable-mm-embeds, --no-enable-mm-embeds`
- **default:** `False`
- **type/nargs:** `bool`

### `--media-io-kwargs`
- **default:** `{}`
- **type/nargs:** `_parse_type`
- **help:** Should either be a valid JSON string or JSON keys passed individually.

### `--mm-processor-kwargs`
- **default:** `None`
- **type/nargs:** `_optional_type`
- **help:** Should either be a valid JSON string or JSON keys passed individually.

### `--mm-processor-cache-gb`
- **default:** `4`
- **type/nargs:** `float`

### `--mm-processor-cache-type`
- **default:** `lru` · choices: `lru`, `shm`
- **type/nargs:** `str`

### `--mm-shm-cache-max-object-size-mb`
- **default:** `128`
- **type/nargs:** `int`

### `--mm-encoder-only, --no-mm-encoder-only`
- **default:** `False`
- **type/nargs:** `bool`

### `--mm-encoder-tp-mode`
- **default:** `weights` · choices: `data`, `weights`
- **type/nargs:** `str`

### `--mm-encoder-attn-backend`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--mm-encoder-attn-dtype`
- **default:** `None` · choices: `fp8`, `None`
- **type/nargs:** `_optional_type`

### `--mm-encoder-fp8-scale-path`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--mm-encoder-fp8-scale-save-path`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--mm-encoder-fp8-scale-save-margin`
- **default:** `1.5`
- **type/nargs:** `float`

### `--interleave-mm-strings, --no-interleave-mm-strings`
- **default:** `False`
- **type/nargs:** `bool`

### `--skip-mm-profiling, --no-skip-mm-profiling`
- **default:** `False`
- **type/nargs:** `bool`

### `--video-pruning-rate`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--mm-tensor-ipc`
- **default:** `direct_rpc` · choices: `direct_rpc`, `torch_shm`
- **type/nargs:** `str`

### `--enable-lora, --no-enable-lora`
- **default:** `None`
- **type/nargs:** `bool`
- **help:** If True, enable handling of LoRA adapters.

### `--max-loras`
- **default:** `1`
- **type/nargs:** `int`

### `--max-lora-rank`
- **default:** `16` · choices: `1`, `8`, `16`, `32`, `64`, `128`, `256`, `320`, `512`
- **type/nargs:** `int`

### `--lora-dtype`
- **default:** `auto` · choices: `auto`, `bfloat16`, `float16`
- **type/nargs:** `str`

### `--enable-tower-connector-lora, --no-enable-tower-connector-lora`
- **default:** `False`
- **type/nargs:** `bool`

### `--max-cpu-loras`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--fully-sharded-loras, --no-fully-sharded-loras`
- **default:** `False`
- **type/nargs:** `bool`

### `--lora-target-modules`
- **default:** `None`
- **type/nargs:** `_optional_type` · nargs=`+`

### `--default-mm-loras`
- **default:** `None`
- **type/nargs:** `_optional_type`
- **help:** Should either be a valid JSON string or JSON keys passed individually.

### `--specialize-active-lora, --no-specialize-active-lora`
- **default:** `False`
- **type/nargs:** `bool`

### `--enable-mixed-moe-lora-format, --no-enable-mixed-moe-lora-format`
- **default:** `False`
- **type/nargs:** `bool`

### `--show-hidden-metrics-for-version`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--otlp-traces-endpoint`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--collect-detailed-traces`
- **default:** `None` · choices: `all`, `model`, `worker`, `None`, `model,worker`, `model,all`, `worker,model`, `worker,all`, `all,model`, `all,worker`
- **type/nargs:** `_optional_type` · nargs=`+`

### `--kv-cache-metrics, --no-kv-cache-metrics`
- **default:** `False`
- **type/nargs:** `bool`

### `--kv-cache-metrics-sample`
- **default:** `0.01`
- **type/nargs:** `float`

### `--cudagraph-metrics, --no-cudagraph-metrics`
- **default:** `False`
- **type/nargs:** `bool`

### `--enable-layerwise-nvtx-tracing, --no-enable-layerwise-nvtx-tracing`
- **default:** `False`
- **type/nargs:** `bool`

### `--enable-mfu-metrics, --no-enable-mfu-metrics`
- **default:** `False`
- **type/nargs:** `bool`

### `--enable-logging-iteration-details, --no-enable-logging-iteration-details`
- **default:** `False`
- **type/nargs:** `bool`

### `--max-num-batched-tokens`
- **default:** `None`
- **type/nargs:** `human_readable_int`
- **help:** Parse human-readable integers like '1k', '2M', etc.     Including decimal values with decimal multipliers.      Examples:     - '1k' -> 1,000     - '1K' -> 1,024     - '25.6k' -> 25,600

### `--max-num-seqs`
- **default:** `None`
- **type/nargs:** `int`

### `--max-num-partial-prefills`
- **default:** `1`
- **type/nargs:** `int`

### `--max-long-partial-prefills`
- **default:** `1`
- **type/nargs:** `int`

### `--long-prefill-token-threshold`
- **default:** `0`
- **type/nargs:** `int`

### `--scheduling-policy`
- **default:** `fcfs` · choices: `fcfs`, `priority`
- **type/nargs:** `str`

### `--enable-chunked-prefill, --no-enable-chunked-prefill`
- **default:** `None`
- **type/nargs:** `bool`

### `--disable-chunked-mm-input, --no-disable-chunked-mm-input`
- **default:** `False`
- **type/nargs:** `bool`

### `--scheduler-cls`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--scheduler-reserve-full-isl, --no-scheduler-reserve-full-isl`
- **default:** `True`
- **type/nargs:** `bool`

### `--disable-hybrid-kv-cache-manager, --no-disable-hybrid-kv-cache-manager`
- **default:** `None`
- **type/nargs:** `bool`

### `--async-scheduling, --no-async-scheduling`
- **default:** `None`
- **type/nargs:** `bool`

### `--stream-interval`
- **default:** `1`
- **type/nargs:** `int`

### `--cudagraph-capture-sizes`
- **default:** `None`
- **type/nargs:** `int` · nargs=`+`

### `--max-cudagraph-capture-size`
- **default:** `None`
- **type/nargs:** `int`

### `--ir-op-priority`
- **default:** `IrOpPriorityConfig(rms_norm=[], fused_add_rms_norm=[])`
- **type/nargs:** `parse_dataclass`
- **help:** API docs: https://docs.vllm.ai/en/v0.22.1/api/vllm/config/#vllm.config.IrOpPriorityConfig  Should either be a valid JSON string or JSON keys passed individually.

### `--enable-flashinfer-autotune, --no-enable-flashinfer-autotune`
- **default:** `None`
- **type/nargs:** `bool`

### `--moe-backend`
- **default:** `auto` · choices: `aiter`, `auto`, `cutlass`, `deep_gemm`, `deep_gemm_mega_moe`, `emulation`, `flashinfer_b12x`, `flashinfer_cutedsl`, `flashinfer_cutlass`, `flashinfer_trtllm`, `humming`, `marlin`, `triton`, `triton_unfused`
- **type/nargs:** `<lambda>`

### `--linear-backend`
- **default:** `auto` · choices: `aiter`, `auto`, `conch`, `cutlass`, `deep_gemm`, `emulation`, `exllama`, `fbgemm`, `flashinfer_cudnn`, `flashinfer_cutlass`, `flashinfer_trtllm`, `machete`, `marlin`, `torch`, `triton`
- **type/nargs:** `<lambda>`

### `--speculative-config, -sc`
- **default:** `None`
- **type/nargs:** `_optional_type`
- **help:** API docs: https://docs.vllm.ai/en/v0.22.1/api/vllm/config/#vllm.config.SpeculativeConfig  Should either be a valid JSON string or JSON keys passed individually.

### `--spec-method`
- **default:** `None` · choices: `custom_class`, `deepseek_mtp`, `dflash`, `draft_model`, `eagle`, `eagle3`, `ernie_mtp`, `exaone4_5_mtp`, `exaone_moe_mtp`, `extract_hidden_states`, `gemma4_mtp`, `glm4_moe_lite_mtp`, `glm4_moe_mtp`, `glm_ocr_mtp`, `hy_v3_mtp`, `longcat_flash_mtp`, `medusa`, `mimo_mtp`, `mimo_v2_mtp`, `mlp_speculator`, `mtp`, `nemotron_h_mtp`, `ngram`, `ngram_gpu`, `pangu_ultra_moe_mtp`, `qwen3_5_mtp`, `qwen3_next_mtp`, `step3p5_mtp`, `suffix`, `None`
- **type/nargs:** `_optional_type`

### `--spec-model`
- **default:** `None`
- **type/nargs:** `_optional_type`

### `--spec-tokens`
- **default:** `None`
- **type/nargs:** `int`

### `--kv-transfer-config`
- **default:** `None`
- **type/nargs:** `_optional_type`
- **help:** API docs: https://docs.vllm.ai/en/v0.22.1/api/vllm/config/#vllm.config.KVTransferConfig  Should either be a valid JSON string or JSON keys passed individually.

### `--kv-events-config`
- **default:** `None`
- **type/nargs:** `_optional_type`
- **help:** API docs: https://docs.vllm.ai/en/v0.22.1/api/vllm/config/#vllm.config.KVEventsConfig  Should either be a valid JSON string or JSON keys passed individually.

### `--ec-transfer-config`
- **default:** `None`
- **type/nargs:** `_optional_type`
- **help:** API docs: https://docs.vllm.ai/en/v0.22.1/api/vllm/config/#vllm.config.ECTransferConfig  Should either be a valid JSON string or JSON keys passed individually.

### `--compilation-config, -cc`
- **default:** `{'mode': None, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': [], 'ir_enable_torch_wrap': None, 'splitting_ops': None, 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': None, 'compile_ranges_endpoints': None, 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': None, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': None, 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': None, 'pass_config': {}, 'max_cudagraph_capture_size': None, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': None, 'static_all_moe_layers': []}`
- **type/nargs:** `parse_dataclass`
- **help:** API docs: https://docs.vllm.ai/en/v0.22.1/api/vllm/config/#vllm.config.CompilationConfig  Should either be a valid JSON string or JSON keys passed individually.

### `--attention-config, -ac`
- **default:** `AttentionConfig(backend=None, flash_attn_version=None, use_prefill_decode_attention=False, flash_attn_max_num_splits_for_cuda_graph=32, tq_max_kv_splits_for_cuda_graph=32, use_trtllm_attention=None, disable_flashinfer_q_quantization=False, mla_prefill_backend=None, use_prefill_query_quantization=False, use_fp4_indexer_cache=False, use_non_causal=False, flex_attn_block_m=None, flex_attn_block_n=None, flex_attn_q_block_size=None, flex_attn_kv_block_size=None)`
- **type/nargs:** `parse_dataclass`
- **help:** API docs: https://docs.vllm.ai/en/v0.22.1/api/vllm/config/#vllm.config.AttentionConfig  Should either be a valid JSON string or JSON keys passed individually.

### `--reasoning-config`
- **default:** `None`
- **type/nargs:** `_optional_type`
- **help:** API docs: https://docs.vllm.ai/en/v0.22.1/api/vllm/config/#vllm.config.ReasoningConfig  Should either be a valid JSON string or JSON keys passed individually.

### `--kernel-config`
- **default:** `KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=[], fused_add_rms_norm=[]), enable_flashinfer_autotune=None, moe_backend='auto', linear_backend='auto')`
- **type/nargs:** `parse_dataclass`
- **help:** API docs: https://docs.vllm.ai/en/v0.22.1/api/vllm/config/#vllm.config.KernelConfig  Should either be a valid JSON string or JSON keys passed individually.

### `--additional-config`
- **default:** `{}`
- **type/nargs:** `union_dict_and_str`

### `--structured-outputs-config`
- **default:** `StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False)`
- **type/nargs:** `parse_dataclass`
- **help:** API docs: https://docs.vllm.ai/en/v0.22.1/api/vllm/config/#vllm.config.StructuredOutputsConfig  Should either be a valid JSON string or JSON keys passed individually.

### `--profiler-config`
- **default:** `ProfilerConfig(profiler=None, torch_profiler_dir='', torch_profiler_with_stack=True, torch_profiler_with_flops=False, torch_profiler_use_gzip=True, torch_profiler_dump_cuda_time_total=True, torch_profiler_record_shapes=False, torch_profiler_with_memory=False, ignore_frontend=False, delay_iterations=0, max_iterations=0, warmup_iterations=0, active_iterations=5, wait_iterations=0)`
- **type/nargs:** `parse_dataclass`
- **help:** API docs: https://docs.vllm.ai/en/v0.22.1/api/vllm/config/#vllm.config.ProfilerConfig  Should either be a valid JSON string or JSON keys passed individually.

### `--optimization-level`
- **default:** `2`
- **type/nargs:** `str`

### `--performance-mode`
- **default:** `balanced` · choices: `balanced`, `interactivity`, `throughput`
- **type/nargs:** `str`

### `--weight-transfer-config`
- **default:** `None`
- **type/nargs:** `_optional_type`
- **help:** API docs: https://docs.vllm.ai/en/v0.22.1/api/vllm/config/#vllm.config.WeightTransferConfig  Should either be a valid JSON string or JSON keys passed individually.

### `--disable-log-stats`
- **default:** `False`
- **type/nargs:** `bool`
- **help:** Disable logging statistics.

### `--aggregate-engine-logging`
- **default:** `False`
- **type/nargs:** `bool`
- **help:** Log aggregate rather than per-engine statistics when using data parallelism.

### `--fail-on-environ-validation, --no-fail-on-environ-validation`
- **default:** `False`
- **type/nargs:** `bool`
- **help:** If set, the engine will raise an error if environment validation fails.

### `--shutdown-timeout`
- **default:** `0`
- **type/nargs:** `int`
- **help:** Shutdown timeout in seconds. 0 = abort, >0 = wait.

### `--gdn-prefill-backend`
- **default:** `None` · choices: `flashinfer`, `triton`, `cutedsl`
- **type/nargs:** `str`
- **help:** Select GDN prefill backend.

### `--enable-log-requests, --no-enable-log-requests`
- **default:** `False`
- **type/nargs:** `bool`
- **help:** Enable logging request information, dependent on log level: - INFO: Request ID, parameters and LoRA request. - DEBUG: Prompt inputs (e.g: text, token IDs). You can set the minimum log level via `VLLM_

---

## Cách tái tạo (nếu vLLM image được cập nhật, hoặc cần verify lại)

```bash
docker run --rm --entrypoint python3 duc0811/qwen35-2b-race:v1 -c "
import vllm.platforms as p
from vllm.platforms.interface import Platform

class FakePlatform(Platform):
    device_type = 'cuda'
    dispatch_key = 'CUDA'

p.current_platform = FakePlatform()   # bypass GPU auto-detect (dev machine không có CUDA thật)

from vllm.entrypoints.openai.cli_args import make_arg_parser
import argparse, json
parser = make_arg_parser(argparse.ArgumentParser())

rows = []
for a in parser._actions:
    if not a.option_strings: continue
    rows.append({'flags': ', '.join(a.option_strings), 'dest': a.dest,
                 'choices': list(a.choices) if a.choices else None,
                 'default': a.default, 'nargs': a.nargs, 'help': a.help})
print(json.dumps(rows))
"
```

Lưu ý: đây chỉ là **default của argparse**, một số giá trị (`None`) được engine tự
tính lại lúc runtime dựa trên GPU/model thật (ví dụ `max_num_seqs`,
`max_model_len`, `enable_prefix_caching`) — `None` ở đây nghĩa là "chưa cố định
ở tầng CLI", không phải "tắt tính năng". Cần đọc thêm log khởi động thật (trên
máy có GPU) để biết giá trị được engine tính ra cuối cùng là gì.
