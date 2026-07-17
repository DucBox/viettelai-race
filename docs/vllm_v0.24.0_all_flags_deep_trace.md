# Bách Khoa Toàn Thư Cờ vLLM (Phiên bản Release v0.24.0)

Dưới đây là kết quả rà soát tự động qua hàng ngàn file mã nguồn của vLLM **bản chính thức v0.24.0** để tìm ra đích xác file nào và hàm nào tiêu thụ từng biến cờ CLI.

| Cờ CLI | Default Value | Vị trí Code Tiêu Thụ (Tối đa 3 file) |
|---|---|---|
| `--model` | ModelConfig.model | `setup.py` (L1249)<br>`tools/pre_commit/generate_attention_backend_docs.py` (L1445,1452,1453,...)<br>`tools/profiler/print_layerwise_table.py` (L52,54,72)<br>... |
| `--enable-return-routed-experts` | ModelConfig.enable_return_routed_experts | `vllm/sampling_params.py` (L329)<br>`vllm/v1/outputs.py` (L280)<br>`vllm/v1/core/sched/scheduler.py` (L305,306,309,...)<br>... |
| `--model-weights` | ModelConfig.model_weights | `vllm/envs.py` (L1022)<br>`vllm/config/vllm.py` (L1923,1924)<br>`vllm/config/model.py` (L108,828,829,...)<br>... |
| `--served-model-name` | ModelConfig.served_model_name | `vllm/pooling_params.py` (L169,178)<br>`vllm/v1/metrics/loggers.py` (L432)<br>`vllm/config/vllm.py` (L1986)<br>... |
| `--tokenizer` | ModelConfig.tokenizer | `vllm/envs.py` (L660)<br>`vllm/logits_process.py` (L22,29,34)<br>`vllm/sampling_params.py` (L191,192,195,...)<br>... |
| `--hf-config-path` | ModelConfig.hf_config_path | `vllm/config/model.py` (L161,372,493,...) |
| `--runner` | ModelConfig.runner | `vllm/envs.py` (L1829,1854)<br>`vllm/env_override.py` (L725)<br>`vllm/v1/kv_cache_interface.py` (L211,888)<br>... |
| `--convert` | ModelConfig.convert | `csrc/libtorch_stable/quantization/machete/generate.py` (L221,226,274,...)<br>`cmake/hipify.py` (L47)<br>`vllm/env_override.py` (L607)<br>... |
| `--skip-tokenizer-init` | ModelConfig.skip_tokenizer_init | `vllm/sampling_params.py` (L885)<br>`vllm/v1/structured_output/__init__.py` (L71)<br>`vllm/renderers/params.py` (L390)<br>... |
| `--enable-prompt-embeds` | ModelConfig.enable_prompt_embeds | `vllm/v1/worker/gpu_model_runner.py` (L453,1733,1782,...)<br>`vllm/renderers/embed_utils.py` (L20)<br>`vllm/renderers/base.py` (L754)<br>... |
| `--tokenizer-mode` | ModelConfig.tokenizer_mode | `vllm/sampling_params.py` (L944,963)<br>`vllm/tokenizers/registry.py` (L51,52,54,...)<br>`vllm/config/speculative.py` (L717)<br>... |
| `--trust-remote-code` | ModelConfig.trust_remote_code | `vllm/renderers/hf.py` (L225,227,245,...)<br>`vllm/tokenizers/mistral.py` (L199)<br>`vllm/tokenizers/registry.py` (L165,177,193,...)<br>... |
| `--allowed-local-media-path` | ModelConfig.allowed_local_media_path | `vllm/config/speculative.py` (L719)<br>`vllm/config/model.py` (L164,373)<br>`vllm/multimodal/utils.py` (L304,325,346)<br>... |
| `--allowed-media-domains` | ModelConfig.allowed_media_domains | `vllm/config/speculative.py` (L720)<br>`vllm/config/model.py` (L168,374)<br>`vllm/multimodal/media/connector.py` (L87,97,124,...)<br>... |
| `--download-dir` | LoadConfig.download_dir | `vllm/tokenizers/mistral.py` (L201,208)<br>`vllm/tokenizers/registry.py` (L95,109,112,...)<br>`vllm/tokenizers/kimi_audio.py` (L62,85,94,...)<br>... |
| `--safetensors-load-strategy` | LoadConfig.safetensors_load_strategy | `vllm/config/load.py` (L61)<br>`vllm/model_executor/model_loader/default_loader.py` (L116,119,123,...)<br>`vllm/model_executor/model_loader/weight_utils.py` (L823,836,857,...) |
| `--safetensors-prefetch-num-threads` | LoadConfig.safetensors_prefetch_num_threads | `vllm/config/load.py` (L83)<br>`vllm/model_executor/model_loader/default_loader.py` (L292,293)<br>`vllm/model_executor/model_loader/weight_utils.py` (L826,901) |
| `--safetensors-prefetch-block-size` | LoadConfig.safetensors_prefetch_block_size | `vllm/config/load.py` (L88)<br>`vllm/model_executor/model_loader/default_loader.py` (L295,296)<br>`vllm/model_executor/model_loader/weight_utils.py` (L827,902) |
| `--load-format` | LoadConfig.load_format | `vllm/v1/worker/gpu_model_runner.py` (L5174,5422)<br>`vllm/v1/worker/gpu/model_runner.py` (L276)<br>`vllm/config/vllm.py` (L1926,1929,1931,...)<br>... |
| `--config-format` | ModelConfig.config_format | `vllm/config/speculative.py` (L732)<br>`vllm/config/model.py` (L263,383,539,...)<br>`vllm/entrypoints/serve/dev/server_info/api_router.py` (L46,52)<br>... |
| `--dtype` | ModelConfig.dtype | `tools/pre_commit/generate_attention_backend_docs.py` (L1491)<br>`tools/gumbel_precision/prove_exponential_race_precision.py` (L40,49,58,...)<br>`vllm/envs.py` (L1384,1388,1392)<br>... |
| `--kv-cache-dtype` | CacheConfig.cache_dtype | `tools/pre_commit/generate_attention_backend_docs.py` (L603)<br>`vllm/_custom_ops.py` (L126,147,173,...)<br>`vllm/_aiter_ops.py` (L2783,2813)<br>... |
| `--seed` | ModelConfig.seed | `tools/gumbel_precision/prove_exponential_race_precision.py` (L27,28,36,...)<br>`vllm/sampling_params.py` (L250,251,365,...)<br>`vllm/v1/kv_offload/tiering/fs/manager.py` (L78)<br>... |
| `--max-model-len` | ModelConfig.max_model_len | `vllm/_xpu_ops.py` (L262,271,282,...)<br>`vllm/v1/utils.py` (L713)<br>`vllm/v1/kv_cache_interface.py` (L237,241,243,...)<br>... |
| `--cudagraph-capture-sizes` | CompilationConfig.cudagraph_capture_sizes | `vllm/v1/cudagraph_dispatcher.py` (L75,108,190,...)<br>`vllm/v1/metrics/loggers.py` (L122)<br>`vllm/v1/attention/backends/triton_attn.py` (L139)<br>... |
| `--max-cudagraph-capture-size` | None | `vllm/v1/cudagraph_dispatcher.py` (L74,272)<br>`vllm/v1/attention/backends/flash_attn.py` (L370)<br>`vllm/v1/attention/backends/mamba_attn.py` (L105,108)<br>... |
| `--ir-op-priority` | None | `vllm/v1/worker/worker_base.py` (L93)<br>`vllm/kernels/oink_ops.py` (L8)<br>`vllm/config/vllm.py` (L114,183)<br>... |
| `--distributed-executor-backend` | ParallelConfig.distributed_executor_backend | `vllm/envs.py` (L854)<br>`vllm/v1/structured_output/__init__.py` (L54)<br>`vllm/v1/executor/abstract.py` (L51,52,53,...)<br>... |
| `--pipeline-parallel-size` | ParallelConfig.pipeline_parallel_size | `vllm/v1/utils.py` (L706)<br>`vllm/v1/kv_offload/file_mapper.py` (L103)<br>`vllm/v1/metrics/perf.py` (L345,1483)<br>... |
| `--master-addr` | ParallelConfig.master_addr | `vllm/v1/executor/multiproc_executor.py` (L142,146)<br>`vllm/distributed/parallel_state.py` (L1574)<br>`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py` (L1963)<br>... |
| `--master-port` | ParallelConfig.master_port | `vllm/distributed/parallel_state.py` (L1575)<br>`vllm/distributed/weight_transfer/nccl_engine.py` (L34,160,359,...)<br>`vllm/config/parallel.py` (L270,761)<br>... |
| `--nnodes` | ParallelConfig.nnodes | `vllm/v1/worker/gpu_worker.py` (L286,288)<br>`vllm/distributed/parallel_state.py` (L1560,1573,1666,...)<br>`vllm/config/parallel.py` (L248,276,676,...) |
| `--node-rank` | ParallelConfig.node_rank | `vllm/v1/executor/multiproc_executor.py` (L141,144)<br>`vllm/config/parallel.py` (L273,672,762)<br>`vllm/entrypoints/openai/dp_supervisor.py` (L42,44) |
| `--distributed-timeout-seconds` | ParallelConfig.distributed_timeout_seconds | `vllm/v1/worker/gpu_worker.py` (L1213,1214)<br>`vllm/config/parallel.py` (L314) |
| `--cpu-distributed-timeout-seconds` | ParallelConfig.cpu_distributed_timeout_seconds | `vllm/distributed/utils.py` (L514)<br>`vllm/config/parallel.py` (L320) |
| `--numa-bind` | ParallelConfig.numa_bind | `vllm/v1/executor/multiproc_executor.py` (L867)<br>`vllm/v1/engine/core.py` (L1174)<br>`vllm/config/parallel.py` (L279,291,468,...)<br>... |
| `--numa-bind-nodes` | ParallelConfig.numa_bind_nodes | `vllm/config/parallel.py` (L286,403,409,...)<br>`vllm/utils/numa_utils.py` (L250,259,263,...) |
| `--numa-bind-cpus` | ParallelConfig.numa_bind_cpus | `vllm/config/parallel.py` (L295,414,420,...)<br>`vllm/utils/numa_utils.py` (L330,339,426) |
| `--device-ids` | None | `vllm/platforms/interface.py` (L298,299)<br>`vllm/platforms/cuda.py` (L278,950,951,...)<br>`vllm/entrypoints/openai/dp_supervisor.py` (L128,142,143,...) |
| `--tensor-parallel-size` | ParallelConfig.tensor_parallel_size | `tools/profiler/visualize_layerwise_profile.py` (L517)<br>`vllm/v1/utils.py` (L704)<br>`vllm/v1/kv_offload/file_mapper.py` (L102)<br>... |
| `--prefill-context-parallel-size` | ParallelConfig.prefill_context_parallel_size | `vllm/v1/kv_cache_interface.py` (L239)<br>`vllm/v1/kv_offload/file_mapper.py` (L104)<br>`vllm/v1/kv_offload/base.py` (L477)<br>... |
| `--decode-context-parallel-size` | ParallelConfig.decode_context_parallel_size | `vllm/v1/kv_cache_interface.py` (L238,529)<br>`vllm/v1/kv_offload/file_mapper.py` (L105)<br>`vllm/v1/kv_offload/base.py` (L476)<br>... |
| `--dcp-comm-backend` | ParallelConfig.dcp_comm_backend | `vllm/v1/attention/backends/flash_attn.py` (L693)<br>`vllm/v1/attention/backends/flashinfer.py` (L635,1422)<br>`vllm/config/vllm.py` (L1975)<br>... |
| `--dcp-kv-cache-interleave-size` | ParallelConfig.dcp_kv_cache_interleave_size | `vllm/v1/attention/backends/flashinfer.py` (L625,626,632,...)<br>`vllm/config/vllm.py` (L2119,2121,2124)<br>`vllm/config/parallel.py` (L344,347) |
| `--cp-kv-cache-interleave-size` | ParallelConfig.cp_kv_cache_interleave_size | `vllm/v1/attention/backend.py` (L754)<br>`vllm/v1/attention/backends/flash_attn.py` (L363,364,505,...)<br>`vllm/v1/attention/backends/utils.py` (L863,885,887,...)<br>... |
| `--data-parallel-size` | ParallelConfig.data_parallel_size | `vllm/forward_context.py` (L86,272)<br>`vllm/v1/utils.py` (L705)<br>`vllm/v1/metrics/perf.py` (L343)<br>... |
| `--data-parallel-rank` | None | `vllm/forward_context.py` (L88)<br>`vllm/v1/utils.py` (L374)<br>`vllm/v1/spec_decode/extract_hidden_states.py` (L44)<br>... |
| `--data-parallel-start-rank` | None | `vllm/entrypoints/cli/serve.py` (L86)<br>`vllm/entrypoints/openai/dp_supervisor.py` (L38,121) |
| `--data-parallel-size-local` | None | `vllm/v1/engine/coordinator.py` (L90)<br>`vllm/v1/engine/core_client.py` (L598,606)<br>`vllm/v1/engine/core.py` (L1198,2289)<br>... |
| `--data-parallel-address` | None | Nội bộ trong EngineArgs |
| `--data-parallel-rpc-port` | None | `vllm/v1/engine/utils.py` (L1179)<br>`vllm/config/parallel.py` (L140,758)<br>`vllm/entrypoints/cli/serve.py` (L224) |
| `--data-parallel-hybrid-lb` | False | `vllm/v1/utils.py` (L376)<br>`vllm/v1/engine/core_client.py` (L1295)<br>`vllm/v1/engine/utils.py` (L1244)<br>... |
| `--data-parallel-external-lb` | False | `vllm/v1/utils.py` (L375)<br>`vllm/v1/engine/core_client.py` (L127,1296)<br>`vllm/v1/engine/core.py` (L951)<br>... |
| `--data-parallel-multi-port-external-lb` | False | `vllm/entrypoints/cli/serve.py` (L79,84,89)<br>`vllm/entrypoints/openai/cli_args.py` (L401)<br>`vllm/entrypoints/openai/dp_supervisor.py` (L125) |
| `--data-parallel-backend` | ParallelConfig.data_parallel_backend | `vllm/v1/executor/vllm_net_devices.py` (L209)<br>`vllm/v1/worker/gpu_worker.py` (L258)<br>`vllm/v1/worker/xpu_worker.py` (L49)<br>... |
| `--enable-expert-parallel` | ParallelConfig.enable_expert_parallel | `vllm/v1/utils.py` (L707)<br>`vllm/v1/metrics/perf.py` (L346)<br>`vllm/v1/executor/multiproc_executor.py` (L622,629)<br>... |
| `--enable-ep-weight-filter` | ParallelConfig.enable_ep_weight_filter | `vllm/config/parallel.py` (L164)<br>`vllm/model_executor/model_loader/default_loader.py` (L366) |
| `--moe-backend` | KernelConfig.moe_backend | `vllm/v1/spec_decode/llm_base_proposer.py` (L1196,1201)<br>`vllm/v1/worker/gpu_worker.py` (L1206)<br>`vllm/distributed/eplb/eplb_utils.py` (L66,73,77)<br>... |
| `--linear-backend` | KernelConfig.linear_backend | `vllm/config/kernel.py` (L196,224)<br>`vllm/model_executor/kernels/linear/__init__.py` (L186,191,195,...)<br>`vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py` (L123,124) |
| `--all2all-backend` | ParallelConfig.all2all_backend | `vllm/v1/utils.py` (L709)<br>`vllm/v1/engine/utils.py` (L593,595,596)<br>`vllm/distributed/device_communicators/xpu_communicator.py` (L27,38)<br>... |
| `--enable-elastic-ep` | ParallelConfig.enable_elastic_ep | `vllm/v1/worker/gpu_model_runner.py` (L5192)<br>`vllm/v1/worker/gpu/eplb_utils.py` (L83)<br>`vllm/v1/engine/coordinator.py` (L92)<br>... |
| `--enable-dbo` | ParallelConfig.enable_dbo | `vllm/v1/worker/gpu_worker.py` (L351)<br>`vllm/v1/worker/xpu_worker.py` (L126)<br>`vllm/platforms/cpu.py` (L156,158)<br>... |
| `--ubatch-size` | ParallelConfig.ubatch_size | `vllm/config/parallel.py` (L210,524,528) |
| `--dbo-decode-token-threshold` | ParallelConfig.dbo_decode_token_threshold | `vllm/v1/worker/ubatch_utils.py` (L44)<br>`vllm/config/parallel.py` (L213) |
| `--dbo-prefill-token-threshold` | ParallelConfig.dbo_prefill_token_threshold | `vllm/v1/worker/ubatch_utils.py` (L46)<br>`vllm/config/parallel.py` (L218) |
| `--disable-nccl-for-dp-synchronization` | ParallelConfig.disable_nccl_for_dp_synchronization | `vllm/v1/worker/dp_utils.py` (L27)<br>`vllm/config/vllm.py` (L1011,1020,1022)<br>`vllm/config/parallel.py` (L224,397) |
| `--eplb-config` | None | `vllm/v1/utils.py` (L725)<br>`vllm/v1/worker/gpu_model_runner.py` (L3336)<br>`vllm/v1/worker/gpu/eplb_utils.py` (L135)<br>... |
| `--enable-eplb` | ParallelConfig.enable_eplb | `vllm/v1/utils.py` (L724)<br>`vllm/v1/spec_decode/medusa.py` (L70)<br>`vllm/v1/worker/gpu_model_runner.py` (L3328,5166,5190,...)<br>... |
| `--expert-placement-strategy` | ParallelConfig.expert_placement_strategy | `vllm/config/parallel.py` (L175)<br>`vllm/model_executor/layers/fused_moe/routed_experts.py` (L228)<br>`vllm/model_executor/layers/fused_moe/layer.py` (L262)<br>... |
| `--max-parallel-loading-workers` | ParallelConfig.max_parallel_loading_workers | `vllm/config/speculative.py` (L997)<br>`vllm/config/parallel.py` (L197,764,918,...) |
| `--block-size` | None | `vllm/_custom_ops.py` (L55,58,64,...)<br>`vllm/_xpu_ops.py` (L484,605,606,...)<br>`vllm/_aiter_ops.py` (L2161,2174)<br>... |
| `--enable-prefix-caching` | None | `vllm/v1/utils.py` (L700)<br>`vllm/v1/core/kv_cache_utils.py` (L647)<br>`vllm/v1/core/sched/scheduler.py` (L258)<br>... |
| `--prefix-caching-hash-algo` | CacheConfig.prefix_caching_hash_algo | `vllm/v1/engine/core.py` (L213)<br>`vllm/config/cache.py` (L94,205) |
| `--disable-sliding-window` | ModelConfig.disable_sliding_window | `vllm/v1/attention/backends/flashinfer.py` (L1026)<br>`vllm/config/model.py` (L234,653,716,...) |
| `--disable-cascade-attn` | ModelConfig.disable_cascade_attn | `vllm/v1/worker/gpu_model_runner.py` (L487)<br>`vllm/platforms/cpu.py` (L120)<br>`vllm/config/vllm.py` (L1028,1034,1399,...)<br>... |
| `--offload-backend` | OffloadConfig.offload_backend | `vllm/config/offload.py` (L83,100,118,...)<br>`vllm/model_executor/offloader/base.py` (L129,136) |
| `--cpu-offload-gb` | UVAOffloadConfig.cpu_offload_gb | `vllm/platforms/cuda.py` (L330)<br>`vllm/config/offload.py` (L23,38,58,...)<br>`vllm/model_executor/offloader/base.py` (L131,143,158)<br>... |
| `--cpu-offload-params` | None | `vllm/config/offload.py` (L34)<br>`vllm/model_executor/offloader/uva.py` (L33,40,44,...)<br>`vllm/model_executor/offloader/base.py` (L159) |
| `--offload-group-size` | PrefetchOffloadConfig.offload_group_size | `vllm/config/offload.py` (L54,64,86,...)<br>`vllm/model_executor/offloader/base.py` (L130,141,150)<br>`vllm/entrypoints/llm.py` (L126,197,324) |
| `--offload-num-in-group` | PrefetchOffloadConfig.offload_num_in_group | `vllm/config/offload.py` (L55,62,101,...)<br>`vllm/model_executor/offloader/base.py` (L151)<br>`vllm/entrypoints/llm.py` (L127,129,198,...) |
| `--offload-prefetch-step` | PrefetchOffloadConfig.offload_prefetch_step | `vllm/config/offload.py` (L66,107,109,...)<br>`vllm/model_executor/offloader/base.py` (L152)<br>`vllm/entrypoints/llm.py` (L131,199,326) |
| `--offload-params` | None | `vllm/config/offload.py` (L70)<br>`vllm/model_executor/offloader/prefetch.py` (L146,152,181,...)<br>`vllm/model_executor/offloader/base.py` (L153)<br>... |
| `--gpu-memory-utilization` | CacheConfig.gpu_memory_utilization | `vllm/v1/utils.py` (L693)<br>`vllm/v1/core/kv_cache_utils.py` (L742,765,1941)<br>`vllm/v1/worker/gpu_worker.py` (L410,422,499,...)<br>... |
| `--kv-cache-memory-bytes` | CacheConfig.kv_cache_memory_bytes | `vllm/v1/utils.py` (L694)<br>`vllm/v1/worker/gpu_worker.py` (L412,419,420,...)<br>`vllm/v1/worker/cpu_worker.py` (L68,188)<br>... |
| `--max-num-batched-tokens` | None | `vllm/v1/utils.py` (L715)<br>`vllm/v1/kv_cache_interface.py` (L445,455,461,...)<br>`vllm/v1/attention/backends/turboquant_attn.py` (L231)<br>... |
| `--max-num-partial-prefills` | SchedulerConfig.max_num_partial_prefills | `vllm/config/scheduler.py` (L70,77,257,...) |
| `--max-long-partial-prefills` | SchedulerConfig.max_long_partial_prefills | `vllm/config/scheduler.py` (L74,263,266,...) |
| `--long-prefill-token-threshold` | SchedulerConfig.long_prefill_token_threshold | `vllm/v1/core/sched/scheduler.py` (L468,469,797)<br>`vllm/config/vllm.py` (L2148,2149)<br>`vllm/config/scheduler.py` (L76,80,242,...) |
| `--max-num-seqs` | None | `vllm/v1/cudagraph_dispatcher.py` (L139,144,148,...)<br>`vllm/v1/utils.py` (L714)<br>`vllm/v1/attention/backends/flash_attn.py` (L300,379,395)<br>... |
| `--max-logprobs` | ModelConfig.max_logprobs | `vllm/sampling_params.py` (L734,735,736,...)<br>`vllm/config/speculative.py` (L730)<br>`vllm/config/model.py` (L216) |
| `--logprobs-mode` | ModelConfig.logprobs_mode | `vllm/_xpu_ops.py` (L298,303,314)<br>`vllm/v1/sample/rejection_sampler.py` (L69,70,71)<br>`vllm/v1/sample/sampler.py` (L26,28,63,...)<br>... |
| `--use-fp64-gumbel` | ModelConfig.use_fp64_gumbel | `vllm/v1/spec_decode/llm_base_proposer.py` (L123,427,1714,...)<br>`vllm/v1/sample/rejection_sampler.py` (L68,180,411,...)<br>`vllm/v1/sample/sampler.py` (L64,67,70)<br>... |
| `--disable-log-stats` | False | `vllm/v1/engine/async_llm.py` (L211,223,250)<br>`vllm/v1/engine/llm_engine.py` (L149,154,182)<br>`vllm/entrypoints/grpc_server.py` (L81,127)<br>... |
| `--aggregate-engine-logging` | False | `vllm/v1/metrics/loggers.py` (L1293,1310)<br>`vllm/v1/engine/async_llm.py` (L83,164,210,...)<br>`vllm/v1/engine/llm_engine.py` (L56,119)<br>... |
| `--revision` | ModelConfig.revision | `vllm/tokenizers/mistral.py` (L200,209)<br>`vllm/tokenizers/registry.py` (L94,113,139,...)<br>`vllm/tokenizers/kimi_audio.py` (L61,84,93,...)<br>... |
| `--code-revision` | ModelConfig.code_revision | `vllm/config/speculative.py` (L123,724)<br>`vllm/config/model.py` (L174,538)<br>`vllm/model_executor/models/registry.py` (L1105,1119)<br>... |
| `--hf-token` | ModelConfig.hf_token | `vllm/config/model.py` (L270,384,542,...)<br>`vllm/entrypoints/llm.py` (L145,204,331)<br>`vllm/transformers_utils/config.py` (L603,618,627,...) |
| `--hf-overrides` | None | `vllm/config/speculative.py` (L731)<br>`vllm/config/model.py` (L274,385,496,...)<br>`vllm/config/utils.py` (L251)<br>... |
| `--tokenizer-revision` | ModelConfig.tokenizer_revision | `vllm/tokenizers/registry.py` (L164,255)<br>`vllm/config/speculative.py` (L725)<br>`vllm/config/vllm.py` (L1965)<br>... |
| `--quantization` | ModelConfig.quantization | `setup.py` (L1165,1273)<br>`csrc/libtorch_stable/quantization/machete/generate.py` (L499)<br>`cmake/hipify.py` (L69)<br>... |
| `--quantization-config` | None | `vllm/config/model_arch.py` (L50)<br>`vllm/config/speculative.py` (L500,503,504,...)<br>`vllm/config/vllm.py` (L1978)<br>... |
| `--allow-deprecated-quantization` | ModelConfig.allow_deprecated_quantization | `vllm/config/model.py` (L207,1059) |
| `--enforce-eager` | ModelConfig.enforce_eager | `vllm/v1/utils.py` (L701)<br>`vllm/v1/spec_decode/extract_hidden_states.py` (L241)<br>`vllm/v1/spec_decode/llm_base_proposer.py` (L401)<br>... |
| `--disable-custom-all-reduce` | ParallelConfig.disable_custom_all_reduce | `vllm/v1/utils.py` (L702)<br>`vllm/v1/worker/gpu_worker.py` (L1208)<br>`vllm/v1/worker/cpu_worker.py` (L93)<br>... |
| `--language-model-only` | MultiModalConfig.language_model_only | `vllm/config/multimodal.py` (L77,315)<br>`vllm/config/model.py` (L335,410,461,...)<br>`vllm/model_executor/models/qwen3_next.py` (L303)<br>... |
| `--limit-mm-per-prompt` | None | `vllm/v1/core/encoder_cache_manager.py` (L291)<br>`vllm/config/model.py` (L336,391,462,...) |
| `--enable-mm-embeds` | MultiModalConfig.enable_mm_embeds | `vllm/v1/worker/gpu_model_runner.py` (L6260)<br>`vllm/config/multimodal.py` (L97)<br>`vllm/config/model.py` (L337,463,678)<br>... |
| `--interleave-mm-strings` | MultiModalConfig.interleave_mm_strings | `vllm/config/multimodal.py` (L180)<br>`vllm/config/model.py` (L350,398,476,...)<br>`vllm/entrypoints/chat_utils.py` (L1888,1927) |
| `--media-io-kwargs` | None | `vllm/renderers/params.py` (L84,110,112)<br>`vllm/renderers/mistral.py` (L72,100)<br>`vllm/renderers/deepseek_v4.py` (L47,74)<br>... |
| `--mm-processor-kwargs` | MultiModalConfig.mm_processor_kwargs | `vllm/renderers/params.py` (L87,114,116)<br>`vllm/renderers/mistral.py` (L73,101)<br>`vllm/renderers/deepseek_v4.py` (L48,75)<br>... |
| `--mm-processor-cache-gb` | MultiModalConfig.mm_processor_cache_gb | `vllm/renderers/base.py` (L669)<br>`vllm/config/multimodal.py` (L123,129)<br>`vllm/config/model.py` (L340,394,466,...)<br>... |
| `--mm-processor-cache-type` | MultiModalConfig.mm_processor_cache_type | `vllm/envs.py` (L1726)<br>`vllm/v1/worker/worker_base.py` (L299,303)<br>`vllm/config/multimodal.py` (L132,137,247,...)<br>... |
| `--mm-shm-cache-max-object-size-mb` | MultiModalConfig.mm_shm_cache_max_object_size_mb | `vllm/config/multimodal.py` (L135,248,249,...)<br>`vllm/config/model.py` (L342,396,468,...)<br>`vllm/multimodal/cache.py` (L461,704) |
| `--mm-encoder-only` | MultiModalConfig.mm_encoder_only | `vllm/v1/worker/gpu_model_runner.py` (L5713,6054,6214)<br>`vllm/config/multimodal.py` (L139)<br>`vllm/config/model.py` (L343,469,684)<br>... |
| `--mm-encoder-tp-mode` | MultiModalConfig.mm_encoder_tp_mode | `vllm/v1/worker/encoder_cudagraph.py` (L151)<br>`vllm/config/multimodal.py` (L145,303)<br>`vllm/config/model.py` (L344,397,470,...)<br>... |
| `--mm-encoder-attn-backend` | MultiModalConfig.mm_encoder_attn_backend | `vllm/config/multimodal.py` (L158,226,241,...)<br>`vllm/config/model.py` (L345,471,686)<br>`vllm/model_executor/models/vision.py` (L108) |
| `--mm-encoder-attn-dtype` | MultiModalConfig.mm_encoder_attn_dtype | `vllm/config/multimodal.py` (L162,169,172,...)<br>`vllm/config/model.py` (L346,472,687)<br>`vllm/model_executor/layers/attention/mm_encoder_attention.py` (L393,399)<br>... |
| `--mm-encoder-fp8-scale-path` | MultiModalConfig.mm_encoder_fp8_scale_path | `vllm/config/multimodal.py` (L167,173,175,...)<br>`vllm/config/model.py` (L347,473,688)<br>`vllm/model_executor/layers/attention/mm_encoder_attention.py` (L406,446) |
| `--mm-encoder-fp8-scale-save-path` | MultiModalConfig.mm_encoder_fp8_scale_save_path | `vllm/config/multimodal.py` (L171,258,262,...)<br>`vllm/config/model.py` (L348,474,689)<br>`vllm/model_executor/layers/attention/mm_encoder_attention.py` (L66,108,428,...) |
| `--mm-encoder-fp8-scale-save-margin` | MultiModalConfig.mm_encoder_fp8_scale_save_margin | `vllm/config/multimodal.py` (L176)<br>`vllm/config/model.py` (L349,475,690)<br>`vllm/model_executor/layers/attention/mm_encoder_attention.py` (L51,433) |
| `--io-processor-plugin` | None | `vllm/config/model.py` (L313,388)<br>`vllm/plugins/io_processors/__init__.py` (L26,49)<br>`vllm/entrypoints/pooling/factories.py` (L70)<br>... |
| `--renderer-num-workers` | 1 | `vllm/renderers/base.py` (L85)<br>`vllm/renderers/hf.py` (L882)<br>`vllm/config/model.py` (L315,704)<br>... |
| `--skip-mm-profiling` | MultiModalConfig.skip_mm_profiling | `vllm/v1/worker/gpu_model_runner.py` (L6243)<br>`vllm/config/multimodal.py` (L183)<br>`vllm/config/model.py` (L351,399,477,...) |
| `--video-pruning-rate` | MultiModalConfig.video_pruning_rate | `vllm/config/multimodal.py` (L190,338)<br>`vllm/config/model.py` (L352,478,693)<br>`vllm/model_executor/models/qwen3_vl_moe.py` (L426)<br>... |
| `--mm-tensor-ipc` | MultiModalConfig.mm_tensor_ipc | `vllm/v1/engine/core_client.py` (L589,590)<br>`vllm/v1/engine/utils.py` (L1103)<br>`vllm/config/vllm.py` (L1039)<br>... |
| `--enable-lora` | False | `vllm/v1/utils.py` (L699)<br>`vllm/config/compilation.py` (L647) |
| `--max-loras` | LoRAConfig.max_loras | `vllm/_custom_ops.py` (L2205,2220)<br>`vllm/v1/cudagraph_dispatcher.py` (L124,130,295,...)<br>`vllm/v1/metrics/loggers.py` (L1027)<br>... |
| `--max-lora-rank` | LoRAConfig.max_lora_rank | `vllm/v1/worker/lora_model_runner_mixin.py` (L105)<br>`vllm/config/lora.py` (L34,94)<br>`vllm/lora/peft_helper.py` (L122,124,125)<br>... |
| `--default-mm-loras` | LoRAConfig.default_mm_loras | `vllm/config/lora.py` (L52,58)<br>`vllm/entrypoints/offline_utils.py` (L66,67,71,...)<br>`vllm/entrypoints/pooling/base/serving.py` (L310,318,322,...)<br>... |
| `--fully-sharded-loras` | LoRAConfig.fully_sharded_loras | `vllm/config/lora.py` (L38,96,119,...)<br>`vllm/lora/model_manager.py` (L654)<br>`vllm/lora/layers/base_linear.py` (L114,123)<br>... |
| `--max-cpu-loras` | LoRAConfig.max_cpu_loras | `vllm/config/lora.py` (L43,110,111,...)<br>`vllm/lora/model_manager.py` (L281,282) |
| `--lora-dtype` | LoRAConfig.lora_dtype | `vllm/config/lora.py` (L46,97,128,...)<br>`vllm/lora/worker_manager.py` (L144)<br>`vllm/lora/layers/vocal_parallel_embedding.py` (L55,65)<br>... |
| `--lora-target-modules` | LoRAConfig.target_modules | Nội bộ trong EngineArgs |
| `--enable-tower-connector-lora` | LoRAConfig.enable_tower_connector_lora | `vllm/v1/engine/input_processor.py` (L171,178)<br>`vllm/config/lora.py` (L62,98)<br>`vllm/lora/model_manager.py` (L200,204,227) |
| `--specialize-active-lora` | LoRAConfig.specialize_active_lora | `vllm/v1/cudagraph_dispatcher.py` (L65)<br>`vllm/v1/worker/gpu/lora_utils.py` (L33)<br>`vllm/config/lora.py` (L67)<br>... |
| `--enable-mixed-moe-lora-format` | LoRAConfig.enable_mixed_moe_lora_format | `vllm/config/lora.py` (L74,99)<br>`vllm/lora/request.py` (L36)<br>`vllm/lora/model_manager.py` (L120,124,941)<br>... |
| `--ray-workers-use-nsight` | ParallelConfig.ray_workers_use_nsight | `vllm/v1/executor/ray_executor.py` (L157)<br>`vllm/v1/executor/ray_executor_v2.py` (L245)<br>`vllm/config/speculative.py` (L999)<br>... |
| `--num-gpu-blocks-override` | CacheConfig.num_gpu_blocks_override | `vllm/v1/core/kv_cache_utils.py` (L942,945,946,...)<br>`vllm/v1/worker/gpu_model_runner.py` (L6325,6326,6327,...)<br>`vllm/config/cache.py` (L86,203) |
| `--model-loader-extra-config` | None | `vllm/config/load.py` (L92)<br>`vllm/model_executor/model_loader/default_loader.py` (L78,81,248)<br>`vllm/model_executor/model_loader/dummy_loader.py` (L27)<br>... |
| `--ignore-patterns` | None | `vllm/config/load.py` (L98,137,139,...)<br>`vllm/model_executor/model_loader/default_loader.py` (L201)<br>`vllm/model_executor/model_loader/weight_utils.py` (L158,180,437,...)<br>... |
| `--enable-chunked-prefill` | None | `vllm/v1/attention/backends/turboquant_attn.py` (L230)<br>`vllm/v1/core/sched/scheduler.py` (L804)<br>`vllm/v1/engine/core.py` (L142,144,260,...)<br>... |
| `--disable-chunked-mm-input` | SchedulerConfig.disable_chunked_mm_input | `vllm/v1/core/encoder_cache_manager.py` (L299)<br>`vllm/v1/core/sched/scheduler.py` (L1371)<br>`vllm/platforms/cuda.py` (L320,323,326)<br>... |
| `--scheduler-reserve-full-isl` | SchedulerConfig.scheduler_reserve_full_isl | `vllm/v1/core/sched/scheduler.py` (L283,284,883)<br>`vllm/config/scheduler.py` (L140) |
| `--prefill-schedule-interval` | SchedulerConfig.prefill_schedule_interval | `vllm/v1/engine/core.py` (L1764,1921,1922)<br>`vllm/config/scheduler.py` (L153) |
| `--watermark` | SchedulerConfig.watermark | `vllm/v1/core/kv_cache_manager.py` (L125,162,163,...)<br>`vllm/v1/core/sched/scheduler.py` (L267)<br>`vllm/config/scheduler.py` (L146,147,151)<br>... |
| `--disable-hybrid-kv-cache-manager` | SchedulerConfig.disable_hybrid_kv_cache_manager | `vllm/v1/core/kv_cache_utils.py` (L1710,2058)<br>`vllm/distributed/kv_transfer/kv_connector/factory.py` (L55)<br>`vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py` (L192)<br>... |
| `--structured-outputs-config` | None | `vllm/sampling_params.py` (L721,730,865,...)<br>`vllm/v1/structured_output/backend_guidance.py` (L90,93)<br>`vllm/v1/structured_output/backend_xgrammar.py` (L39)<br>... |
| `--reasoning-parser` | StructuredOutputsConfig.reasoning_parser | `vllm/v1/structured_output/__init__.py` (L88,89,91,...)<br>`vllm/config/reasoning.py` (L22,74,78,...)<br>`vllm/config/structured_outputs.py` (L35)<br>... |
| `--reasoning-parser-plugin` | None | `vllm/v1/structured_output/__init__.py` (L82,83,85,...)<br>`vllm/config/structured_outputs.py` (L38)<br>`vllm/entrypoints/openai/api_server.py` (L529,530,674,...) |
| `--speculative-config` | None | `vllm/sampling_params.py` (L720,728,850,...)<br>`vllm/v1/utils.py` (L659,678,681,...)<br>`vllm/v1/metrics/loggers.py` (L441)<br>... |
| `--spec-method` | None | `vllm/entrypoints/llm.py` (L165,217,341) |
| `--spec-model` | None | `vllm/entrypoints/llm.py` (L166,218,342) |
| `--spec-tokens` | None | `vllm/v1/structured_output/utils.py` (L114,118,134)<br>`vllm/v1/sample/thinking_budget_state.py` (L454,460)<br>`vllm/v1/worker/gpu_model_runner.py` (L1296,1299,1300)<br>... |
| `--diffusion-config` | None | `vllm/config/vllm.py` (L323,515,516,...)<br>`vllm/model_executor/models/config.py` (L140,145)<br>`vllm/model_executor/models/diffusion_gemma.py` (L794,795,800,...) |
| `--show-hidden-metrics-for-version` | ObservabilityConfig.show_hidden_metrics_for_version | `vllm/config/observability.py` (L21,32,34,...) |
| `--otlp-traces-endpoint` | ObservabilityConfig.otlp_traces_endpoint | `vllm/v1/engine/async_llm.py` (L114,894)<br>`vllm/v1/engine/llm_engine.py` (L66)<br>`vllm/tracing/__init__.py` (L68,74)<br>... |
| `--collect-detailed-traces` | ObservabilityConfig.collect_detailed_traces | `vllm/config/observability.py` (L39,86,87,...) |
| `--kv-cache-metrics` | ObservabilityConfig.kv_cache_metrics | `vllm/v1/metrics/loggers.py` (L428)<br>`vllm/v1/core/kv_cache_coordinator.py` (L9)<br>`vllm/v1/core/block_pool.py` (L14)<br>... |
| `--kv-cache-metrics-sample` | None | `vllm/v1/core/sched/scheduler.py` (L92)<br>`vllm/config/observability.py` (L53) |
| `--cudagraph-metrics` | ObservabilityConfig.cudagraph_metrics | `vllm/v1/metrics/loggers.py` (L119)<br>`vllm/v1/worker/gpu_model_runner.py` (L3920)<br>`vllm/config/observability.py` (L56) |
| `--enable-layerwise-nvtx-tracing` | ObservabilityConfig.enable_layerwise_nvtx_tracing | `vllm/v1/worker/gpu_model_runner.py` (L3943)<br>`vllm/config/observability.py` (L60)<br>`vllm/compilation/wrapper.py` (L85) |
| `--enable-mfu-metrics` | ObservabilityConfig.enable_mfu_metrics | `vllm/v1/metrics/loggers.py` (L142)<br>`vllm/v1/core/sched/scheduler.py` (L302)<br>`vllm/config/observability.py` (L65) |
| `--enable-logging-iteration-details` | ObservabilityConfig.enable_logging_iteration_details | `vllm/v1/engine/core.py` (L435)<br>`vllm/config/observability.py` (L73) |
| `--jit-monitor-verbose` | ObservabilityConfig.jit_monitor_verbose | `vllm/v1/worker/gpu_worker.py` (L769)<br>`vllm/config/observability.py` (L79) |
| `--enable-mm-processor-stats` | ObservabilityConfig.enable_mm_processor_stats | `vllm/v1/worker/gpu_model_runner.py` (L2931)<br>`vllm/config/observability.py` (L68)<br>`vllm/multimodal/registry.py` (L354) |
| `--scheduling-policy` | SchedulerConfig.policy | Nội bộ trong EngineArgs |
| `--scheduler-cls` | SchedulerConfig.scheduler_cls | `vllm/config/scheduler.py` (L127,181,197,...) |
| `--pooler-config` | ModelConfig.pooler_config | `vllm/pooling_params.py` (L109,110,118,...)<br>`vllm/config/vllm.py` (L1238,1989)<br>`vllm/config/model.py` (L327,389,622,...)<br>... |
| `--compilation-config` | None | `vllm/envs.py` (L1773)<br>`vllm/env_override.py` (L466)<br>`vllm/forward_context.py` (L130,215,216,...)<br>... |
| `--attention-config` | None | `tools/pre_commit/generate_attention_backend_docs.py` (L1475)<br>`vllm/_aiter_ops.py` (L104,2704,2746,...)<br>`vllm/v1/utils.py` (L657,668)<br>... |
| `--mamba-config` | None | `vllm/v1/worker/gpu_model_runner.py` (L7327)<br>`vllm/v1/worker/gpu/model_runner.py` (L456)<br>`vllm/config/vllm.py` (L315,899)<br>... |
| `--kernel-config` | None | `vllm/v1/spec_decode/llm_base_proposer.py` (L1199,1200)<br>`vllm/v1/worker/worker_base.py` (L93)<br>`vllm/v1/worker/gpu_worker.py` (L1206)<br>... |
| `--enable-flashinfer-autotune` | None | `vllm/config/vllm.py` (L214,235,256,...)<br>`vllm/config/kernel.py` (L171,238,245)<br>`vllm/model_executor/warmup/kernel_warmup.py` (L71,72,75)<br>... |
| `--worker-cls` | ParallelConfig.worker_cls | `vllm/v1/worker/worker_base.py` (L250,252,256)<br>`vllm/platforms/rocm.py` (L820,821)<br>`vllm/platforms/cpu.py` (L153,154)<br>... |
| `--worker-extension-cls` | ParallelConfig.worker_extension_cls | `vllm/v1/worker/worker_base.py` (L261,262,263,...)<br>`vllm/config/parallel.py` (L262,772) |
| `--profiler-config` | None | `vllm/v1/worker/gpu_worker.py` (L151,155,158,...)<br>`vllm/v1/worker/xpu_worker.py` (L140,157)<br>`vllm/v1/worker/cpu_worker.py` (L95,97,98,...)<br>... |
| `--kv-transfer-config` | None | `vllm/v1/utils.py` (L663,664)<br>`vllm/v1/kv_offload/factory.py` (L34,35,36,...)<br>`vllm/v1/kv_offload/base.py` (L453,454,455)<br>... |
| `--kv-events-config` | None | `vllm/v1/kv_offload/base.py` (L456,457,459)<br>`vllm/v1/kv_offload/cpu/spec.py` (L128)<br>`vllm/v1/kv_offload/tiering/spec.py` (L72,117,149)<br>... |
| `--ec-transfer-config` | None | `vllm/v1/core/sched/scheduler.py` (L159)<br>`vllm/v1/executor/ray_executor.py` (L93,94)<br>`vllm/v1/engine/core.py` (L205,206)<br>... |
| `--reasoning-config` | None | `vllm/v1/sample/thinking_budget_state.py` (L21,26,29,...)<br>`vllm/v1/worker/gpu_input_batch.py` (L108,111)<br>`vllm/v1/worker/gpu_model_runner.py` (L686,689,7017)<br>... |
| `--generation-config` | ModelConfig.generation_config | `vllm/sampling_params.py` (L610,613,623)<br>`vllm/config/diffusion.py` (L26)<br>`vllm/config/model.py` (L277,1412,1422,...)<br>... |
| `--enable-sleep-mode` | ModelConfig.enable_sleep_mode | `vllm/v1/worker/gpu_worker.py` (L211)<br>`vllm/config/model.py` (L289,520) |
| `--enable-cumem-allocator` | ModelConfig.enable_cumem_allocator | `vllm/v1/worker/gpu_worker.py` (L205)<br>`vllm/config/vllm.py` (L838,844)<br>`vllm/config/model.py` (L292,523,527,...) |
| `--override-generation-config` | None | `vllm/config/model.py` (L284,1436,1453)<br>`vllm/model_executor/models/llama4.py` (L718)<br>`vllm/model_executor/models/config.py` (L167,168)<br>... |
| `--model-impl` | ModelConfig.model_impl | `vllm/config/model.py` (L299,602,775)<br>`vllm/model_executor/models/registry.py` (L1126,1138,1185,...)<br>`vllm/model_executor/model_loader/utils.py` (L194,195,225) |
| `--override-attention-dtype` | ModelConfig.override_attention_dtype | `vllm/config/model.py` (L308,386,514) |
| `--attention-backend` | AttentionConfig.backend | `tools/pre_commit/generate_attention_backend_docs.py` (L1478,1481)<br>`vllm/v1/utils.py` (L667,717)<br>`vllm/v1/spec_decode/llm_base_proposer.py` (L1212)<br>... |
| `--calculate-kv-scales` | CacheConfig.calculate_kv_scales | `vllm/v1/worker/gpu_model_runner.py` (L466,4297,4300)<br>`vllm/config/cache.py` (L110,256,258,...)<br>`vllm/model_executor/layers/attention/attention.py` (L241,244,255,...)<br>... |
| `--kv-cache-dtype-skip-layers` | None | `vllm/v1/attention/backends/turboquant_attn.py` (L151)<br>`vllm/platforms/interface.py` (L699)<br>`vllm/config/cache.py` (L115)<br>... |
| `--mamba-cache-dtype` | CacheConfig.mamba_cache_dtype | `vllm/v1/attention/backends/flash_attn.py` (L86)<br>`vllm/config/cache.py` (L125,131,132)<br>`vllm/model_executor/layers/mamba/short_conv.py` (L216)<br>... |
| `--mamba-ssm-cache-dtype` | CacheConfig.mamba_ssm_cache_dtype | `vllm/v1/attention/backends/flash_attn.py` (L85)<br>`vllm/config/vllm.py` (L900)<br>`vllm/config/cache.py` (L129)<br>... |
| `--mamba-block-size` | None | `vllm/v1/attention/backends/mamba_attn.py` (L340,344,347,...)<br>`vllm/platforms/interface.py` (L735,736,737,...)<br>`vllm/platforms/xpu.py` (L278)<br>... |
| `--mamba-cache-mode` | CacheConfig.mamba_cache_mode | `vllm/v1/kv_cache_interface.py` (L634,649,654)<br>`vllm/v1/attention/backends/mamba_attn.py` (L111,192,435,...)<br>`vllm/v1/attention/backends/mamba1_attn.py` (L49)<br>... |
| `--mamba-backend` | MambaBackendEnum.TRITON | Nội bộ trong EngineArgs |
| `--enable-mamba-cache-stochastic-rounding` | MambaConfig.enable_stochastic_rounding | Nội bộ trong EngineArgs |
| `--mamba-cache-philox-rounds` | MambaConfig.stochastic_rounding_philox_rounds | Nội bộ trong EngineArgs |
| `--additional-config` | None | `vllm/config/vllm.py` (L358,476,477,...)<br>`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (L168,170,171) |
| `--use-tqdm-on-load` | LoadConfig.use_tqdm_on_load | `vllm/v1/worker/gpu_model_runner.py` (L6717)<br>`vllm/config/load.py` (L101)<br>`vllm/model_executor/model_loader/default_loader.py` (L264,270,275,...)<br>... |
| `--pt-load-map-location` | LoadConfig.pt_load_map_location | `vllm/config/load.py` (L104)<br>`vllm/model_executor/model_loader/default_loader.py` (L304,313)<br>`vllm/model_executor/model_loader/weight_utils.py` (L1136,1146,1155,...)<br>... |
| `--logits-processors` | ModelConfig.logits_processors | `vllm/sampling_params.py` (L820)<br>`vllm/v1/sample/logits_processor/__init__.py` (L47,87,100,...)<br>`vllm/v1/worker/gpu_model_runner.py` (L654,656)<br>... |
| `--async-scheduling` | SchedulerConfig.async_scheduling | `vllm/v1/core/sched/scheduler.py` (L1244)<br>`vllm/v1/executor/multiproc_executor.py` (L637)<br>`vllm/v1/worker/gpu_input_batch.py` (L60)<br>... |
| `--stream-interval` | SchedulerConfig.stream_interval | `vllm/v1/engine/async_llm.py` (L141)<br>`vllm/v1/engine/output_processor.py` (L147,182,220,...)<br>`vllm/v1/engine/llm_engine.py` (L100)<br>... |
| `--kv-sharing-fast-prefill` | CacheConfig.kv_sharing_fast_prefill | `vllm/v1/worker/gpu_model_runner.py` (L812,2372,4126,...)<br>`vllm/v1/engine/async_llm.py` (L306)<br>`vllm/config/vllm.py` (L1301,2082)<br>... |
| `--optimization-level` | VllmConfig.optimization_level | `vllm/config/vllm.py` (L364,1136,1162)<br>`vllm/utils/argparse_utils.py` (L324,325,326) |
| `--performance-mode` | VllmConfig.performance_mode | `vllm/config/vllm.py` (L370,860,861,...) |
| `--kv-offloading-size` | CacheConfig.kv_offloading_size | `vllm/config/vllm.py` (L781,782,798,...)<br>`vllm/config/cache.py` (L176,185) |
| `--kv-offloading-backend` | CacheConfig.kv_offloading_backend | `vllm/config/vllm.py` (L785,791,800)<br>`vllm/config/cache.py` (L180,182) |
| `--tokens-only` | False | `vllm/entrypoints/generate/api_router.py` (L187)<br>`vllm/entrypoints/serve/disagg/api_router.py` (L80)<br>`vllm/entrypoints/openai/cli_args.py` (L154) |
| `--shutdown-timeout` | 0 | `vllm/v1/engine/core.py` (L1330,1331,1336,...)<br>`vllm/config/vllm.py` (L380)<br>`vllm/entrypoints/launcher.py` (L113)<br>... |
| `--weight-transfer-config` | None | `vllm/v1/worker/gpu_worker.py` (L386,388,1028)<br>`vllm/config/vllm.py` (L377) |
| `--fail-on-environ-validation` | False | Nội bộ trong EngineArgs |
| `--gdn-prefill-backend` | None | `vllm/v1/attention/backends/gdn_attn.py` (L103,104,356)<br>`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (L170,296,555,...) |
