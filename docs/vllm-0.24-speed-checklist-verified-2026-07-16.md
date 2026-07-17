# vLLM 0.24 Speed Checklist, Đã Check Chéo Source

Ngày cập nhật: 2026-07-16

Phạm vi tài liệu này:
- Chỉ tối ưu `serving speed`, `TTFT`, `TPOT`.
- Bỏ qua accuracy nếu cần, miễn vẫn là tối ưu serving hợp lệ và không chạm anti-cheat.
- Base trên:
  - kết quả A/B trong repo hiện tại
  - survey tài liệu `vLLM`, issue/paper hệ thống serving
  - PDF nội bộ: `docs/Tối Ưu Hóa vLLM Cực Hạn Trên MIG H200 (18GB VRAM, 3 CPU Cores).pdf`
  - source vendored `vendor/vllm-0.24.0`

Mục tiêu của file này không phải là liệt kê mọi cờ. Mục tiêu là:
- chỉ ra `vấn đề thật`
- giải thích `vì sao cờ/kỹ thuật đó đáng làm`
- nói rõ `làm như nào`
- đánh dấu `đã verify trong source vLLM 0.24 hay chưa`
- nói thẳng `nên tin`, `nên A/B`, hay `không bê nguyên`

## Kết luận ngắn

Nếu chỉ săn tốc độ, 4 hướng đáng tiền nhất hiện tại là:

1. Giảm tải `CPU ingress / tokenizer / frontend_prep`
2. Điều khiển `scheduler budget` và `decode concurrency`
3. Đẩy nhiều hơn bước `sampling/postprocess` xuống GPU
4. Tối ưu riêng cho kiến trúc `Qwen3.5 hybrid GDN + attention`, không áp công thức Transformer thường

Điểm quan trọng nhất sau khi check source `vLLM 0.24`:
- Một số ý trong PDF nội bộ là đúng về bản chất CPU-starved.
- Nhưng vài khuyến nghị trong PDF là `không nên bê nguyên` cho `vLLM 0.24 + Qwen3.5 hybrid`.

## Mức ưu tiên

### P0: nên làm hoặc A/B đầu tiên

- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- `VLLM_USE_FASTOKENS=1`
- `--renderer-num-workers=2` kèm `--mm-processor-cache-gb 0` nếu cần
- `--max-num-seqs=10` làm baseline TPOT-first
- micro-sweep `--max-num-batched-tokens` quanh `2174 / 2208 / 2304 / 3216`
- `--gdn-prefill-backend=flashinfer`
- `--disable-log-stats`, giữ request logging ở trạng thái off
- `VLLM_USE_FLASHINFER_SAMPLER=1`

### P1: đáng A/B tiếp theo

- `--async-scheduling`
- `--prefix-caching-hash-algo=xxhash`
- `--disable-hybrid-kv-cache-manager`
- `--speculative-config` theo hướng `ngram`, nhưng phải smoke test đúng runner/model path

### P2: advanced, chỉ làm khi đã bóc xong P0/P1

- `--mamba-block-size`
- `--kv-sharing-fast-prefill`
- `--kv-cache-memory-bytes` thay vì chỉ `gpu-memory-utilization`
- `--compilation-config`, `--performance-mode`, `--optimization-level`

## Checklist triển khai

## 1. CPU Thread Hygiene

- [ ] Ép toàn bộ CPU math libs về 1 thread

Vấn đề:
- Máy chấm kiểu `MIG H200 18GB + 3 CPU cores` rất dễ chết vì CPU oversubscription hơn là vì GPU compute thiếu.

Vì sao nên làm:
- Nếu OpenMP / MKL / OpenBLAS tự bung thread pool, scheduler loop và API/input path của `vLLM` sẽ bị tranh CPU vô ích.

Làm như nào:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

Check chéo `vLLM 0.24`:
- Đây không phải cờ của `vLLM`, mà là hygiene mức process/OS.
- Kết luận: nên xem như baseline bắt buộc.

Trạng thái:
- `NÊN GIỮ`

## 2. Fast Tokenizer Backend

- [ ] Thử `VLLM_USE_FASTOKENS=1`

Vấn đề:
- `frontend_prep` đang là một phần lớn của `TTFT`.
- Repo hiện tại đã thấy tokenize/template/handoff là một nguồn nghẽn thật.

Vì sao nên làm:
- `fastokens` thay Rust BPE backend bên dưới HF fast tokenizer bằng shim nhanh hơn.
- Nếu bài đang CPU-bound ở tokenize thì đây là cú đánh đúng chỗ.

Làm như nào:

```bash
export VLLM_USE_FASTOKENS=1
```

Điều kiện:
- package `fastokens >= 0.2.0` phải có trong image.

Check chéo `vLLM 0.24`:
- Env var có thật: [envs.py](../vendor/vllm-0.24.0/vllm/envs.py#L663)
- Patch tokenizer có thật: [fastokens.py](../vendor/vllm-0.24.0/vllm/tokenizers/fastokens.py#L1)
- Model config cũng ghi rõ cơ chế này: [model.py](../vendor/vllm-0.24.0/vllm/config/model.py#L128)

Trạng thái:
- `P0 - A/B SỚM`

## 3. Renderer Workers

- [ ] A/B `--renderer-num-workers=2`

Vấn đề:
- Prompt dài làm CPU frontend bị backlog.
- Đây là chỗ có thể giảm `TTFT tail` mà không phải đụng scheduler sâu ngay lập tức.

Vì sao nên làm:
- Song song hóa phần renderer/tokenizer phía frontend.

Làm như nào:

```bash
--renderer-num-workers=2
--mm-processor-cache-gb=0
```

Ghi chú:
- Với text-only, `mm_processor_cache_gb=0` là cách an toàn để tránh validation fail khi `renderer-num-workers > 1`.

Check chéo `vLLM 0.24`:
- Flag có thật: [arg_utils.py](../vendor/vllm-0.24.0/vllm/engine/arg_utils.py#L876)
- Guard thread-safety có thật: [model.py](../vendor/vllm-0.24.0/vllm/config/model.py#L703)

Check chéo repo:
- `v40` đã cho thấy `renderer=2` giảm `TTFT p95` đáng kể trong regime default.

Trạng thái:
- `P0 - A/B SỚM`

## 4. Không Tin Nguyên Xi `api-server-count`

- [ ] Chỉ thử `--api-server-count` nếu đã chứng minh ingress là nghẽn chính

Vấn đề:
- Ý tưởng tăng số process API server nghe có vẻ hợp lý khi CPU frontend chậm.

Vì sao chưa nên tin ngay:
- `vLLM 0.24` V1 mặc định đã là multi-process architecture.
- Tăng `api-server-count` không giải quyết engine core đơn lẻ; thậm chí có thể tăng ZMQ/context-switch overhead.

Làm như nào:
- Không xem là default win.
- Chỉ A/B sau khi đã thử `fastokens` và `renderer-num-workers`.

Check chéo `vLLM 0.24`:
- Flag có thật: [cli_args.py](../vendor/vllm-0.24.0/vllm/entrypoints/openai/cli_args.py#L361)
- Docs kiến trúc nói rõ V1 mặc định đã có `1 API server process`: [arch_overview.md](../vendor/vllm-0.24.0/docs/design/arch_overview.md#L81)
- Docs metrics nói multiprocess mode thực tế liên quan `--api-server-count > 1`: [metrics.md](../vendor/vllm-0.24.0/docs/design/metrics.md#L82)

Check chéo repo:
- Repo đã có A/B và ghi nhận `api-server-count` có thể hại.

Trạng thái:
- `KHÔNG XEM LÀ WIN MẶC ĐỊNH`

## 5. Bỏ Hẳn Ý `disable-frontend-multiprocessing` Từ PDF

- [ ] Không mang nguyên khuyến nghị `--disable-frontend-multiprocessing` vào plan `v0.24`

Vấn đề:
- PDF nội bộ coi frontend multiprocessing là nghẽn chính và khuyên tắt nó.

Vì sao không bê nguyên:
- Trong source vendored `vLLM 0.24`, không thấy cờ CLI này trong `arg_utils.py`.
- Kiến trúc V1 hiện tại đã thay đổi so với tư duy cũ.

Check chéo `vLLM 0.24`:
- Không tìm thấy cờ này trong `vendor/vllm-0.24.0/vllm/engine/arg_utils.py`
- Docs V1 mô tả multi-process là kiến trúc mặc định: [arch_overview.md](../vendor/vllm-0.24.0/docs/design/arch_overview.md#L81)

Kết luận:
- Insight “CPU/frontend là nghẽn” là đúng.
- Nhưng `cách sửa` trong PDF không còn map 1-1 sang `vLLM 0.24`.

Trạng thái:
- `KHÔNG DÙNG`

## 6. Decode Concurrency Là Núm TPOT Số 1

- [ ] Giữ baseline `--max-num-seqs=10`

Vấn đề:
- Khi concurrency cao, `TTFT` đẹp nhưng `TPOT` đội lên mạnh.
- Workload của bạn đang ở regime mà `TPOT` gần floor đáng giá hơn.

Vì sao nên làm:
- Repo thực nghiệm đã chứng minh `max-num-seqs` là lever lớn nhất cho `TPOT`.

Làm như nào:

```bash
--max-num-seqs=10
```

Sau đó chỉ quét hẹp quanh `8 / 10 / 12` nếu còn thời gian.

Check chéo `vLLM 0.24`:
- Flag có thật: [arg_utils.py](../vendor/vllm-0.24.0/vllm/engine/arg_utils.py#L1400)
- Dính trực tiếp tới cudagraph dispatcher và attention backends theo trace source: [docs/vllm_v0.24.0_all_flags_deep_trace.md](vllm_v0.24.0_all_flags_deep_trace.md)

Check chéo repo:
- `v24/v25/v26/v31` đều xác nhận vùng `seqs=10` mạnh hơn `20/128` cho score hiện tại.

Trạng thái:
- `P0 - BASELINE CHÍNH`

## 7. Scheduler Token Budget Là Núm TTFT Chính

- [ ] Micro-sweep `--max-num-batched-tokens`

Vấn đề:
- Prompt dài bị cắt thành nhiều prefill steps thì queue và TTFT đội lên.
- Nhưng tăng quá tay lại làm step dài hơn, đẩy `TPOT` xấu đi.

Vì sao nên làm:
- Đây là núm cân trực tiếp giữa `queue/prefill` và `decode`.

Làm như nào:
- Giữ `seqs=10`, sweep:
  - `2174`
  - `2208`
  - `2304`
  - `3216`

Vùng này hợp lý vì repo đã chỉ ra regime block-align quanh `2144` với `FP8 KV`, và `2174` cho headroom nhỏ để không rớt về 1 block khi decode chen vào.

Check chéo `vLLM 0.24`:
- Flag có thật: [arg_utils.py](../vendor/vllm-0.24.0/vllm/engine/arg_utils.py#L1393)
- Chunked prefill là cờ thật: [arg_utils.py](../vendor/vllm-0.24.0/vllm/engine/arg_utils.py#L1423)
- Scheduler docs/source trong repo đã trace cụ thể cửa này: [vllm-v0.24.0-ttft-tpot-breakdown-rootcause.md](vllm-v0.24.0-ttft-tpot-breakdown-rootcause.md)

Check chéo repo:
- `v26/v31` là vùng thắng lớn hiện tại.

Trạng thái:
- `P0 - A/B CHÍNH`

## 8. Chunked Prefill: Giữ, Nhưng Đừng Tin Công Thức Chung

- [ ] Giữ `enable-chunked-prefill`, nhưng tune qua budget chứ không theo khẩu hiệu

Vấn đề:
- PDF nội bộ đúng ở chỗ chunk quá nhỏ làm scheduler bị gọi lại liên tục.

Vì sao phải cẩn thận:
- Với model hybrid của bạn, chunking behavior còn bị chi phối bởi Mamba/GDN alignment.

Làm như nào:
- Giữ chunked prefill.
- Tune thông qua `max-num-batched-tokens` trước.
- Chỉ đụng `long_prefill_token_threshold` hoặc partial-prefill sau.

Check chéo `vLLM 0.24`:
- `enable_chunked_prefill` default thật: [scheduler.py](../vendor/vllm-0.24.0/vllm/config/scheduler.py#L84)
- `long_prefill_token_threshold` mặc định là `0`: [scheduler.py](../vendor/vllm-0.24.0/vllm/config/scheduler.py#L80)
- Giá trị auto `0.04 * max_model_len` chỉ kích hoạt khi `max_num_partial_prefills > 1`: [scheduler.py](../vendor/vllm-0.24.0/vllm/config/scheduler.py#L257)

Trạng thái:
- `P0 CHO BUDGET`, `P2 CHO THRESHOLD/PARTIAL`

## 9. Async Scheduling

- [ ] A/B `--async-scheduling`

Vấn đề:
- Scheduler Python overhead mỗi step có thể ăn vào `TPOT`.

Vì sao nên làm:
- Ý tưởng là overlap scheduler CPU với GPU execution, che bớt phần book-keeping / input prep.

Làm như nào:

```bash
--async-scheduling
```

Check chéo `vLLM 0.24`:
- Flag có thật: [arg_utils.py](../vendor/vllm-0.24.0/vllm/engine/arg_utils.py#L1449)
- Có code path thật trong scheduler / executor / input batch:
  - [scheduler.py](../vendor/vllm-0.24.0/vllm/v1/core/sched/scheduler.py#L1244)
  - [docs/vllm_v0.24.0_all_flags_deep_trace.md](vllm_v0.24.0_all_flags_deep_trace.md)

Check chéo repo:
- Có dấu hiệu đây là lever thật, nhưng chưa phải win mặc định tuyệt đối cho mọi regime.

Trạng thái:
- `P1 - NÊN A/B`

## 10. GDN Prefill Backend

- [ ] Giữ `--gdn-prefill-backend=flashinfer`

Vấn đề:
- `Qwen3.5` hybrid có nhiều layer GDN; backend prefill của GDN là lever riêng của model này.

Vì sao nên làm:
- Đây là knob kiến trúc-specific, ít đội khai thác đúng nếu chỉ nghĩ theo Transformer thường.

Làm như nào:

```bash
--gdn-prefill-backend=flashinfer
```

Check chéo `vLLM 0.24`:
- Flag có thật: [arg_utils.py](../vendor/vllm-0.24.0/vllm/engine/arg_utils.py#L1570)
- Có code path thật: [gdn_attn.py](../vendor/vllm-0.24.0/vllm/v1/attention/backends/gdn_attn.py)

Check chéo repo:
- `flashinfer` đã thắng `triton/cutedsl` trong các A/B trước đó.

Trạng thái:
- `P0 - GIỮ`

## 11. FlashInfer Sampler

- [ ] Giữ hoặc explicit set `VLLM_USE_FLASHINFER_SAMPLER=1`

Vấn đề:
- Top-k/top-p/sampling và một phần postprocess có thể kéo CPU và làm chậm vòng decode.

Vì sao nên làm:
- Trên CUDA, `vLLM 0.24` ưu tiên FlashInfer sampler khi support được.

Làm như nào:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=1
```

Hoặc để mặc định nếu build đã hỗ trợ.

Check chéo `vLLM 0.24`:
- Env var có thật và default là `True` nếu hỗ trợ: [envs.py](../vendor/vllm-0.24.0/vllm/envs.py#L796)
- Dispatch logic thật: [topk_topp_sampler.py](../vendor/vllm-0.24.0/vllm/v1/sample/ops/topk_topp_sampler.py#L16)

Kết luận:
- Không phải “ý tưởng bên ngoài”; đây là nhánh code thật trong `v0.24`.

Trạng thái:
- `P0 - GIỮ / EXPLICIT CHO RÕ`

## 12. Greedy Và Không Logprobs

- [ ] Dùng `temperature=0`
- [ ] Không yêu cầu `logprobs`

Vấn đề:
- Sampling phức tạp và logprobs làm tăng CPU/GPU postprocess.

Vì sao nên làm:
- Nếu chỉ tối ưu speed, greedy search là cấu hình hợp lệ và trực tiếp giảm hậu xử lý.
- Logprobs làm vLLM phải giữ thêm dữ liệu và xử lý thêm.

Làm như nào:
- Buộc request path dùng `temperature=0`.
- Không bật `logprobs` ở client/benchmark.

Check chéo `vLLM 0.24`:
- `max_logprobs` và `logprobs_mode` là config thật: [docs/vllm_v0.24.0_all_flags_deep_trace.md](vllm_v0.24.0_all_flags_deep_trace.md)
- `kv-sharing-fast-prefill` còn bị cảnh báo sai logprobs cho prompt tokens, càng củng cố việc speed path nên tránh logprobs:
  - [async_llm.py](../vendor/vllm-0.24.0/vllm/v1/engine/async_llm.py#L305)
  - [config/vllm.py](../vendor/vllm-0.24.0/vllm/config/vllm.py#L1313)

Trạng thái:
- `P0 - GIỮ`

## 13. Logging Hygiene

- [ ] Bật `--disable-log-stats`
- [ ] Giữ request logging ở trạng thái off

Vấn đề:
- Logging định kỳ và request logging là gánh nặng không cần thiết trên CPU bóp nghẹt.

Vì sao nên làm:
- Có thể giảm overhead nhỏ nhưng sạch sẽ.

Làm như nào:

```bash
--disable-log-stats
# KHÔNG bật --enable-log-requests
```

Check chéo `vLLM 0.24`:
- `--disable-log-stats` có thật: [arg_utils.py](../vendor/vllm-0.24.0/vllm/engine/arg_utils.py#L1542)
- Trong `v0.24`, request logging là cờ `--enable-log-requests`, mặc định off:
  - [arg_utils.py](../vendor/vllm-0.24.0/vllm/engine/arg_utils.py#L2677)

Kết luận:
- PDF nói `--disable-log-requests` theo kiểu cũ. Với `v0.24`, cách hiểu đúng là `đừng bật --enable-log-requests`.

Trạng thái:
- `P0 - GIỮ`

## 14. Prefix Caching: Đừng Tắt Vội

- [ ] Giữ `--enable-prefix-caching`
- [ ] Nếu nghi CPU hash, thử `--prefix-caching-hash-algo=xxhash`

Vấn đề:
- PDF nội bộ nghi ngờ prefix cache làm CPU hash nặng và khuyên tắt.

Vì sao chưa nên tắt:
- Trace/bài của bạn có tính prefix continuity.
- Với model hybrid, prefix cache đúng là có bug/rủi ro, nhưng đây vẫn là nhánh đáng verify thay vì tắt mặc định.

Làm như nào:

```bash
--enable-prefix-caching
--prefix-caching-hash-algo=xxhash
```

Sau đó đo hit thật bằng tooling trong repo.

Check chéo `vLLM 0.24`:
- `enable-prefix-caching` có thật: [arg_utils.py](../vendor/vllm-0.24.0/vllm/engine/arg_utils.py#L1158)
- `prefix-caching-hash-algo` có thật: [arg_utils.py](../vendor/vllm-0.24.0/vllm/engine/arg_utils.py#L1165)
- Docs thiết kế prefix caching có nhắc rõ hashing algorithm:
  - [prefix_caching.md](../vendor/vllm-0.24.0/docs/design/prefix_caching.md)

Check chéo repo:
- Repo đã có ghi chú rõ prefix caching trên hybrid là experimental/bug-prone, nhưng chưa đủ để kết luận “tắt hết là tốt nhất”.

Trạng thái:
- `P1 - VERIFY, KHÔNG TẮT BỪA`

## 15. Block Size / Mamba Block Size

- [ ] Không mang thẳng khuyến nghị `--block-size 32` từ PDF sang config hiện tại
- [ ] Chỉ thử `--mamba-block-size` ở vòng advanced

Vấn đề:
- PDF nội bộ khuyên block lớn để giảm pointer chasing của scheduler.

Vì sao phải cẩn thận:
- `Qwen3.5 hybrid` không hành xử như Transformer thường.
- Repo hiện tại đã chỉ ra alignment/block regime đặc biệt xoay quanh GDN/Mamba.

Làm như nào:
- Không xem `block-size 32` là default fix.
- Nếu muốn đào sâu hybrid cache manager, hãy thử `mamba-block-size` có kiểm chứng.

Check chéo `vLLM 0.24`:
- `--block-size` có thật: [docs/vllm_v0.24.0_all_flags_deep_trace.md](vllm_v0.24.0_all_flags_deep_trace.md)
- `--mamba-block-size` có thật: [arg_utils.py](../vendor/vllm-0.24.0/vllm/engine/arg_utils.py#L1183)
- Source còn bắt buộc `mamba-block-size` chỉ dùng khi prefix caching bật: [config/vllm.py](../vendor/vllm-0.24.0/vllm/config/vllm.py#L2176)

Trạng thái:
- `P2 - ADVANCED`

## 16. Hybrid KV Cache Manager

- [ ] A/B `--disable-hybrid-kv-cache-manager`

Vấn đề:
- Với model hybrid, manager mặc định có thể không phải luôn tối ưu hoặc có bug/regression theo cấu hình.

Vì sao nên thử:
- Đây là một trong ít lever rất đúng “kiến trúc-specific”.

Làm như nào:

```bash
--disable-hybrid-kv-cache-manager
```

Check chéo `vLLM 0.24`:
- Flag có thật: [arg_utils.py](../vendor/vllm-0.24.0/vllm/engine/arg_utils.py#L1445)
- Code validation có thật: [config/vllm.py](../vendor/vllm-0.24.0/vllm/config/vllm.py#L1518)
- Deep trace trong repo cũng đã đánh dấu nó là lever thật: [vllm_v0.24.0_all_flags_deep_trace.md](vllm_v0.24.0_all_flags_deep_trace.md)

Trạng thái:
- `P1 - A/B`

## 17. KV Sharing Fast Prefill

- [ ] Không bật mặc định; chỉ thử khi đã hiểu rõ model path

Vấn đề:
- Cờ này nghe rất hấp dẫn vì tên gợi ý tăng tốc prefill.

Vì sao phải cực kỳ cẩn thận:
- Source `v0.24` cảnh báo rõ:
  - cần thay đổi phía model để đúng correctness và có savings thật
  - sai logprobs cho prompt tokens trong một số trường hợp

Check chéo `vLLM 0.24`:
- Flag có thật: [arg_utils.py](../vendor/vllm-0.24.0/vllm/engine/arg_utils.py#L1174)
- Async engine chặn case prompt logprobs: [async_llm.py](../vendor/vllm-0.24.0/vllm/v1/engine/async_llm.py#L305)
- Config warning rất rõ: [config/vllm.py](../vendor/vllm-0.24.0/vllm/config/vllm.py#L1313)

Trạng thái:
- `P2 - KHÔNG BẬT BỪA`

## 18. Speculative Decoding: Chỉ Còn Cửa `ngram`

- [ ] Nếu thử speculative, ưu tiên `ngram`, không quay lại MTP

Vấn đề:
- Repo đã cho thấy `MTP` làm chậm trên case này.

Vì sao vẫn còn đáng thử:
- `ngram` không cần draft model riêng.
- Nếu prompt/output có pattern lặp, đây có thể là đòn TPOT đáng kể.

Làm như nào:
- Thử một cấu hình nhỏ, ví dụ:

```bash
--speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":2,"prompt_lookup_min":2}'
```

Check chéo `vLLM 0.24`:
- `--speculative-config` có thật: [arg_utils.py](../vendor/vllm-0.24.0/vllm/engine/arg_utils.py#L1498)
- `ngram` có config/schema thật:
  - [speculative.py](../vendor/vllm-0.24.0/vllm/config/speculative.py#L141)
  - [docs/features/speculative_decoding/README.md](../vendor/vllm-0.24.0/docs/features/speculative_decoding/README.md#L107)
- `ngram` proposer có code path thật:
  - [ngram_proposer.py](../vendor/vllm-0.24.0/vllm/v1/spec_decode/ngram_proposer.py#L16)
  - [gpu_model_runner.py](../vendor/vllm-0.24.0/vllm/v1/worker/gpu_model_runner.py#L564)

Nuance quan trọng:
- Trong `config/vllm.py` có comment rằng `ngram/ngram_gpu` chưa support ở `v2 model runner`: [config/vllm.py](../vendor/vllm-0.24.0/vllm/config/vllm.py#L2027)
- Nghĩa là: `source có thật`, nhưng phải smoke test đúng runner/model path của Qwen3.5 hiện tại.

Trạng thái:
- `P1 - CÓ THỂ ĐÁNG GIÁ, NHƯNG PHẢI SMOKE TEST`

## 19. CUDA Graph / Compile / VRAM

- [ ] Đừng ưu tiên quá sớm, nhưng giữ trong backlog

Vấn đề:
- CPU kernel launch overhead là thật.

Vì sao chưa phải ưu tiên số 1:
- Repo hiện tại cho thấy `seqs/budget/frontend` còn là lever lớn hơn.
- Tự tay thêm compile/capture sizes chưa chắc thắng, có thể chỉ tăng warmup/memory.

Làm như nào:
- Giữ `enforce_eager` ở trạng thái không phá cudagraph.
- Chỉ thử `--compilation-config`, `--performance-mode`, `--optimization-level`, `--cudagraph-capture-sizes` sau khi P0/P1 đã sạch.

Check chéo `vLLM 0.24`:
- Các cờ này đều có thật trong `arg_utils.py`, nhưng chưa có bằng chứng mạnh bằng các lever P0/P1 cho case repo này.

Trạng thái:
- `P2 - SAU`

## Những ý từ PDF nên giữ, sửa, hoặc bỏ

### Giữ

- CPU mới là nghẽn chính ở cấu hình `3 cores`
- FlashInfer là hướng đúng cho Hopper + decode path
- Logging hygiene là hợp lý
- Greedy / bỏ logprobs là hợp lệ nếu chỉ tối ưu speed

### Sửa cách hiểu

- PDF nói “tắt frontend multiprocessing”
- Trong `vLLM 0.24`, cần đổi sang tư duy:
  - giảm CPU ingress bằng `fastokens`, `renderer-num-workers`
  - không lạm dụng `api-server-count`

- PDF nói “prefix cache có thể hại vì CPU hash”
- Trong case repo này:
  - không tắt ngay
  - verify hit-rate thật
  - nếu nghi CPU hash thì thử `xxhash` trước

### Không bê nguyên

- `--disable-frontend-multiprocessing`
- `--block-size 32` như một default rule
- `--no-enable-prefix-caching` như mặc định
- `api-server-count` như một win chắc chắn

## Checklist chạy A/B gợi ý

### Wave 1: rẻ, sạch, đúng source

- [ ] `fastokens` on
- [ ] `renderer-num-workers=2`, `mm-processor-cache-gb=0`
- [ ] explicit `VLLM_USE_FLASHINFER_SAMPLER=1`
- [ ] `disable-log-stats`, request logging off

### Wave 2: đòn chính TTFT/TPOT

- [ ] giữ `max-num-seqs=10`
- [ ] sweep `max-num-batched-tokens=2174/2208/2304/3216`
- [ ] `async-scheduling`

### Wave 3: hybrid-specific

- [ ] `prefix-caching-hash-algo=xxhash`
- [ ] `disable-hybrid-kv-cache-manager`
- [ ] `mamba-block-size` nếu còn thời gian

### Wave 4: speculative only-if

- [ ] thử `ngram` speculation nhỏ
- [ ] bỏ ngay nếu runner/model path reject hoặc acceptance thấp

## Source Check Summary

Các mục sau đã được xác nhận là `có thật` trong source vendored `vLLM 0.24.0`:

- `VLLM_USE_FASTOKENS`
- `VLLM_USE_FLASHINFER_SAMPLER`
- `--renderer-num-workers`
- `--api-server-count`
- `--enable-prefix-caching`
- `--prefix-caching-hash-algo`
- `--max-num-batched-tokens`
- `--max-num-seqs`
- `--disable-hybrid-kv-cache-manager`
- `--async-scheduling`
- `--speculative-config`
- `--gdn-prefill-backend`
- `--mamba-block-size`
- `--disable-log-stats`
- `--enable-log-requests`
- `--kv-sharing-fast-prefill`

Các mục sau `không nên coi là lever trực tiếp của v0.24` dù PDF có nhắc:

- `--disable-frontend-multiprocessing`
- `--disable-log-requests` theo tên cũ

## Nguồn

- PDF nội bộ: [Tối Ưu Hóa vLLM Cực Hạn Trên MIG H200 (18GB VRAM, 3 CPU Cores).pdf](</Users/ngoquangduc/Desktop/workspace/viettelai-race/docs/Tối Ưu Hóa vLLM Cực Hạn Trên MIG H200 (18GB VRAM, 3 CPU Cores).pdf>)
- Tổng hợp cờ từ source vendored: [vllm_v0.24.0_all_flags_deep_trace.md](vllm_v0.24.0_all_flags_deep_trace.md)
- Phân rã TTFT/TPOT theo code path: [vllm-v0.24.0-ttft-tpot-breakdown-rootcause.md](vllm-v0.24.0-ttft-tpot-breakdown-rootcause.md)
- Guide tối ưu repo hiện tại: [vllm-optimization-guide.md](vllm-optimization-guide.md)
- Source vendored: `vendor/vllm-0.24.0`

