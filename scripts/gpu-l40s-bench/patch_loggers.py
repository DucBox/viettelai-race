#!/usr/bin/env python3
# Chen 1 dong log INFO "REQSTAT ..." per-request vao vLLM v1 loggers, de lay
# queue_time / prefill_time / decode_time / mean_tpot / cached_tokens cho TUNG
# request (Prometheus /metrics chi cho aggregate). merge_request_metrics.py se
# join cac dong REQSTAT nay voi client replay theo server_request_id.
# Idempotent: chay lai nhieu lan khong nhan doi patch.
import vllm.v1.metrics.loggers as _lg
path = _lg.__file__  # path-agnostic: hoạt động cả khi vLLM cài trong venv/site-packages
with open(path) as f:
    content = f.read()

if "REQSTAT request_id=%s" in content:
    print("da patch tu truoc, bo qua")
    raise SystemExit(0)

old = "        for finished_request in iteration_stats.finished_requests:\n"
assert content.count(old) == 1, f"expected 1 match, got {content.count(old)}"

new = old + (
    '            logger.info(\n'
    '                "REQSTAT request_id=%s queued_time=%.4f prefill_time=%.4f "\n'
    '                "decode_time=%.4f inference_time=%.4f e2e_latency=%.4f "\n'
    '                "mean_tpot=%.4f num_prompt_tokens=%d num_generation_tokens=%d "\n'
    '                "num_cached_tokens=%d finish_reason=%s",\n'
    '                finished_request.request_id,\n'
    '                finished_request.queued_time,\n'
    '                finished_request.prefill_time,\n'
    '                finished_request.decode_time,\n'
    '                finished_request.inference_time,\n'
    '                finished_request.e2e_latency,\n'
    '                finished_request.mean_time_per_output_token,\n'
    '                finished_request.num_prompt_tokens,\n'
    '                finished_request.num_generation_tokens,\n'
    '                finished_request.num_cached_tokens,\n'
    '                finished_request.finish_reason,\n'
    '            )\n'
)
content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
print("patched OK")
