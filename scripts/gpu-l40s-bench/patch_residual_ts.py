#!/usr/bin/env python3
"""Patch #1 cho cây phân rã TTFT: bóc RESIDUAL.

Chèn monkeypatch vào IterationStats.update_from_finished_request (vllm/v1/metrics/
stats.py) để ghi 1 dòng JSON per-request chứa các TIMESTAMP thô mà REQSTAT không
có. Gate bằng env RESIDUAL_TRACE=<file>; không set -> patch trơ (zero-cost).

Ghi mỗi request:
  rid            : request_id (khớp server_request_id ở client)
  arrival_time   : frontend wall-clock lúc engine nhận request (time.time)
  ftl            : first_token_latency = arrival -> first token (clock frontend)
                   = TTFT server-side. Dùng để tách:
                     frontend_prep   = ftl - queued_time - prefill_time  (tokenize+handoff)
                     client_transport= client_ttft - ftl                 (net+HTTP+detok/stream)
  queued_ts      : monotonic core, lúc vào waiting  ─┐ cùng clock với 't' của
  scheduled_ts   : monotonic core, lúc schedule đầu  │ sched trace -> correlate
  first_token_ts : monotonic core, lúc có token đầu  │ per-request để bóc queue
  last_token_ts  : monotonic core, token cuối       ─┘ và prefill-interleave, TPOT
  queued_time    : scheduled_ts - queued_ts (tiện đối chiếu)
  prefill_time   : first_token_ts - scheduled_ts
  num_gen        : số token sinh ra

Lưu ý clock: arrival_time & ftl theo frontend (time.time); *_ts theo core
(monotonic). KHÔNG trộn 2 clock khi trừ. frontend_prep hợp lệ vì cả ftl lẫn
(queued_time+prefill_time) đều KẾT THÚC ở cùng biến cố (first token) -> hiệu =
khoảng arrival->queued, độc lập clock.

    python3 patch_residual_ts.py apply | revert | status
Chạy serve với:  RESIDUAL_TRACE=/root/out/rests.jsonl <serve cmd>
"""
import sys

MARK = "# === RESIDUAL_TRACE PATCH ==="
BLOCK = '''

# === RESIDUAL_TRACE PATCH ===
import os as _r_os, json as _r_json
if _r_os.environ.get("RESIDUAL_TRACE"):
    _r_orig_ufr = IterationStats.update_from_finished_request
    _r_f = open(_r_os.environ["RESIDUAL_TRACE"], "a")
    def _r_ufr(self, finish_reason, request_id, num_prompt_tokens,
               max_tokens_param, req_stats, num_cached_tokens=0):
        _ret = _r_orig_ufr(self, finish_reason, request_id, num_prompt_tokens,
                           max_tokens_param, req_stats, num_cached_tokens)
        try:
            qt = req_stats.scheduled_ts - req_stats.queued_ts
            pt = req_stats.first_token_ts - req_stats.scheduled_ts
            _r_f.write(_r_json.dumps({
                "rid": request_id,
                "arrival_time": round(req_stats.arrival_time, 6),
                "ftl": round(req_stats.first_token_latency, 6),
                "queued_ts": round(req_stats.queued_ts, 6),
                "scheduled_ts": round(req_stats.scheduled_ts, 6),
                "first_token_ts": round(req_stats.first_token_ts, 6),
                "last_token_ts": round(req_stats.last_token_ts, 6),
                "queued_time": round(qt, 6),
                "prefill_time": round(pt, 6),
                "num_gen": int(req_stats.num_generation_tokens),
            }) + "\\n")
            _r_f.flush()
        except Exception:
            pass
        return _ret
    IterationStats.update_from_finished_request = _r_ufr
# === END RESIDUAL_TRACE PATCH ===
'''


def target():
    import vllm.v1.metrics.stats as s
    return s.__file__


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    path = target()
    src = open(path).read()
    patched = MARK in src
    if action == "status":
        print(f"file: {path}\nstatus: {'PATCHED' if patched else 'ORIGINAL'}")
    elif action == "apply":
        if patched:
            print("da patched, bo qua")
        else:
            open(path, "w").write(src + BLOCK)
            print(f"patched -> {path}")
    elif action == "revert":
        if not patched:
            print("chua patched")
        else:
            i = src.index("\n\n" + MARK)
            open(path, "w").write(src[:i] + "\n")
            print(f"reverted -> {path}")
    else:
        sys.exit("action: apply|revert|status")


if __name__ == "__main__":
    main()
