#!/usr/bin/env python3
"""Chèn instrument per-STEP vào V1 scheduler (giống patch_loggers: sửa thẳng
file .py đã cài). Mỗi lần schedule() chạy, ghi 1 dòng JSON vào file trỏ bởi
env SCHED_TRACE. Nếu SCHED_TRACE không set -> không ghi (patch trơ).

Đo được mỗi step:
  sched_ms     : thời gian CPU của schedule() (logic lập lịch)
  exec_gap_ms  : khoảng END(schedule trước) -> START(schedule này)
                 ≈ THỜI GIAN EXECUTE MODEL của step trước (wall-clock 1 iteration)
  n_running    : số request đang active
  n_waiting    : số request đang chờ admit
  n_new_admit  : số request MỚI được admit trong step này
  n_prefilling : trong running, bao nhiêu request còn đang prefill (chưa xong prompt)
  tokens_sched : tổng token được schedule trong step (so với max_num_batched_tokens
                 để biết có bị nghẽn budget không)

    python3 patch_sched_trace.py apply    # chèn
    python3 patch_sched_trace.py revert    # gỡ
    python3 patch_sched_trace.py status
"""
import sys

MARK = "# === SCHED_TRACE PATCH ==="
BLOCK = '''

# === SCHED_TRACE PATCH ===
import time as _t_time, json as _t_json, os as _t_os
if _t_os.environ.get("SCHED_TRACE"):
    _t_orig = Scheduler.schedule
    _t_f = open(_t_os.environ["SCHED_TRACE"], "w")
    _t_last = [None]
    def _t_schedule(self, *_a_args, **_k_args):
        # 0.24: schedule(self, throttle_prefills=False) -> phai forward arg,
        # nếu không TypeError vỡ scheduler (core.py goi schedule(throttle) moi step)
        _a = _t_time.monotonic()
        out = _t_orig(self, *_a_args, **_k_args)
        _b = _t_time.monotonic()
        try:
            nst = getattr(out, "total_num_scheduled_tokens", None)
            if nst is None:
                d = getattr(out, "num_scheduled_tokens", {}) or {}
                nst = sum(d.values())
            n_pref = 0
            pref_ids = []
            for r in self.running:
                try:
                    if r.num_computed_tokens < r.num_prompt_tokens:
                        n_pref += 1
                        pref_ids.append(getattr(r, "request_id", None))
                except Exception:
                    pass
            new_ids = []
            for r in (getattr(out, "scheduled_new_reqs", []) or []):
                new_ids.append(getattr(r, "req_id", getattr(r, "request_id", None)))
            gap = (_a - _t_last[0]) * 1000 if _t_last[0] is not None else 0.0
            _t_f.write(_t_json.dumps({
                "t": round(_b, 6),
                "sched_ms": round((_b - _a) * 1000, 3),
                "exec_gap_ms": round(gap, 3),
                "n_running": len(self.running),
                "n_waiting": len(self.waiting),
                "n_new_admit": len(getattr(out, "scheduled_new_reqs", []) or []),
                "n_prefilling": n_pref,
                "tokens_sched": int(nst),
                "new_ids": new_ids,      # request_id admit trong step (correlate queue)
                "pref_ids": pref_ids,    # request_id đang prefill (correlate prefill-interleave)
            }) + "\\n")
            _t_f.flush()
            _t_last[0] = _b
        except Exception:
            pass
        return out
    Scheduler.schedule = _t_schedule
# === END SCHED_TRACE PATCH ===
'''


def target():
    import vllm.v1.core.sched.scheduler as s
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
