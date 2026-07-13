# Checklist dựng server GPU mới (sau khi thuê)

Đúc kết từ các lần vấp thật (leak VRAM, shim `python3`, `pkill` giết nhầm session).
Làm tuần tự từ trên xuống.

---

## 0. Kết nối — quy tắc vàng

- **SSH 1 LẦN duy nhất** vào 1 `tmux` cố định. SSH lặp lại hay rớt/treo.
  ```bash
  tmux new -s race          # lần đầu
  tmux attach -t race       # rớt mạng thì vào lại
  ```

## 1. Verify sạch trước khi làm gì

```bash
nvidia-smi
```
- **VRAM used phải ~0.** Nếu còn process ma (CUDA context orphan, không reclaim được)
  → **trả máy thuê máy khác**. Đừng cố `gpu-reset` (container share thường
  *Insufficient Permissions*).
- Ghi lại: tên GPU (L40S / H200-slice), VRAM, và **`sm` version**
  (Ada = sm89 / Hopper = sm90) — quyết định W4AFP8 có chạy được không.

## 2. Bẫy môi trường — phải né

- **`python3` có thể là shim** (`/opt/sys-venv/shim`, KHÔNG có vllm).
  Luôn dùng **`/usr/bin/python3`**. Verify:
  ```bash
  /usr/bin/python3 -c "import vllm; print(vllm.__version__)"   # phải ra 0.22.1
  ```
- **KHÔNG `pkill -f vllm`** (chuỗi "vllm" khớp cả lệnh SSH đang chạy → tự giết
  session). Luôn dùng full pattern:
  ```bash
  pkill -f "vllm.entrypoints.openai.api_server"
  ```
- Entrypoint serve = `python3 -m vllm.entrypoints.openai.api_server`
  (KHÔNG đổi sang `vllm-server` — khớp ràng buộc compose submit).

## 3. Kill hygiene — chống stack VRAM giữa các rep

- Kill = **SIGTERM trước** (`pkill -15`), poll `memory.used < 1000 MiB`, chỉ
  `SIGKILL + fuser` khi timeout. `SIGKILL` thẳng để lại CUDA context → server
  chồng nhau → rep sau OOM. (Logic này đã nằm trong `kill_server()` của `run_ab.sh`.)
- Trước mỗi rep A/B: cold restart, verify VRAM load về đúng 1 con số ở mọi rep.

## 4. Copy code + artefact lên máy mới

Copy **cả thư mục** `scripts/gpu-l40s-bench/` + `trace-round1.jsonl` + model lên `/root`:

```bash
scp -r scripts/gpu-l40s-bench/* trace-round1.jsonl <host>:/root/
```

Rồi `mkdir /root/ab` **trước** khi bench (thiếu dir → `tee` lỗi).

### Giá trị từng file

**Serve configs** (mỗi cái = 1 điểm cấu hình A/B):
| File | Nội dung |
|---|---|
| `serve_v12.sh` | **Baseline chuẩn** = config submit v12 (FP8 + FP8-KV + prefix-cache, `max-num-seqs=32`, `gmu=0.37`). Mọi A/B lấy đây làm gốc. |
| `serve_v20.sh` | v12 + `--max-num-batched-tokens=4096`. |
| `serve_v21_dp2.sh` | v12 + `batched-tokens=2048`, `gmu=0.185` (nhánh data-parallel — đã loại). |
| `serve_v22_spf.sh` | v12 + `--scheduling-policy=priority` (SPF, cần patch serving.py trước). |
| `serve_specdecode.sh` | v12 + MTP spec-decode `num_speculative_tokens=1`. |

**Patch runtime** (monkeypatch vLLM, chạy 1 lần sau khi cài):
| File | Nội dung |
|---|---|
| `patch_loggers.py` | Vá `loggers.py` để emit **REQSTAT per-request** (queued_time, prefill_time, decode_time, mean_tpot, num_cached_tokens). Nền tảng của mọi đo per-request. |
| `patch_serving_priority.py` | apply / revert / status patch SPF (priority = `len(prompt_token_ids)`) ở `chat_completion/serving.py`. |

**Replay + đo**:
| File | Nội dung |
|---|---|
| `replay_trace_detailed.py` | **Công cụ đo chính.** Bắn full 120 request đúng timestamp thật, ghi **per-request** TTFT/TPOT/user_id/turn_index/server_request_id + sampling CPU/GPU liên tục 0.5s (util, sm_clock, throttle%). |
| `replay_trace.py` | Bản gọn: chỉ TTFT, gom theo round (xem prefix-cache có hit turn 2-6 không). |
| `replay_per_round_metrics.py` | Snapshot `/metrics` sau khi mỗi round hoàn tất (queue/prefill sum chỉ update lúc request done). |
| `merge_request_metrics.py` | Join kết quả client với REQSTAT trong log server theo `server_request_id`. |
| `compare_ab.py` | So sánh median-of-3 per-turn giữa 2 nhánh A/B. |

**Driver A/B** (chạy trọn thí nghiệm, interleaved, cold mỗi rep):
| File | Nội dung |
|---|---|
| `run_ab.sh` | A/B **baseline vs SPF**, 3 rep, có `kill_server()` chuẩn (SIGTERM + poll VRAM). |
| `run_ab_spec.sh` | A/B **baseline vs spec-decode (MTP)**, kèm snapshot acceptance từ `/metrics`. |
| `bench_detailed.sh` | Wrapper 1-shot cho `replay_trace_detailed` + join REQSTAT. |
| `bench_with_diag.sh` | Chụp host state trước/sau bench (loadavg, throttle) để tương quan TTFT. |

**Setup + phụ trợ**:
| File | Nội dung |
|---|---|
| `setup_server.sh` | Chạy trên máy mới: tải model + apply `patch_loggers` + verify GPU sạch. Chạy đầu tiên. |
| `warmup.py` | Warm bằng nội dung **tự sinh** (không phải trace) để JIT kernel Mamba/GDN lúc khởi động — hợp lệ. |
| `warmup_shared_prefix.py` | **DIAGNOSTIC, không nộp được** — warm bằng nội dung trace (đo trần lợi ích prefix-cache). |

## 5. Dựng vLLM & check health

```bash
bash /root/serve_v12.sh
# đợi tới khi health = 200:
for i in $(seq 1 300); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health)" = "200" ] \
    && { echo "UP sau ${i}s"; break; }
  sleep 1
done
```
Health = 200 mới coi là server sẵn sàng. Xong bước này là bench được.
