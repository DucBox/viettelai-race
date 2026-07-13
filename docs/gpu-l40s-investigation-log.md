# Nhật ký điều tra trên GPU thật (L40S, Vast.ai) — 2026-07-11

> ⚠️ **Máy Vast.ai thuê theo phiên, KHÔNG cố định IP/port.** Mọi lệnh SSH
> (`ssh -p 9307 root@75.19.25.4`) trong file này chỉ đúng cho phiên thuê hôm
> 2026-07-11. Lần thuê sau, IP/port sẽ khác — script trong
> `scripts/gpu-l40s-bench/` vẫn dùng lại được, chỉ cần thay host/port khi scp/ssh.

## 1. Mục tiêu

Kết quả nộp bài dừng ở ~29.71 điểm (v12), trong khi có đội đạt ~90 điểm.
Thuê GPU thật (L40S, CUDA 13.0 — khớp `vllm==0.22.1` build) để:
đo cache hit thật, tìm root cause bằng log + metric thật, thử nghiệm các
lever mà trước đó chỉ đọc source code tĩnh (không GPU) không xác nhận được.

## 2. Setup

- **Máy:** Vast.ai, offer L40S 46GB VRAM, CUDA 13.2 max, $0.433/hr.
- **Template:** `vastai/vllm:v0.22.1-cuda-13.0` (Docker Options chỉ để
  `-p 8000:8000`, On-start Script để trống, Launch Mode = "Interactive shell
  server, SSH" để container không tự auto-serve).
- **Model:** tải `Qwen/Qwen3.5-2B` trực tiếp từ HuggingFace trên máy remote
  (nhanh hơn scp từ local). **Đã verify sha256 khớp 100%** với file
  `serve/models/qwen3.5-2b/model.safetensors-00001-of-00001.safetensors`
  local (`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`)
  → đảm bảo cùng revision với image đã build cho submission.
- **Mô phỏng ngân sách 18GB MIG H200** trên GPU 46GB: `--gpu-memory-utilization`
  = `0.95 × 18 / 46 ≈ 0.37`.
- **Xác nhận môi trường:** `vllm==0.22.1`, `torch 2.11.0+cu130`, CUDA 13.0 —
  khớp chính xác image `duc0811/qwen35-2b-race:v1` đã build cho submission.

## 3. Script đã viết (lưu tại `scripts/gpu-l40s-bench/`)

| File | Mục đích |
|---|---|
| `serve_v12.sh` | Serve model với đúng config v12 (best: fp8 weight+KV, flashinfer GDN backend) |
| `serve_v20.sh` | Biến thể `max-num-batched-tokens=4096` (test tăng batch size) |
| `serve_v21_dp2.sh` | Biến thể `--data-parallel-size=2` (test DP — **thất bại**, xem mục 6) |
| `replay_trace.py` | Bắn `data/trace-round1.jsonl` đúng timestamp thật, đo TTFT theo round |
| `replay_per_round_metrics.py` | Như trên nhưng snapshot `/metrics` Prometheus theo từng round để tách `queue_time`/`prefill_time` riêng biệt |
| `warmup.py` | Warmup bằng nội dung **tự sinh** (không dùng trace, đúng luật chống gian lận) để test giả thuyết JIT |
| `bench_with_diag.sh` | Wrapper: chụp `cpu.stat` (throttle) + `nvidia-smi` trước/sau bench để đối chiếu nhiễu host |
| `replay_trace_detailed.py` | **(mới)** Log TỪNG request riêng lẻ (không chỉ mean/median theo round) + sample CPU/GPU LIÊN TỤC mỗi 0.5s trong suốt phiên bench (không chỉ before/after), rồi join theo timestamp để biết chính xác tại thời điểm 1 request bị TTFT cao thì GPU util/sm_clock/CPU-throttle đang ở mức nào |
| `bench_detailed.sh` | Wrapper gọi `replay_trace_detailed.py`, ghi 2 file JSON (`replay_detailed_requests.json`, `replay_detailed_samples.json`) + bảng in console sắp xếp theo TTFT giảm dần |

**Ghi chú phương pháp (áp dụng từ lần bench tiếp theo trở đi):** `bench_with_diag.sh`
(before/after) chỉ cho biết trạng thái trung bình toàn phiên — không đủ để
kết luận vì có thể có throttle/spike ngắn bị pha loãng khi lấy trung bình.
Từ nay dùng `bench_detailed.sh` làm mặc định: mọi lần bench đều phải kèm
(1) trạng thái GPU/CPU tại đúng thời điểm từng request, và (2) số liệu chi
tiết từng request thay vì chỉ tổng hợp overall theo round.

## 4. Chuỗi thực nghiệm và phát hiện (theo đúng thứ tự đã làm)

### 4.1. Đo TTFT theo round, lặp lại 3 lần (kiểm tra độ ổn định)

| Round | Run1 | Run2 | Run3 | CV (độ lệch chuẩn/trung bình) |
|---|---|---|---|---|
| 1 (cold) | 4013ms | 2590ms | 2057ms | **35.0%** |
| 2 | 1981ms | 794ms | 665ms | **63.2%** |
| 3 | 999ms | 914ms | 937ms | 4.6% |
| 4 | 931ms | 952ms | 963ms | 1.7% |
| 5 | 963ms | 906ms | 870ms | 5.1% |
| 6 | 1356ms | 1158ms | 1197ms | 8.5% |

**Kết luận:** round 3-6 cực ổn định (tin được). Round 1-2 dao động rất mạnh
và **giảm dần đơn điệu qua từng lần restart** — cần điều tra tiếp, không
được vội kết luận "config X tốt/xấu hơn" chỉ từ 1 lần đo round 1.

### 4.2. Loại trừ giả thuyết GPU clock chưa "nóng"

`nvidia-smi --query-gpu=clocks.sm,pstate` lúc idle: **2520MHz, P0** — đã max
sẵn ngay cả khi không tải. Không phải nguyên nhân.

### 4.3. Loại trừ giả thuyết JIT compile Triton (bằng thực nghiệm, không suy đoán)

Log tự vLLM cảnh báo:
```
WARNING: Triton kernel JIT compilation during inference: _zero_kv_blocks_kernel.
This causes a latency spike; consider extending warmup to cover this shape/config.
```
5 kernel Mamba/GDN bị compile sống lúc round 1. **Thử nghiệm:** chạy
`warmup.py` (nội dung tự sinh, đa dạng độ dài 13k-27k token) trước khi bắn
trace → xác nhận 0 sự kiện JIT mới trong lúc replay (loại bỏ hoàn toàn) —
nhưng **round 1 TTFT không cải thiện** (2033ms vs 2057ms không-warmup).
**→ JIT có thật nhưng không phải nguyên nhân chính.**

### 4.4. Tìm ra root cause thật qua Prometheus `/metrics`

```
Queue time trung bình (120 request): 587-846ms
Prefill compute trung bình:          206-288ms
```
**Queue time > prefill time 2.5-2.9 lần** → vấn đề là XẾP HÀNG (admission),
không phải compute chậm. Khớp với: `max_num_partial_prefills=1` (mặc định,
không đổi được — đổi sẽ crash "Concurrent Partial Prefill is not supported"
đúng như đã gặp ở v6/v15/v19 khi nộp thật).

### 4.5. Thử tăng `max-num-batched-tokens` 2048→4096 — BỊ BÁC BỎ

Giả thuyết: mỗi request xong prefill nhanh hơn → nhường "chỗ" sớm hơn cho
request tiếp theo trong hàng đợi. **Đo được:** `queue_time` **846ms (tệ hơn
1.44x)**, không phải tốt hơn. Lý do suy luận: tổng compute cho 1 request
không đổi dù chia chunk to hay nhỏ — không giải quyết gốc rễ (P=1).

### 4.6. Loại trừ giả thuyết CPU throttle bởi host chia sẻ (bằng cgroup accounting thật)

Phát hiện máy có dấu hiệu multi-tenant: `cpuset.cpus.effective=0-127` (không
gán core cố định, chỉ quota nổi ~30.7 core-tương-đương trên 128 core host),
CPU model thực tế (AMD EPYC 9534) khác với offer ban đầu ghi (Xeon
SapphireRapids), load average nền 11-16 dù process ta chỉ dùng ~30% 1 core.

`cgroup cpu.stat` xác nhận **có** bị throttle tích lũy (27% throttle/usage
ratio từ lúc container khởi động). Nhưng khi đo **delta throttle đúng trong
cửa sổ 1 lần bench cụ thể** (dùng `bench_with_diag.sh`): **throttle = 0%**
trong khi round 1 vẫn cao (2499ms cold-start thật). **→ Throttle không giải
thích được cho lần đo cold-start cụ thể này** — loại trừ như nguyên nhân
trực tiếp (dù vẫn là rủi ro nền cần cảnh giác cho các so sánh khác).

⚠️ Lưu ý: có 1 lần đo cho kết quả round1=190ms — **đây là lỗi phương pháp**,
do chạy lại đúng trace trên server CHƯA restart (cache hit 100% từ chính
lần chạy trước, không phải cold start thật) — không đại diện cho bài thi
thật (trace chỉ bắn 1 lần vào server sạch).

### 4.7. Per-round breakdown queue_time/prefill_time — PHÁT HIỆN QUYẾT ĐỊNH

Sau khi sửa 1 bug đo lường (metric `queue_time`/`prefill_time` được vLLM ghi
nhận lúc request **hoàn tất toàn bộ**, không phải lúc đạt TTFT — snapshot
phải căn đúng thời điểm này):

| Round | queue (ms) | prefill (ms) | tỷ lệ queue/prefill |
|---|---|---|---|
| **1** | **1621** | 335 | **4.84x** |
| 2 | 381 | 191 | 1.99x |
| 3 | 308 | 248 | 1.24x |
| 4 | 231 | 233 | 0.99x |
| 5 | 132 | 256 | 0.52x |
| 6 | 288 | 276 | 1.05x |

**Tỷ lệ queue/prefill giảm dần đều từ round 1 (4.84x) → round 4-5 (~1x hoặc
thấp hơn).** Xác nhận: gate "1 prefill tại 1 thời điểm" gây hại **nặng nhất
ở round 1** (20 session hoàn toàn mới phải xếp hàng admit tuần tự); ở
round 2-6 (session tiếp diễn từ trước) overhead admit nhẹ hơn nhiều.

### 4.8. Tra cứu GitHub — xác nhận đây là giới hạn đã biết, đang mở (dùng subagent research)

- **Issue #14003** "[Feature]: Implement Concurrent Partial Prefills In V1
  Engine" — mở từ 2025-02-28, **vẫn đang mở** tới 2026-05-29. Một contributor
  có branch thử nghiệm cho thấy TTFT giảm 721ms→470ms (~35%) nhưng
  **chưa merge**.
- **PR #21651** (thử support `max_long_partial_prefills` cho V1) — **đóng,
  không merge**.
- **Chưa version vLLM nào (tính đến 2026-07-11) gỡ bỏ giới hạn này.**
- Semantics metric `queue_time`/`prefill_time` được xác nhận sạch (không có
  bug double-counting trong V1) → kết luận "queue-bound" đứng vững, không
  phải artifact đo lường.
- Lever hợp pháp tìm được: `--data-parallel-size` (N replica song song, mỗi
  replica vẫn P=1 nhưng N replica cho hiệu ứng N-song-song thật).

### 4.9. Thử `--data-parallel-size=2` — THẤT BẠI, lý do cấu trúc

```
AssertionError: DP adjusted local rank 1 is out of bounds.
assert self.local_rank < torch.accelerator.device_count()
```

**Nguyên nhân:** `--data-parallel-size` của vLLM yêu cầu **1 GPU vật lý
riêng cho mỗi replica** (replica thứ N cần device index N-1). Máy test chỉ
có 1 GPU → assertion fail ngay khi replica thứ 2 khởi tạo.

**Hệ quả quan trọng:** môi trường chấm thật là **"1 instance MIG H200"**
— cũng chỉ 1 GPU (lát cắt) duy nhất. **Lever này KHÔNG áp dụng được cho
bài thi**, dù chạy tốt trên máy nhiều GPU. Đây là giới hạn cấu trúc, không
phải điều có thể lách qua bằng cấu hình khác trong vLLM.

## 5. Tổng kết — mọi lever trong phạm vi vLLM 0.22.1 / 1-GPU-slice đã cạn

| Hướng đã thử | Kết quả |
|---|---|
| Tăng `max-num-batched-tokens` | ❌ Bị bác bỏ bằng đo đạc (queue_time tệ hơn) |
| Warmup tự sinh (loại JIT) | ❌ Không cải thiện TTFT dù loại bỏ JIT hoàn toàn |
| `--data-parallel-size=2` | ❌ Không áp dụng được (yêu cầu N-GPU, ta chỉ có 1) |
| Xác minh CPU throttle | ❌ Loại trừ — không tương quan với 1 lần đo cụ thể |
| Xác minh prefix caching | ✅ Hoạt động tốt (~80% hit, gần sát lý thuyết) — không phải vấn đề |

**Root cause ban đầu (session 1), sau đó bị ĐÍNH CHÍNH ở session 2 (xem mục
6.1):** lúc này nghi ngờ gate cứng `max_num_partial_prefills=1` của vLLM
0.22.1 V1 engine (Issue #14003, vẫn mở) là nguyên nhân — gây queue_time gấp
4.84x prefill_time ở round 1. **Kết luận này SAI về cơ chế** (dù đúng về
triệu chứng: queue-bound). Xem mục 6.1 để biết cơ chế thật.

## 6. Hướng chưa thử — cần làm tiếp

- Sweep `--long-prefill-token-threshold` nhỏ hơn (giảm chunk mỗi request,
  giảm mức độ 1 request "chiếm chỗ" — khác cơ chế với việc tăng batch-tokens
  đã thử và thất bại).
- Đo lại toàn bộ trên máy khác (không multi-tenant) nếu có điều kiện, để
  loại hoàn toàn biến số nhiễu host khỏi các so sánh tinh vi sau này.

## 6.1. Session 2 (GPU L40S mới, đọc source trực tiếp) — ĐÍNH CHÍNH root cause + fix thật (SPF)

**Đính chính quan trọng:** đọc trực tiếp source `vllm/v1/core/sched/scheduler.py`
bên trong image xác nhận `max_num_partial_prefills`/`max_long_partial_prefills`
**là dead code trong V1** — `grep` toàn bộ package chỉ thấy 2 field này ở
`arg_utils.py` (chỗ reject CLI) và `config/scheduler.py` (chỗ validate), **không
nơi nào trong scheduler thật dùng chúng để giới hạn concurrency**. Vòng lặp
admission thật (`scheduler.py:548`):
```python
while (self.waiting or self.skipped_waiting) and token_budget > 0:
```
đã admit **nhiều request mới cùng lúc mỗi step**, miễn còn `token_budget`
(=`max_num_batched_tokens`). Vậy Issue #14003 không phải nguyên nhân — nó nói
về 1 feature V0 cũ, không áp dụng cho cơ chế thật của V1.

**Cơ chế thật:** `policy="fcfs"` (mặc định) — 20 request mới cùng lúc, mỗi
request ~13k-27k token prefill, tổng nhu cầu vượt xa token budget/step →
request đến sau trong FCFS phải chờ nhiều step mới được cấp token đầu tiên.
Đây thuần là bài toán **xếp hàng (queueing)**, không phải KV cache hay prefill
compute nữa (2 cái đó đã xác nhận ổn ở mục 4.7/6.2).

**Fix thử: Shortest-Prefill-First (SPF) qua priority scheduling.**
vLLM có sẵn `policy="priority"` (`--scheduling-policy=priority`) + field
`priority` trong OpenAI request schema (`chat_completion/protocol.py`, mặc
định 0 — vô dụng nếu không set). Vì grading harness gửi trace cố định,
không có field `priority`, nên **patch server-side** 1 dòng tại
`chat_completion/serving.py` (được gọi ngay trước `engine_client.generate`):
```python
# truoc:
priority=request.priority,
# sau (patch da ap dung va test):
priority=(len(prompt_token_ids) if prompt_token_ids else request.priority),
```
Prompt càng ngắn → priority số càng nhỏ → được xử lý trước (đúng cơ chế
priority queue: số nhỏ hơn = ưu tiên hơn, tie-break bằng `arrival_time`).
Không đụng đến nội dung/timestamp/concurrency-config mà client gửi — chỉ đổi
cách server tự sắp thứ tự nội bộ.

**Kết quả đo (cold run, patch server-side + `--scheduling-policy=priority`,
so với FCFS baseline v12 cold run ở mục 4.7):**

> ⛔ **BẢNG DƯỚI ĐÂY ĐÃ BỊ BÁC BỎ — xem mục 6.3.** Nó đo 2 cold run ở **2 phiên
> khác nhau**, mà round 1-2 có **CV 35-63%** (docs mục 4.1) → phần lớn "-48%/-66%"
> là **nhiễu cold-start**, không phải hiệu lực SPF. A/B sạch median-of-3 cùng phiên
> (mục 6.3) cho kết quả rất khác: SPF **không giảm tổng chờ**, chỉ phân phối lại.

| Round | FCFS (baseline) mean TTFT | SPF (patch) mean TTFT | Δ |
|---|---|---|---|
| 1 | 4055.8ms | **2096.2ms** | **-48%** |
| 2 | 2049.9ms | **687.8ms** | **-66%** |
| 3 | 1095.9ms | 928.8ms | -15% |
| 4 | 957.7ms | 960.5ms | ~0% |
| 5 | 862.6ms | 893.4ms | +4% |
| 6 | 1205.4ms | 1167.6ms | -3% |

**Việc còn lại trước khi đưa vào submission thật:**
- Đóng gói patch vào Dockerfile (bake sẵn file `serving.py` đã sửa vào image
  `duc0811/qwen35-2b-race:v1`, không patch tay trên máy chấm).
- Kiểm tra lại câu chữ luật thi "không đổi concurrency config" — về bản chất
  đây là đổi thứ tự xử lý hàng đợi nội bộ server, không đổi nội dung/timestamp
  trace, nhưng nên soát kỹ trước khi nộp thật.
- Cân nhắc dùng `len(prompt_token_ids)` hay 1 công thức khác (VD trừ bớt cho
  request đã có nhiều prefix cache hit, để tránh phạt oan request dài nhưng
  phần lớn đã cache) — hướng tinh chỉnh thêm nếu còn thời gian.

## 6.2. Session 2 — verify lại các nghi vấn cũ trên GPU/model thật

Dùng script `bench_detailed.sh` (per-request + continuous system sampling)
trên phiên thuê GPU L40S mới, xác nhận lại bằng log thật (không chỉ suy luận):
- `config.py:355`: *"Mamba cache mode is set to 'align' for
  Qwen3_5ForConditionalGeneration by default"* — xác nhận **không phải
  `none`** như argparse-default tĩnh, độc lập kiến trúc GPU (hardcode theo
  tên model trong vLLM, nên chắc chắn cũng xảy ra trên H200 thật).
- `qwen_gdn_linear_attn.py:299`: *"GDN prefill backend 'flashinfer' is
  selected but cannot use this kernel on the current platform. Falling back
  to Triton/FLA"* — xác nhận fallback thật xảy ra trên **L40S (Ada)**. Chưa
  chắc xảy ra trên **H200 (Hopper)** — cần test trên máy thi đấu thật hoặc
  submit thật mới biết chắc, đừng suy rộng kết luận này sang H200.
- `vllm.py:977`: *"Asynchronous scheduling is enabled"* — xác nhận tự bật.
- `/metrics` live: prefix_cache hit rate **74.2%** (6.49M/8.75M) — cache hoạt
  động tốt, không phải "tính lại gần như toàn bộ lịch sử". CPU throttle 0.0%
  suốt timeline, GPU util 95-100%, SM clock 2.3-2.5GHz ổn định — loại trừ
  hoàn toàn giả thuyết host contention/CPU throttle.

## 6.3. Session 3 (L40S mới, GPU SẠCH) — A/B controlled median-of-3, ĐÍNH CHÍNH hiệu lực SPF

Chạy lại A/B **đúng chuẩn khoa học** trên instance L40S mới (`vllm/vllm-openai:v0.22.1`),
sửa hết các sai sót phương pháp của các lần trước:

- **Baseline đúng = v12** (khác v22 ĐÚNG 1 biến = `--scheduling-policy` + patch
  `priority=len(prompt_token_ids)`). KHÔNG dùng `serve_v21_dp2.sh` (khác 3 biến:
  DP2, gpu-mem 0.185, batched-tokens 2048 → confound; và DP2 fail trên 1 GPU).
- **Cùng phiên, median-of-3, interleaved** (baseline1, spf1, baseline2, spf2, …)
  để loại nhiễu cold-start (round 0-1 CV 35-63%) và host-drift theo thời gian.
- **Mỗi rep = server cold mới** (kill graceful → serve lại → bắn trace 1 lần), đúng
  ngữ cảnh bài thi, không cache carryover.
- **6 rep footprint đồng nhất tuyệt đối: `mem_used=17418 MiB` mọi lần** (fair),
  120/120 request join đủ server-side stats mỗi rep.
- Bài học vận hành: **kill vLLM phải SIGTERM (graceful)**; `pkill -9` để lại CUDA
  context mồ côi không trả VRAM → server sau chồng lên → OOM. Và một `-9` lên
  vLLM đang crash có thể leak ~9GB không thu hồi được trong container (không có
  quyền `gpu-reset`) → phải thuê instance mới.

**Kết quả A/B sạch (gộp 360 request/bên):**

| stat | QUEUE base | QUEUE SPF | Δ | TTFT base | TTFT SPF | Δ |
|---|---|---|---|---|---|---|
| mean | 610 | 628 | **+2.9%** | 1177 | 1144 | -2.8% |
| p50 | 407 | 293 | **-28%** | 1056 | 976 | **-7.6%** |
| p75 | 795 | 964 | **+21%** | 1563 | 1544 | -1.2% |
| p90 | 1466 | 1531 | +4.5% | 2149 | 2156 | +0.3% |
| p95 | 2380 | 2452 | +3.0% | 2825 | 2810 | -0.5% |
| p99 | 3192 | 3381 | +5.9% | 3588 | 3718 | +3.6% |
| max | 3369 | 3567 | +5.9% | 3757 | 3939 | +4.8% |

- `% request queue<50ms` (nhảy hàng gần tức thì): baseline **15.8%** → SPF **23.6%**.
- `prefill` và `tpot`: giống hệt (Δ<1%) — SPF không đụng compute (đúng kỳ vọng).

**Kết luận đính chính (thay cho "-48%/-66%" ở mục 6.1):** SPF làm **đúng lý thuyết
SJF — phân phối lại độ trễ, KHÔNG giảm tổng.** Median tốt hơn rõ (queue p50 -28%,
TTFT p50 -7.6%, nhiều request nhảy hàng hơn), NHƯNG **mean gần như không đổi
(+2.9% queue)** và **p75 + đuôi (p95/p99/max) hơi TỆ hơn** (+3-6%, request dài bị
đẩy lùi). Đây là **đánh đổi có điều kiện**, không phải cải thiện thuần.

→ **SPF có đáng nộp hay không phụ thuộc hàm chấm điểm:** chấm theo mean/tổng →
~hòa; theo median/p50 → lợi ~8% TTFT; phạt đuôi p95/p99 → hơi hại. Cần xác định
metric của grader trước khi quyết định bake patch vào image submission.

Toolkit tái lập: `scripts/gpu-l40s-bench/{setup_server.sh, run_ab.sh,
patch_loggers.py, patch_serving_priority.py, replay_trace_detailed.py,
merge_request_metrics.py, compare_ab.py}`.

## 6.4. Session 3 (tiếp) — PHÁT HIỆN QUYẾT ĐỊNH: công thức chấm điểm + đính chính "tối ưu sai trục"

Đọc `scripts/08_ers_score.py` (tái tạo công thức BTC round 1 từ PDF §1.3):

```
Score = 100 × ERS × f(Δ)                       (f(Δ) = accuracy, tính riêng)
ERS   = mean 120 request của:  0.5·s_ttft + 0.5·s_tpot   (=0 nếu request fail)
s     = clamp((C − x)/(C − F), 0, 1)²           ← LIÊN TỤC, γ=2 (dốc, gần Floor điểm bình phương)
   TTFT: F=100ms,  C=1500ms      (cửa sổ rộng 15×)
   TPOT: F=20ms,   C=45ms        (cửa sổ hẹp 2.25×)  ← TPOT = mean inter-token latency/request
```

**Điểm KHÔNG phải đếm SLO pass** — mà là **càng gần Floor càng nhiều điểm theo bình phương**.

**Verify ngược với `submit/results.csv` (điểm dashboard thật) — công thức khớp 100%:**

| version | TPOT | s_tpot | TTFT p50 | s_ttft≈ | ERS dự đoán | Score thật |
|---|---|---|---|---|---|---|
| baseline | 58ms | 0 (vỡ trần 45) | 652 | ~0.30 | ~0.15 | 15.78 ✓ |
| v2 | 26ms | 0.52 | 1941 | ~0.06 | ~0.29 | 29.18 ✓ |
| v12 | 28ms | 0.46 | 1366 | ~0.13 | ~0.30 | 29.71 ✓ |

**Cú sốc — L40S đã LỪA chúng ta.** Tính ERS thật trên data A/B L40S sạch:
`baseline ERS=0.613 (~61đ), spf ERS=0.631 (~63đ)`. Nhưng H200 thật chỉ ~30đ.
**Toàn bộ chênh lệch 61→30 nằm ở TPOT:** L40S TPOT=13ms → s_tpot=**1.0** (kịch Floor);
H200 thật TPOT=28ms → s_tpot=**0.46**. Nửa TPOT rớt 0.50→0.23 → kéo score 61→30.
L40S decode quá nhanh nên TPOT maxed, che mất chỗ mất điểm thật.

**Trên H200 thật, cả 2 nửa mới đi ~nửa đường:** s_ttft≈0.13 (TTFT 1366 sát Ceiling 1500 → ~0đ),
s_tpot≈0.46 (TPOT 28 còn xa Floor 20). Đội 90–100đ ép **CẢ HAI về sát Floor**: cần
TTFT~172ms + TPOT~21ms trên mọi request để ERS≈0.9.

**ĐÍNH CHÍNH chiến lược — ta đã tối ưu SAI TRỤC:**
- **SPF/queue chỉ nhích ERS 0.613→0.631 (+0.02) = hạt cát.** Vì TTFT 1366 sát Ceiling;
  giảm 8% (→1260) vẫn ~0đ. **TTFT phải giảm BẬC ĐỘ LỚN (~8×, 1366→~300) mới ăn điểm**,
  không phải vài %.
- **Baseline→v2 (15→29) TOÀN BỘ là trục TPOT** (FP8 KV: tpot 58→26, s_tpot 0→0.52). Xác nhận
  TPOT là trục kiếm điểm lớn nhất lịch sử. Mọi thứ v10–v22 sau đó (~27-30) chỉ vọc TTFT/kernel
  ở rìa → không phá được vì TPOT kẹt ~28ms và TTFT kẹt sát ceiling.

**Độ nhạy TPOT (nơi bỏ lỡ nhiều điểm nhất, γ=2 nên mỗi ms đáng ~2.5-3đ gần floor):**
28ms→s_tpot 0.46 | 24ms→0.71 (+6đ) | 22ms→0.85 (+10đ) | 20ms→1.00 (+13đ).

**Hai lever thật (mỗi cái ~25-30đ), thay cho SPF:**
1. **TPOT 28→20ms (+~27đ)** — bandwidth-bound; cách DUY NHẤT phá sàn = sinh >1 token/lần đọc
   weight = **speculative decoding**. Qwen3.5-2B có sẵn MTP head. v5/v8 fail vì **mistuned**
   (acceptance thấp → overhead > lợi). Đáng đào lại nghiêm túc — lever đúng lý thuyết để xuống
   dưới floor 27ms mà docs cũ tưởng bất khả.
2. **TTFT 1366→~300ms (+~30đ)** — queue-bound; cần loại gần hết queue burst (giảm bậc độ lớn).
   Đây mới là chỗ queue/scheduling thuộc về, nhưng SPF quá nhỏ so với mức cần.

## 6.5. Session 3 (tiếp) — spec-decode MTP: ĐO XONG, CHẾT vì cấu trúc GDN

Đào thẳng vào code + bench thật để chốt lever TPOT (spec-decode) đề xuất ở mục 6.4:

- **Model có đúng 1 MTP layer** (`mtp.layers.0`, `n_predict=1`). vLLM cho `num_speculative_tokens>1`
  bằng cách chạy autoregressive lại 1 layer đó, NHƯNG warn "lower acceptance". → `tokens=1` là
  điểm ngọt. v5/v8 dùng tokens=1 là **đúng số** (không phải bug ở đây).
- **MTP tương thích** model (chỉ cần `mamba-cache-mode=align`, đã là default). Không rào cản cứng.
- **A/B chuẩn (L40S, median-of-3, interleaved, cold mỗi rep):** acceptance **67.3%** (ổn định cả
  3 rep, →1.67 token/step) NHƯNG:

| metric (L40S) | baseline | +MTP spec | Δ |
|---|---|---|---|
| TPOT | 13.4ms | 19.1ms | +42.6% |
| TTFT | 1158ms | 2340ms | **+102%** |
| prefill | 189ms | 262ms | +38.8% |
| **ERS(L40S)** | **0.615** | **0.481** | **−22%** |

- overhead draft+verify = **18.6ms/step** (median-of-3, chính xác hơn 31ms của 1-run đơn lẻ).
- **Suy ra H200 (base 28ms):** TPOT ~27.8ms = **~break-even** (overhead cân bằng đúng lợi 1.67×) —
  KHÔNG âm nặng như 1-run đoán. NHƯNG **TTFT +102%** → trên H200 1366→~2700ms **vượt ceiling
  1500 → s_ttft sụp ~0** → mất −6 đến −13đ. **Net LOSS qua trục TTFT.** Khớp v8 (score 3.14).
- **Giải mã v8: KHÔNG phải acceptance (67% tốt), mà là overhead + phá TTFT** (spec làm mỗi
  decode-step nặng hơn → giữ GPU lâu hơn → queue burst đội gấp đôi trên workload queue-bound).

**Nguyên nhân GỐC (cấu trúc, không sửa được bằng tune):** spec-decode cần bước *verify* (target
chạy trên token đã draft). Transformer: verify ~free (batch 1 forward). **GDN/Mamba
(recurrent/linear-attn): mỗi position phải xử lý TUẦN TỰ qua recurrent state + rollback khi reject
→ ~2× forward/step, bất khả tránh, độc lập drafter** (ngram free-draft vẫn trả verify cost). →
**Speculative decoding về bản chất không hợp linear-attention.** Lever TPOT "+27đ" ở mục 6.4 CHẾT.

**Hệ quả:** TPOT thực tế đã gần kịch trần khai thác (~26-28ms trên H200, floor 20ms không với
tới được bằng lever nào đã biết). Cả 2 trục điểm (TTFT-queue, TPOT-decode) đều gần trần trong
ràng buộc 1-slice + vLLM 0.22.1 + kiến trúc GDN. Điểm ~30 có thể gần trần thực tế của hướng này;
đội 90-100 nhiều khả năng dùng cách tiếp cận khác hẳn (chưa xác định được, không có H200 để test).

## 7. Cách dùng lại khi thuê GPU mới

**Lưu ý quan trọng — dùng tmux, đừng SSH lặp lại nhiều lần:** SSH nhiều lần
liên tục (mỗi lệnh 1 connection mới) hay bị lỗi connect. Từ nay: SSH vào
**1 lần duy nhất**, mở `tmux new -s bench` (hoặc `tmux attach -t bench` nếu
session đã tồn tại), và chạy MỌI THỨ (serve, warmup, bench) bên trong tmux
đó. Các lần sau chỉ cần `ssh -p <PORT> root@<IP> -t tmux attach -t bench`
để nối lại đúng phiên đang chạy, không mất trạng thái, không cần lặp lại
setup, và server vẫn sống nếu SSH bị rớt giữa chừng.

```bash
# 1. Thuê L40S (hoặc GPU CUDA>=13 tương đương) trên Vast.ai, image
#    vastai/vllm:v0.22.1-cuda-13.0, Docker Options chỉ "-p 8000:8000",
#    On-start Script để trống, Launch Mode = Interactive shell + SSH.
# 2. Lấy lệnh SSH mới (IP/port đổi mỗi lần thuê), thay vào các lệnh dưới.

scp -P <PORT> scripts/gpu-l40s-bench/*.sh scripts/gpu-l40s-bench/*.py root@<IP>:/root/
ssh -p <PORT> root@<IP> "python3 -c 'from huggingface_hub import snapshot_download; \
  snapshot_download(repo_id=\"Qwen/Qwen3.5-2B\", local_dir=\"/root/model\")'"
scp -P <PORT> data/trace-round1.jsonl root@<IP>:/root/trace-round1.jsonl

# 3. Mo 1 SSH duy nhat, tao tmux session, lam moi thu ben trong no:
ssh -p <PORT> root@<IP> -t "tmux new -s bench -A"
#   (trong tmux): bash /root/serve_v12.sh
#   (trong tmux, doi health check 200 roi): cd /root && bash bench_detailed.sh
# Cac lan sau chi can:
ssh -p <PORT> root@<IP> -t "tmux attach -t bench"
```
