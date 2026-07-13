# trace-round1.jsonl — Thống kê token THẬT

Đo bằng **tokenizer + chat template thật của Qwen3.5-2B** (`serve/models/qwen3.5-2b/`),
KHÔNG dùng proxy `cl100k_base` như tài liệu BTC. Tái tạo:
`.venv/bin/python3 scripts/20_trace_token_stats.py`. Cập nhật lần cuối: 2026-07-10.

> ⚠️ Vì sao con số ở đây khác tài liệu BTC: BTC đếm bằng tiktoken `cl100k_base`
> làm proxy (mean prefill ~18,900). Tokenizer THẬT của Qwen sinh **nhiều token
> hơn ~6-7%** cho cùng văn bản → mean thật **20,161**. Mọi tính toán ngân sách
> bộ nhớ/compute nên dùng số thật dưới đây, không dùng số proxy.

## 1. Cấu trúc workload — 20 session song song × 6 lượt (xác nhận 100%)

| Round | Request | Timestamp | Msg/session | Mean prefill (token thật) |
|---|---|---|---|---|
| 1 | 20 | 0 – 475 ms | 2 | 12,949 |
| 2 | 20 | 5000 – 5475 ms | 4 | 15,835 |
| 3 | 20 | 10000 – 10475 ms | 6 | 18,720 |
| 4 | 20 | 15000 – 15475 ms | 8 | 21,603 |
| 5 | 20 | 20000 – 20475 ms | 10 | 24,488 |
| 6 | 20 | 25000 – 25475 ms | 12 | 27,373 |

- Mỗi round là 1 **burst 20 request** dồn trong **475 ms** (IAT nội bộ = 25 ms).
- Khoảng nghỉ giữa các round: **chính xác 4525 ms** ("think time" của user).
- **Prefix continuity: 100/100 cặp khớp tuyệt đối** — request thứ i của round N
  là prefix đúng của request thứ i của round N+1. Đây là **20 hội thoại nối dài
  qua 6 lượt**, KHÔNG phải 120 request rời rạc.

## 2. Token in/out

| | Giá trị |
|---|---|
| Prefill min / max | 12,936 / 27,398 token |
| Prefill mean / median | **20,161** / 20,164 token |
| Output (`max_tokens`) | **200 cố định** cho cả 120 request |
| Tỷ lệ prefill : decode | **≈ 100.8 : 1** (BTC proxy ước ~95:1) |
| `temperature` / `seed` | 0 / 42 (cố định — greedy, tất định) |

Bài toán nghiêng **hoàn toàn** về phía xử lý context dài (prefill), decode chỉ
là phần rất nhỏ.

## 3. Ngân sách token prefill — không cache vs cache hoàn hảo

| Kịch bản | Token prefill |
|---|---|
| **KHÔNG cache** (tính lại từ đầu mỗi request) | **2,419,342** |
| Round 1 cold-start (không thể né) | 258,984 |
| Token MỚI round 2-6 (nếu cache hoàn hảo) | 288,471 |
| **Tổng nếu CACHE HOÀN HẢO** | **547,455** |
| **→ Cache hoàn hảo giảm compute** | **4.42×** |

**Hệ quả chiến lược:**
- Prefix caching giảm tổng compute **4.42 lần** → không phải tối ưu tùy chọn mà
  là **điều kiện tiên quyết**. Nhưng caching chỉ ăn cho phần KV của 6/24 layer
  full-attention; 18/24 layer GDN cần `--mamba-cache-mode` mới được cache (mặc
  định `none` = TẮT — xem [vllm-v0.22.1-flags-reference.md](vllm-v0.22.1-flags-reference.md)).
- Round 1 = **258,984 token cold-start dồn trong 475 ms** trên 1 lát MIG 18GB —
  không cache nào cứu được. Đây là "cuộc đua thông lượng prefill thuần túy" và
  là nơi các đội mạnh tách nhau.
- Mỗi lượt hội thoại thêm ~2,885 token mới/session (một message assistant có sẵn
  trong trace + một message user mới).

## 4. Liên hệ tới ngân sách bộ nhớ (đối chiếu `--max-model-len`)

Request dài nhất = **27,398 token prefill + 200 output ≈ 27,600 token**. Vậy
`--max-model-len=48000` (đang dùng ở v2+) **dư an toàn** cho cả request dài nhất
cộng đệm chat template — không cần nâng, và hạ xuống ~32768 cũng vẫn đủ (32768 >
27600) nếu cần thêm KV pool. Con số 262144 của baseline BTC là thừa thãi ~9.5×.
