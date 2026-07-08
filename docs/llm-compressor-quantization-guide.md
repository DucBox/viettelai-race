# llm-compressor — quy trình quantize FP8 cho Qwen3.5-2B

Tài liệu tham chiếu cho bước **quantize model** (tách khỏi tối ưu cờ vLLM, xem
`docs/vllm-optimization-guide.md`). Gồm 2 phần: (A) tóm tắt quy trình 5 bước
chính thức của llm-compressor áp luôn vào case của mình, (B) rủi ro
version/dependency đã điều tra thật (không phải suy đoán) giữa llm-compressor
↔ compressed-tensors ↔ vLLM v0.22.1.

---

## A. Quy trình 5 bước chính thức (docs.vllm.ai/projects/llm-compressor)

> Thứ tự bắt buộc: **Choose model → Choose scheme → Choose algorithm → Choose
> dataset (nếu cần) → Compress**.

### 1. Choose your model
llm-compressor hỗ trợ chính thức decoder-only, VLM, MoE — **không có rào cản
chính thức nào với kiến trúc hybrid GDN** của mình, chỉ cần load đúng class.
Model mình là VLM thật (`Qwen3_5ForConditionalGeneration`) → loader đã verify
ở `scripts/06_module_tree.py`: **`AutoModelForImageTextToText`** (thử trước
`AutoModelForCausalLM`/`AutoModel`), cần transformers-from-source (main
branch) vì `qwen3_5` chưa có ở bản stable nào.

### 2. Choose your compression scheme — **chốt: W8A8-FP8**
Bảng hardware/scheme chính thức (rút gọn phần liên quan tới GPU mình có —
Hopper/Ada; bỏ Blackwell vì không có phần cứng đó):

| GPU | Compute cap. tối thiểu | Scheme khuyến nghị | Ghi chú |
|---|---|---|---|
| **Ada (L40S — máy test)** | **8.9** | **W8A8-FP8** | max throughput |
| **Hopper (H200 — máy chấm thi)** | 9.0 | W8A8-FP8 (khuyến nghị chính), hoặc W4AFP8 | |
| Blackwell (SM100) | 10.0 | NVFP4/MXFP4/MXFP8 | **không áp dụng — không có phần cứng này** |

→ **NVFP4/MXFP4/MXFP8 loại khỏi cân nhắc hoàn toàn** (cần Blackwell, mình chỉ
có Ada/Hopper). Trong dải phần cứng mình có, **W8A8-FP8 là lựa chọn nén cao
nhất khả dụng** — không phải mình tự chọn tối ưu, đây là trần công nghệ thật.

Guide cũng phân loại theo **kịch bản dùng** (đã nói ở lượt trước, nhắc lại vì
quan trọng): *online* (latency-bound, input nhỏ) → weight-only (W4A16); còn
*offline/compute-bound* (GPU luôn bận) → **W8A8**. Bài mình prefill-bound
thật sự → khớp đúng nhánh W8A8.

Format lưu đĩa (bảng compressor của compressed-tensors) — quan trọng để hiểu
checkpoint thực chất mang tên gì: **W8A8-float (FP8) → compressor
`float_quantized`**.

### 3. Choose your compression algorithm — **chốt: RTN (chính là FP8_DYNAMIC)**
Bảng thuật toán chính thức:

| Thuật toán | Cần dataset? | Khi nào dùng | Accuracy recovery |
|---|---|---|---|
| **RTN** | **Không** (data-free) | nhanh, đơn giản, real-time | "moderate... **good recovery riêng cho FP8/FP4**" |
| SmoothQuant | Có | cân bằng W8A8, xử lý outlier | tốt, calib nhanh |
| AWQ | Có | general purpose | cao nhưng đắt |
| GPTQ | Có | broad compat | cao nhưng đắt |
| AutoRound | Có | broad compat | cao nhưng đắt |

`FP8_DYNAMIC` (weight per-channel static + activation per-token dynamic) mà
mình đang dùng **chính là biến thể RTN cho FP8** — data-free, và guide xác
nhận thẳng RTN "good recovery" riêng cho FP8 (không phải fallback yếu, là lựa
chọn hợp lý cho đúng format này). AWQ/GPTQ/SmoothQuant chủ yếu đáng giá ở
INT4/INT8 bit thấp — không phải bài toán của mình.

### 4. Choose your dataset — **bỏ qua bước này** (vì chọn RTN/data-free)
Chỉ cần dataset nếu dùng AWQ/GPTQ/SmoothQuant/AutoRound, hoặc scheme có
activation quantize tĩnh (NVFP4, static-per-tensor). `FP8_DYNAMIC` không rơi
vào nhóm này → **không cần chuẩn bị calibration dataset cho bản chính**.

**Lever dự phòng** (nếu sau này GPQA sát ngưỡng 0.30): chuyển 1 phần sang
GPTQ/AWQ có calibrate, **128–256 sample** là đủ (guide: "diminishing returns"
sau vài trăm sample). Dataset gợi ý: `ultrachat-200k` (tổng quát) hoặc tự
soạn sample kiểu Q&A gần miền GPQA — **không được dùng nội dung trace thật**
(luật thi cấm pre-compute).

### 5. Compress — mixed-precision là pattern CHÍNH THỐNG, không phải hack
Guide có hẳn mục "Mixed-precision quantization for accuracy recovery": áp
scheme khác nhau cho layer khác nhau trong cùng 1 lần chạy, ví dụ *"Preserve
sensitive layers (attention blocks...) at FP8"* trong khi phần khác dùng
precision khác. → **Ignore-list của mình (giữ nguyên `linear_attn.*`, quantize
phần còn lại) là đúng pattern được support chính thức**, không phải workaround.

Recipe mẫu chính thức (tham số `oneshot` tổng quát, model MoE Qwen3-30B-A3B,
khác model mình nhưng cấu trúc lệnh giữ nguyên):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.offload import dispatch_model

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",          # bài mình dùng DYNAMIC, không phải BLOCK
    ignore=["lm_head", "re:.*linear_attn.*", "re:.*visual.*", "re:.*mtp.*"],
)
oneshot(model=model, recipe=recipe)

# >>> BƯỚC HAY BỊ BỎ QUÊN: sanity-check generate trước khi tin/save <<<
dispatch_model(model)                      # compressed_tensors.offload — cần cho model đã bị oneshot xử lý
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))          # câu vô nghĩa/lặp => quantize hỏng, đừng save

model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

Model mình dùng `AutoModelForImageTextToText` thay vì `AutoModelForCausalLM`
(khác ví dụ text-thuần) — xem mục A.1.

**Prerequisites chính thức:** Linux, Python ≥3.10, GPU khuyến nghị — khớp
100% với hạ tầng đã dựng (K8s pod, native serve, Python 3.11).

---

## B. Rủi ro version thật (đã verify bằng cách tải wheel/METADATA thật, không suy đoán)

### Ignore-list: đổi theo tiền lệ Qwen3-Next, KHÔNG theo §4b cũ

Ví dụ chính thức cho **Qwen3-Next** (`examples/quantization_w8a8_fp8/qwen3_next_example.py`)
— kiến trúc hybrid GDN + full-attention **cùng họ, cùng tên module
`linear_attn`** với Qwen3.5:

```python
ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$", "re:.*linear_attn.*"]
```

`re:.*linear_attn.*` = **bỏ qua toàn bộ khối GDN** (`in_proj_qkv`,
`in_proj_z`, `out_proj`), không chỉ mấy gate nhỏ như plan cũ ở
`vllm-optimization-guide.md` §4b đang ghi. **→ §4b cần cập nhật lại theo hướng
này** (chưa sửa, ghi nhận ở đây trước).

### Version pin — không có tổ hợp nào được test chính thức khớp compressed-tensors 0.15.x

vLLM v0.22.1 pin cứng `compressed-tensors==0.15.0.1` (từ PyPI JSON thật).
Nhưng lịch sử pin của llm-compressor **nhảy qua** dải 0.15.x:

| llm-compressor | compressed-tensors pin | Ngày release |
|---|---|---|
| 0.9.0 / 0.9.0.3 | `==0.13.0` | |
| 0.10.0 | `==0.14.0` | 2026-02-27 (ct) |
| 0.10.0.2 | `==0.14.0.1` | 2026-05-01 (**sau khi 0.15.0.1 đã tồn tại 3 tuần, nhưng KHÔNG bump lên**) |
| **0.10.1a20260407** (alpha) | `>=0.14.1a2` (mở, không chặn trên) | 2026-04-08 — **duy nhất tự nhiên chấp nhận 0.15.0.1** |
| 0.11.0 | `==0.16.0` | 2026-06-02 |
| 0.12.0 (stable mới nhất) | `==0.17.1` | 2026-06-15 |

torch/transformers giữa llm-compressor 0.12.0 và vLLM v0.22.1 thực ra **khớp
nhau** (torch 2.11.0 nằm trong `2.10.0–2.12.0`; transformers 5.9.0–5.10.1
không rơi vào vùng vLLM loại trừ 5.0–5.5.0) — **chỉ `compressed-tensors` là
điểm xung đột thật**, và đúng dải version vLLM cần chưa từng được
llm-compressor test chính thức.

### Lệnh pip đề xuất thử trước (chạy trên GPU pod, Python ≥3.10)

```bash
python3.11 -m venv .venv-quantize && source .venv-quantize/bin/activate

pip install --pre "llmcompressor==0.10.1a20260407" "compressed-tensors==0.15.0.1"

pip install "https://github.com/huggingface/transformers/archive/refs/heads/main.zip" accelerate

python -c "
import torch, transformers, compressed_tensors, llmcompressor
print('torch', torch.__version__, '| transformers', transformers.__version__,
      '| compressed_tensors', compressed_tensors.__version__, '| llmcompressor', llmcompressor.__version__)
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
print('imports OK')
"
```

Nếu bước cuối lỗi import → fallback thử ép `llmcompressor==0.11.0 --no-deps`
rồi `pip install "compressed-tensors==0.15.0.1"` đè lên.

**Bằng chứng cộng đồng cũng gặp đúng lỗi kiểu này (không phải lo xa):**
[llm-compressor#2258](https://github.com/vllm-project/llm-compressor/issues/2258)
(vllm cài đè compressed-tensors xuống bản cũ hơn →
`ModuleNotFoundError: compressed-tensors.modeling`),
[llm-compressor#2268](https://github.com/vllm-project/llm-compressor/issues/2268)
(checkpoint quantize bằng 1 bản llm-compressor chỉ chạy được trên đúng 1 dải
vLLM cụ thể, không portable — câu hỏi bị đóng "not planned", chưa ai giải).

---

## Việc chưa làm / bước tiếp theo

1. Chạy lệnh pip ở mục B trên GPU pod thật, xác nhận import OK + đúng version.
2. Cập nhật `vllm-optimization-guide.md` §4b: đổi ignore-list sang loại cả
   `re:.*linear_attn.*` (theo tiền lệ Qwen3-Next), không chỉ giữ 2 gate nhỏ.
3. Viết `scripts/12_quantize_fp8.py`: load bằng `AutoModelForImageTextToText`,
   recipe `FP8_DYNAMIC` + ignore-list mới, **bắt buộc có bước sanity-generate**
   trước khi save (xem mục A.5) — đừng bỏ qua bước này.
4. Load checkpoint FP8 thật trên đúng image `vllm/vllm-openai:v0.22.1`, verify
   không lỗi parse/import.
5. Chạy `09_gpqa_accuracy.py` so sánh với baseline BF16, giữ margin ≥5 điểm so
   ngưỡng 0.30. Nếu tụt sát ngưỡng → cân nhắc lever dự phòng ở mục A.4
   (GPTQ/AWQ calibrate 128–256 sample).

**Sources:** [Choosing your model](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-model/) ·
[Choosing scheme](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-scheme/) ·
[Choosing algorithm](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-algo/) ·
[Choosing dataset](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-dataset/) ·
[Qwen3-Next FP8 example](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_fp8/qwen3_next_example.py) ·
[llm-compressor#2258](https://github.com/vllm-project/llm-compressor/issues/2258) ·
[llm-compressor#2268](https://github.com/vllm-project/llm-compressor/issues/2268) ·
[PyPI vllm 0.22.1 JSON](https://pypi.org/pypi/vllm/0.22.1/json) ·
[PyPI llmcompressor JSON](https://pypi.org/pypi/llmcompressor/json)
