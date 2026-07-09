# Giải thích chi tiết: vì sao ignore-list trong scripts/14_quantize_fp8.py lại chọn đúng những layer đó

Tài liệu này giải thích lại từ đầu, không dùng thuật ngữ mà không định nghĩa trước — dành cho người chưa quen quantization. Nếu bạn đã hiểu các khái niệm nền, có thể nhảy thẳng xuống phần "Áp vào từng dòng ignore".

---

## 1. Các khái niệm nền, giải thích từ số 0

### 1.1. Ma trận và phép nhân ma trận (GEMM)

Bên trong một mạng neural network, phần lớn phép tính là **nhân ma trận**: lấy một bảng số (ma trận) đại diện cho dữ liệu đầu vào, nhân với một bảng số khác đại diện cho "trọng số" (weight) mà model đã học được, ra một bảng số kết quả.

**GEMM** là viết tắt của **General Matrix Multiply** — "phép nhân ma trận tổng quát". Đây là tên gọi kỹ thuật (bắt nguồn từ thư viện toán học BLAS rất lâu đời) cho đúng phép toán "nhân hai ma trận với nhau". Trong ngữ cảnh model ngôn ngữ, hầu như mọi lớp `Linear` (lớp tuyến tính — một lớp mà công thức là `đầu_ra = đầu_vào × ma_trận_trọng_số`) khi chạy trên GPU đều thực chất là một phép GEMM.

Vì sao phải gọi tên riêng cho nó? Vì GPU có phần cứng chuyên biệt (gọi là **Tensor Core** trên các GPU NVIDIA đời mới) được thiết kế để làm phép GEMM cực nhanh. Nếu một phép tính trong model **là** GEMM, nó tận dụng được phần cứng chuyên biệt này. Nếu một phép tính **không phải** GEMM (ví dụ chỉ là tra bảng, hoặc phép cộng đơn giản), thì dù bạn có tối ưu kiểu gì, phần cứng Tensor Core cũng không giúp được — không có tốc độ để mà tăng.

**Vì sao điều này quan trọng cho việc chọn layer nào để quantize:** quantize (nói ở mục dưới) chỉ đem lại lợi ích tốc độ thật sự cho các phép tính LÀ GEMM. Với phép tính không phải GEMM, quantize nó chỉ có rủi ro (làm sai kết quả) mà không có lợi ích tốc độ đi kèm — đây là lý do đầu tiên khiến một layer bị đưa vào danh sách "ignore" (bỏ qua, không quantize).

### 1.2. Số thực trong máy tính: BF16 vs FP8

Máy tính lưu số thực (có phần thập phân) theo chuẩn dấu phẩy động (floating point). Số bit dùng để lưu một số càng nhiều thì độ chính xác càng cao, nhưng cũng tốn càng nhiều bộ nhớ và băng thông để di chuyển dữ liệu đó qua lại.

- **BF16** (Brain Float 16): dùng 16 bit (2 byte) để lưu mỗi số. Đây là định dạng gốc mà model Qwen3.5-2B được huấn luyện và lưu trữ.
- **FP8** (Float 8): dùng 8 bit (1 byte) để lưu mỗi số — chỉ bằng một nửa BF16. Ít bit hơn nghĩa là: (a) file model nhỏ hơn một nửa, (b) khi GPU đọc/tính trên các số này, lượng dữ liệu cần di chuyển giảm một nửa, và Tensor Core đời mới (Hopper — chính là GPU H200 dùng để chấm thi, và Ada — GPU L40S dùng để test) có mạch tính GEMM trên FP8 nhanh hơn hẳn so với tính trên BF16.

Đánh đổi: 8 bit biểu diễn được ít giá trị khác nhau hơn 16 bit rất nhiều, nên khi ép một số BF16 về FP8, bạn buộc phải **làm tròn** nó về giá trị FP8 gần nhất có thể biểu diễn được — tức là mất thông tin, sinh ra sai số.

### 1.3. Quantize là gì

**Quantize** (lượng tử hóa) là quá trình chuyển đổi trọng số (và đôi khi cả activation — giá trị trung gian trong lúc tính toán) từ định dạng nhiều bit (BF16) sang định dạng ít bit hơn (FP8), chấp nhận đánh đổi một chút sai số để đổi lấy: model nhỏ hơn, chạy nhanh hơn.

**W8A8** là tên gọi tắt cho kiểu quantize mà script này dùng: **W**eight 8-bit (trọng số dùng 8 bit) + **A**ctivation 8-bit (giá trị trung gian lúc tính cũng dùng 8 bit). Cả hai vế của phép GEMM đều là FP8, nên Tensor Core có thể chạy đúng chế độ FP8×FP8 nhanh nhất.

### 1.4. RTN / FP8_DYNAMIC — cách quantize không cần dữ liệu mẫu

Có nhiều thuật toán quantize khác nhau. Cách đơn giản nhất gọi là **RTN — Round To Nearest** ("làm tròn về giá trị gần nhất"): với mỗi số BF16, tìm giá trị FP8 gần nó nhất rồi thay thế, không cần phân tích gì thêm, không cần chạy thử model qua dữ liệu mẫu (gọi là "data-free" — không cần dữ liệu).

`FP8_DYNAMIC` mà script dùng chính là biến thể RTN cho FP8: trọng số được làm tròn 1 lần khi quantize (per-channel — tính hệ số scale riêng cho từng "kênh" của ma trận để giảm sai số), còn activation (giá trị trung gian lúc chạy) được tính scale động ngay tại thời điểm suy luận (dynamic, "per-token" — mỗi token một hệ số riêng). Đây là kiểu nhanh nhất, đơn giản nhất, và may mắn là với đúng định dạng FP8 (khác với việc ép xuống 4-bit chẳng hạn), tài liệu chính thức của llm-compressor xác nhận RTN cho kết quả "good recovery" — tức độ chính xác giữ được tốt, không phải giải pháp yếu tạm bợ.

Các thuật toán khác (SmoothQuant, AWQ, GPTQ...) chính xác hơn nhưng cần một tập dữ liệu mẫu để "hiệu chỉnh" (calibrate) — phức tạp hơn, chậm hơn, và luật thi cấm dùng nội dung trace thật để làm việc này (tránh gian lận kiểu học trước câu hỏi).

### 1.5. Vì sao có layer "nhạy cảm" hơn layer khác với sai số làm tròn

Không phải phép tính nào trong model cũng chịu đựng sai số làm tròn như nhau. Có hai kiểu kiến trúc xử lý dữ liệu rất khác nhau bên trong Qwen3.5-2B:

- **Full attention (self-attention chuẩn)**: mỗi token nhìn lại tất cả token trước đó thông qua một phép tính tương đối "phẳng" — sai số nhỏ ở một chỗ thường không bị khuếch đại dồn dập theo chiều dài chuỗi.
- **Gated DeltaNet (GDN — linear attention kiểu hồi quy)**: đây là cơ chế thay thế attention truyền thống ở phần lớn các layer của Qwen3.5 (18/24 layer). Nó hoạt động như một "trạng thái" (state) có kích thước cố định, được cập nhật liên tục qua từng token — giống như một bộ nhớ chạy xuyên suốt cả câu, mỗi bước cộng dồn thêm thông tin mới vào state đó. Vấn đề: nếu trọng số dùng để cập nhật state này bị làm tròn sai một chút, sai số đó **không biến mất** mà bị cuốn vào state và tiếp tục ảnh hưởng đến mọi bước sau — với model này cho phép chuỗi dài tới 262.144 token, sai số có rất nhiều bước để cộng dồn/khuếch đại.

Đây là lý do kỹ thuật cốt lõi khiến khối GDN được đối xử khác hẳn phần còn lại.

---

## 2. Nguyên tắc chọn quantize hay ignore

Sau khi có đủ khái niệm nền, nguyên tắc chọn có thể tóm gọn qua 2 câu hỏi cho mỗi layer:

**Câu hỏi 1 — Layer này có phải phép GEMM lớn, và có thật sự được chạy khi phục vụ hay không?**
Nếu không phải GEMM (ví dụ chỉ là tra bảng), hoặc layer đó không bao giờ được dùng trong cách bạn phục vụ model (ví dụ phần xử lý hình ảnh mà bạn chỉ phục vụ văn bản) → quantize nó **không đem lại lợi ích tốc độ nào**, chỉ có rủi ro sai. Ignore.

**Câu hỏi 2 — Nếu có, layer này có chịu được sai số làm tròn tốt không, và mức tiết kiệm có xứng đáng với rủi ro không?**
Nếu layer đó nhỏ (tiết kiệm được ít) nhưng lại nhạy cảm với sai số (như cơ chế hồi quy state ở trên, hoặc một layer nhỏ mà lỗi lan rộng) → tỷ lệ lợi ích/rủi ro xấu. Ignore.

Nếu cả hai câu trả lời đều thuận lợi (là GEMM lớn thật sự được dùng, và có tiền lệ/bằng chứng chịu tốt sai số 8-bit) → quantize.

---

## 3. Áp nguyên tắc vào từng dòng trong IGNORE (scripts/14_quantize_fp8.py:56-66)

```python
IGNORE = [
    "lm_head",
    "re:.*embed_tokens$",
    "re:^mtp.*",
    "re:.*visual.*",
    "re:.*linear_attn.*",
]
```

### `lm_head`
Đây là lớp cuối cùng của model, biến trạng thái ẩn (hidden state) thành điểm số (logit) cho từng từ trong từ điển (vocabulary) để chọn ra token tiếp theo. Bình thường đây LÀ một phép GEMM lớn (số từ trong từ điển của Qwen3.5 là 248.044 — một ma trận rất to). Nhưng ở model này, `lm_head` được **buộc chung trọng số (tied weight)** với `embed_tokens` (lớp đầu vào, biến token thành vector) — nghĩa là hai lớp này thực chất dùng chung một bảng số duy nhất trong bộ nhớ, không phải hai bảng riêng. Vì `embed_tokens` là kiểu lớp **tra bảng** (Embedding — xem mục dưới), không phải GEMM, công cụ quantize (chỉ nhắm vào lớp `Linear`) vốn dĩ không đụng vào nó. Để giữ hai bên nhất quán (không thể quantize một bên của cùng một bảng số mà bên kia lại không), `lm_head` cũng được giữ nguyên BF16.

### `re:.*embed_tokens$` (chữ `re:` nghĩa là dòng này là một mẫu regex — biểu thức để so khớp tên lớp, không phải tên chính xác)
`embed_tokens` là lớp **Embedding** — nhiệm vụ của nó chỉ là: với mỗi token (một số nguyên đại diện cho một từ/mảnh từ), tra ra dòng tương ứng trong một bảng số lớn. Đây là phép **tra bảng** (index lookup), không phải phép nhân ma trận GEMM. Vì công cụ quantize trong script này chỉ nhắm vào lớp kiểu `Linear` (`targets="Linear"`), dòng ignore này thực ra không match được gì thêm — nó được liệt kê ra chỉ để rõ ràng, đúng với cấu trúc của checkpoint gốc mà script tham chiếu (RedHatAI).

### `re:^mtp.*`
`mtp` là viết tắt của **Multi-Token Prediction** — một cơ chế đoán trước nhiều token cùng lúc để tăng tốc sinh văn bản (một dạng "speculative decoding" — dự đoán trước rồi kiểm chứng lại, thay vì sinh từng token một cách tuần tự chậm chạp). Phần này chỉ chiếm 2,7% tổng số tham số của model — quantize nó tiết kiệm được rất ít bộ nhớ/tốc độ. Nhưng nếu phần dự đoán trước này bị sai số làm giảm chất lượng, tỷ lệ đoán trúng (accept rate) của cơ chế speculative sẽ giảm, khiến hệ thống phải làm lại nhiều hơn — tức có thể làm **chậm đi** thay vì nhanh lên. Được ít, mất nhiều nếu sai — nên ignore.

### `re:.*visual.*`
Đây là toàn bộ phần xử lý hình ảnh/video (vision tower) của model — vì Qwen3.5-2B thực chất là model đa phương tiện (nhận cả ảnh lẫn văn bản), dù bài toán của bạn chỉ phục vụ văn bản thuần. Khi chạy `vllm serve` với cờ `--language-model-only` (đã xác nhận có tồn tại và đúng tên ở README của model), toàn bộ phần vision này **không được nạp vào GPU khi phục vụ**. Quantize một phần không bao giờ được chạy là công cốc — không có lợi ích gì, chỉ tốn thời gian quantize và thêm rủi ro (nếu lỡ có đường dẫn code nào đó vẫn chạm vào weight này thì lại thêm một nguồn lỗi tiềm ẩn).

### `re:.*linear_attn.*`
Đây là dòng quan trọng nhất, ignore **toàn bộ khối Gated DeltaNet** (5 lớp con trong mỗi block: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`) trên 18 trong tổng số 24 layer của model (6 layer còn lại dùng full attention chuẩn).

Như giải thích ở mục 1.5, Gated DeltaNet là cơ chế "state hồi quy" — trọng số của nó quyết định cách một bộ nhớ trạng thái được cập nhật liên tục qua toàn bộ chiều dài chuỗi (tối đa 262.144 token). Sai số làm tròn khi quantize sẽ bị cuốn vào state này và cộng dồn qua rất nhiều bước, khác hẳn với attention chuẩn nơi sai số ít có cơ hội khuếch đại theo cách đó.

Quan trọng: đây **không phải suy luận lý thuyết suông của script**. Có hai bằng chứng thực nghiệm thật:
1. Checkpoint FP8 chính thức đã công bố `RedHatAI/Qwen3.5-4B-FP8-dynamic` (một model cùng họ, cùng kiến trúc) — khi đọc trực tiếp `config.json` của checkpoint đó, danh sách ignore của họ cũng loại bỏ đúng toàn bộ khối `linear_attn`, không chỉ một phần nhỏ.
2. Ví dụ chính thức của chính thư viện llm-compressor cho **Qwen3-Next** (một model khác nhưng cùng cơ chế Gated DeltaNet, cùng tên module `linear_attn`) cũng ignore đúng pattern `re:.*linear_attn.*`.

Nói cách khác: những người từng thử quantize đúng cơ chế này trước đây (cả đội ngũ phát hành checkpoint thật, cả đội phát triển thư viện) đều đưa ra cùng một kết luận — quantize khối này rủi ro không đáng, nên script không phải là người đầu tiên "đánh cược" thử FP8 lên state hồi quy này mà không có tiền lệ nào để dựa vào.

---

## 4. Phần KHÔNG bị ignore — được quantize

Hai loại lớp được quantize xuống FP8:

- `self_attn.{q,k,v,o}_proj` — 4 phép chiếu (projection) tạo nên cơ chế full-attention chuẩn, chỉ có ở 6/24 layer.
- `mlp.{gate,up,down}_proj` — 3 phép chiếu tạo nên mạng feed-forward (MLP), có ở toàn bộ 24 layer (cả layer dùng GDN lẫn layer dùng full-attention).

Đây là các phép GEMM **lớn nhất, tốn tính toán nhiều nhất** trong model (theo báo cáo tham số thực đo được khi chạy script: 45% tổng số tham số kiểu Linear nằm ở đây) — và là loại lớp mà cộng đồng đã kiểm chứng rất nhiều lần là chịu tốt sai số làm tròn FP8 kiểu RTN. Đúng cả hai điều kiện ở mục 2: vừa là GEMM lớn thật sự được dùng khi phục vụ, vừa có tiền lệ an toàn — nên đây là nơi FP8 đem lại lợi ích tốc độ thật mà rủi ro thấp.

---

## Tóm tắt một câu

Ignore một layer khi: nó không phải phép nhân ma trận lớn thật sự tăng tốc được (embedding, phần vision không dùng tới), **hoặc** nó nhạy cảm bất thường với sai số làm tròn do cách nó xử lý dữ liệu (state hồi quy Gated DeltaNet, hoặc một layer nhỏ mà lỗi ảnh hưởng lan rộng như đầu dự đoán trước). Còn lại — attention chuẩn và MLP — thì quantize, vì đó là nơi vừa có lợi ích tốc độ thật, vừa đã có bằng chứng thực tế là an toàn.
