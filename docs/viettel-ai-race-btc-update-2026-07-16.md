# Viettel AI Race: BTC Update So Với `VIETTEL AI RACE.pdf`

Ngày cập nhật: 2026-07-16

Tài liệu này dùng để thay thế khi tra cứu rule chính thức và anti-cheat. File [VIETTEL AI RACE.pdf](/Users/ngoquangduc/Desktop/workspace/viettelai-race/docs/VIETTEL%20AI%20RACE.pdf) hiện là bản tổng hợp/phân tích nội bộ cập nhật ngày 2026-07-05, không còn nên dùng như nguồn rule chính thức sau khi BTC update đề.

Nguồn đối chiếu:
- Bản cũ trong repo: `docs/VIETTEL AI RACE.pdf`
- Bản update mới của BTC: file đính kèm `pasted-text.txt`

## Kết luận nhanh

PDF cũ đã lệch ở nhiều điểm nền tảng:
- Không còn an toàn để giả định `Qwen/Qwen3.5-2B`, `120 request`, `MIG H200 18GB`, `Ubuntu 22.04`, `CUDA 12.x`.
- Quy trình chấm đã đổi: vòng online chỉ chấm ERS; accuracy không chạy ngay trên từng submission.
- Anti-cheat được viết chặt hơn và nhấn mạnh hậu kiểm thủ công, hành vi serving thực tế, và quyền void bài sau khi rà soát.

Nếu team cần một tài liệu để bám khi làm bài từ bây giờ, hãy bám file này thay vì bám trực tiếp `VIETTEL AI RACE.pdf`.

## Những thay đổi lớn so với PDF cũ

### 1. Phạm vi đề bài

PDF cũ mô tả rất cụ thể một bài vòng 1:
- Model: `Qwen/Qwen3.5-2B`
- Workload: `120 request`
- Accuracy gate: `100` câu GPQA Diamond cố định
- Nộp bài: Docker Hub public + `docker-compose.yml`

Bản update mới của BTC đã chuyển cách mô tả sang mức tổng quát hơn:
- Model do BTC chỉ định theo từng vòng, không đóng cứng ở `Qwen3.5-2B`.
- Trace công khai chỉ còn timestamp + token counts; prompt thật chỉ được gửi lúc chấm.
- Sau vòng online, mỗi đội chỉ được chọn tối đa `5 submissions` để BTC hậu kiểm rồi mới chạy GPQA Diamond full.

Hệ quả:
- Phần phân tích/chiến lược trong PDF cũ bám rất mạnh vào trace và model của một vòng cụ thể, nên không thể xem là rule chung nữa.

### 2. Môi trường đánh giá

PDF cũ ghi:
- `1 instance MIG H200`
- `18GB VRAM`, `3 CPU cores`, `8GB RAM`
- `Ubuntu 22.04 LTS`, `CUDA 12.x`

Bản update mới ghi:
- `NVIDIA H200 GPU`
- `Ubuntu 24.04 LTS`
- `NVIDIA driver 590.x` với `CUDA 13.x`

Hệ quả:
- Mọi giả định tuning trước đây dựa trên slice MIG 18GB có thể đã không còn đúng.
- Những quyết định giới hạn `max-model-len`, memory budget, batching budget trong PDF cũ không nên dùng lại như mặc định.

### 3. Cơ chế chấm

PDF cũ mô tả score cuối cùng của vòng 1 là:
- `Score = 100 x ERS x f(Δ)`

Bản update mới làm rõ quy trình hai tầng:
- Vòng online: chỉ chấm `ERS`.
- Kết thúc vòng online: đội tự chọn tối đa `5 submissions`.
- BTC hậu kiểm tính hợp lệ trước.
- Chỉ các submission hợp lệ mới bị chấm `GPQA Diamond full` để tính `f(Δ)` và điểm tổng.

Hệ quả:
- Không thể dùng một cấu hình chỉ để tối ưu latency rồi đổi hành vi ở bước accuracy.
- Bài top online vẫn có thể bị void hoặc rơi hạng sau hậu kiểm/accuracy.

### 4. Trace và prompt

PDF cũ mô tả nhiều chi tiết rất cụ thể về trace vòng 1, như burst pattern và prefix continuity.

Bản update mới nhấn mạnh:
- Arrival Poisson đã được "đóng băng" thành timestamp cố định.
- Mọi đội chạy cùng timeline deterministic.
- Bản trace public chỉ để provision tải.
- Prompt thật không lộ ra trong trace public; BTC chỉ gửi prompt thật vào lúc chấm.

Hệ quả:
- Mọi chiến thuật tận dụng nội dung trace public để học tủ hoặc pre-bake câu trả lời đều rủi ro cao và nằm đúng vùng BTC đang siết.

## Anti-Cheat: phần cần bám chặt

### Những gì BTC nói thẳng là không được làm

Bản update mới liệt kê các nhóm vi phạm sau:
- `Pre-bake / Hardcode`: tính sẵn đáp án thay vì suy luận tại thời điểm phục vụ.
- `Dual-path`: rẽ nhánh hành vi giữa lúc đo độ trễ và lúc kiểm tra chất lượng.
- `Gaming metrics`: đệm rỗng, cắt ngắn chuỗi sinh trái phép để né hậu kiểm.
- `Can thiệp hạ tầng`: gọi mạng ngoài, sửa tokenizer/weights, làm bẩn tài nguyên.
- `Bất trung thực quy trình`: tráo image sau khi nộp, lộ dữ liệu.

BTC cũng nói rõ:
- Điểm hệ thống tự động chưa phải kết quả cuối.
- BTC có thể rà soát image, cấu hình, log, và luồng serving.
- Submission gian lận có thể bị `void` hoặc điều chỉnh hạng trực tiếp.

### Diễn giải an toàn cho team

Đây là checklist vận hành an toàn:
- Không hardcode output theo request ID, prompt pattern, trace order, hay probe subset.
- Không chuẩn bị sẵn đáp án cho hidden prompts hoặc cache sẵn câu trả lời dựa trên trace public.
- Không chạy hai mode khác nhau giữa "benchmark nhanh" và "accuracy thật".
- Không trả token rác, token trống, hoặc chủ động cắt output sai spec chỉ để kéo TTFT/TPOT xuống.
- Không gọi API/model/dịch vụ bên ngoài trong lúc serving.
- Không thay image sau khi chọn submission để hậu kiểm.
- Không làm gì khiến cùng một image nhưng hành vi thực tế khác đi tùy ngữ cảnh chấm.

### Điểm mơ hồ cần hiểu theo hướng bảo thủ

Cụm `sửa tokenizer/weights` trong bản update mới hơi mơ hồ nếu đọc tách rời, vì ở phần "phương pháp tối ưu được phép" BTC vẫn cho phép các hướng như quantization.

Cách hiểu an toàn nhất hiện tại:
- Được tối ưu serving stack và quantize theo hướng phục vụ hợp lệ.
- Không được tráo tokenizer/model sang thực thể khác, không patch weights theo kiểu cài đáp án, và không chỉnh tài nguyên/hành vi để đánh lừa hệ thống chấm.

Nếu team định làm một kỹ thuật nằm gần ranh giới này, nên tự hỏi:
- Kỹ thuật đó có còn hợp lý trong một production serving stack trung thực không?
- Nếu BTC đọc image, config và log thì có thể kết luận mình đang "benchmark hack" không?

Nếu câu trả lời không chắc chắn, nên tránh.

## Tác động trực tiếp tới repo hiện tại

Các nội dung trong PDF cũ không còn nên xem là "fact" cho rule hiện hành, đặc biệt là:
- Model cố định `Qwen3.5-2B`
- Hạ tầng `MIG H200 18GB`
- Trace `120 request`
- Accuracy gate `100 câu`
- Một số chiến lược tối ưu được suy ra từ trace vòng 1 cũ

Repo này vẫn dùng được tốt cho hướng nghiên cứu serving, profiling, trace replay và tuning. Nhưng khi ra quyết định nộp bài, rule nên bám theo update mới của BTC trước.

## Khuyến nghị ngắn gọn

- Dùng file này như tài liệu rule tạm thời mới.
- Giữ `VIETTEL AI RACE.pdf` làm tài liệu lịch sử/phân tích cũ, không dùng làm nguồn anti-cheat chính.
- Mọi tối ưu nên chịu được hậu kiểm dưới cùng một image, cùng một hành vi serving, và không dựa vào hidden prompt knowledge.

