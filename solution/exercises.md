# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
Qua bốn mức temperature, có thể thấy temperature càng cao thì câu trả lời càng đa dạng và ít ổn định hơn. Ở temperature 0.0, mô hình trả lời rất nhất quán và tập trung vào một thông tin cụ thể, trong khi ở 1.0–1.5, cách diễn đạt và nội dung thay đổi nhiều hơn, thậm chí xuất hiện thêm các chi tiết khác nhau.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
Em sẽ chọn temperature khoảng 0.2–0.3 cho chatbot hỗ trợ khách hàng. Mức này giúp câu trả lời chính xác nhưng vẫn giữ tính robust, đồng thời vẫn linh hoạt để diễn đạt tự nhiên thay vì quá máy móc.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
Tổng output/ngày: 10.000 × 3 × 350 = 10.500.000 token
GPT-4o: 10,5 × $10 = $105/ngày
GPT-4o mini: 10,5 × $0,60 = $6,30/ngày
GPT-4o đắt gấp ~16,7 lần so với GPT-4o mini về phần output.
Trường hợp dùng GPT-4o: cần phân tích phức tạp, yêu cầu suy luận logic sâu
hoặc sáng tạo cao.
Trường hợp dùng mini: chatbot đơn giản, trả lời thông tin cơ bản, tóm tắt
văn bản với lượng lớn người dùng.
---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
Hai phản hồi khác nhau về độ dài, từ vựng và cách minh họa. System prompt “giáo viên tiểu học” khiến model trả lời ngắn gọn, dùng từ ngữ đơn giản và ví dụ gần gũi như “cuốn sổ ghi điểm”. Ngược lại, prompt “chuyên gia tài chính” tạo câu trả lời dài và chuyên sâu hơn, sử dụng các thuật ngữ như phi tập trung, bất biến, hash, node và giải thích theo hướng kỹ thuật. Điều này cho thấy system prompt có ảnh hưởng mạnh đến vai trò, phong cách, mức độ chi tiết và cách model lựa chọn nội dung khi trả lời.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
Số từ: 119
Token ước tính: 158.67
Token thực tế (tiktoken): 152
Chênh lệch: 4.20%
Tokenizer của mô hình thường được tối ưu tốt hơn cho tiếng Anh; nhiều từ tiếng Việt có dấu và cách tách từ khiến chúng dễ bị chia thành nhiều token hơn.
---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
Streaming quan trọng nhất khi ứng dụng cần phản hồi dài hoặc người dùng cần thấy kết quả ngay, chẳng hạn như chatbot, vì nội dung được hiển thị từng phần giúp giảm cảm giác phải chờ đợi. Ngược lại, non-streaming phù hợp hơn với các tác vụ cần nhận toàn bộ kết quả rồi mới xử lý tiếp, như gọi API phía backend, tạo dữ liệu có cấu trúc hoặc khi ứng dụng không cần hiển thị kết quả theo thời gian thực.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
Exponential backoff giúp giảm dần tần suất gửi request khi API quá tải, từ đó giảm áp lực lên server và cho phép hệ thống tự phục hồi. Nếu hàng nghìn client cùng retry với delay cố định, chúng có thể gây ra một "cơn bão request" (thường gọi là thảm họa thundercat), làm trầm trọng hơn tình trạng quá tải và kéo dài thời gian lỗi.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
System prompt: "Bạn là trợ giảng thân thiện của khóa học AI, trả lời ngắn gọn và dễ hiểu bằng tiếng Việt."
- "trợ giảng thân thiện": Xác định tông giọng (tone) cởi mở, nhiệt tình, giúp người dùng thoải mái hỏi bài.
- "ngắn gọn": Giới hạn độ dài câu trả lời để tiết kiệm token (chi phí) và giảm độ trễ khi hiển thị trên CLI.
- "bằng tiếng Việt": Đảm bảo model không tự ý chuyển sang tiếng Anh khi giải thích các thuật ngữ AI.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
- Hạn chế lớn nhất: History bị giới hạn cứng ở 3 lượt gần nhất. Nếu người dùng hỏi lại thông tin từ đầu phiên chat, model sẽ quên hoàn toàn.
- Cải thiện: Sử dụng kỹ thuật Tóm tắt ngữ cảnh (Context Summarization).
- Cách triển khai: Thay vì xóa hẳn các tin nhắn cũ, ta gọi một model phụ (như GPT-4o-mini) để tóm tắt các tin nhắn bị loại bỏ thành một đoạn văn ngắn, sau đó chèn đoạn tóm tắt này vào System Prompt để model chính luôn nắm được bối cảnh tổng thể của toàn bộ cuộc trò chuyện mà không tốn quá nhiều token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
