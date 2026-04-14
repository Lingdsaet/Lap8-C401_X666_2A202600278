# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Phan Hoài Linh 
**Vai trò trong nhóm:**Tech Lead / Retrieval Owner / Eval Owner / Documentation Owner  
**Ngày nộp:** 13/4/2026
**Độ dài:** ~650 từ

---

## 1. Tôi đã làm gì trong lab này?

Tôi thực hiện toàn bộ pipeline RAG một mình, lần lượt qua cả 4 sprint.

**Sprint 1 — Build RAG Index:** Implement `preprocess_document()` để parse metadata từ header file (Source, Department, Effective Date, Access) bằng regex theo pattern `"Key: Value"`, đồng thời làm sạch text — chuẩn hóa khoảng trắng, loại bỏ ký tự điều khiển, chuẩn hóa dấu ngoặc kép curly. Phần chunking trong `chunk_document()` được thiết kế theo cấu trúc tự nhiên: ưu tiên cắt tại ranh giới section (`=== ... ===`), nếu section quá dài thì `_split_by_size()` cắt tiếp theo paragraph (`\n\n`), cuối cùng mới fallback sang `_split_long_paragraph()` cắt theo câu. Overlap được tính bằng cách lấy lại các paragraph cuối của chunk trước, không cắt cứng theo số ký tự.

**Sprint 2 — Embedding & Storage:** Chọn Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`) chạy hoàn toàn local, không cần API key. Model được cache qua singleton `_get_model()` để tránh load lại mỗi lần gọi `get_embedding()`. Lưu vào ChromaDB với cosine similarity, upsert theo batch từng file thay vì từng chunk.

**Sprint 3 & 4 — Eval & Cải thiện:** Chạy `list_chunks()` và `inspect_metadata_coverage()` để kiểm tra chất lượng index, dùng `search()` để test retrieval thực tế, sau đó điều chỉnh `CHUNK_SIZE` và `CHUNK_OVERLAP` dựa trên kết quả quan sát.

---

## 2. Điều tôi hiểu rõ hơn sau lab này

**Chunking theo cấu trúc tự nhiên quan trọng hơn con số token tuyệt đối.**

Lúc đầu tôi nghĩ đặt `CHUNK_SIZE = 400` và cắt đều là đủ. Khi implement `_split_by_size()` theo cách cắt cứng theo ký tự, tôi nhận ra một điều khoản như "Điều 3: Khách hàng được hoàn tiền trong vòng 30 ngày..." bị tách đôi — nửa đầu ở chunk này, điều kiện đi kèm lại nằm ở chunk sau. Model trả về đúng con số "30 ngày" nhưng thiếu hoàn toàn phần điều kiện. Sau khi chuyển sang cắt theo paragraph rồi mới theo câu, context được giữ nguyên trong một chunk và câu trả lời đầy đủ hơn hẳn.

**Metadata là bộ lọc độc lập với embedding, không phải thông tin thừa.**

Ban đầu tôi chỉ coi metadata như nhãn để debug. Qua `inspect_metadata_coverage()`, tôi nhận ra `effective_date` có thể dùng để lọc trước khi query — tránh trường hợp câu hỏi về chính sách hiện tại lại trả về chunk từ phiên bản cũ hơn. Embedding tìm theo ngữ nghĩa, metadata lọc theo điều kiện cứng — hai lớp này bổ sung cho nhau.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

**Khó khăn chính: logic `header_done` trong `preprocess_document()` dễ fail âm thầm.**

Giả thuyết ban đầu là mọi file `.txt` đều có ít nhất một dòng bắt đầu bằng `===` để đánh dấu hết phần header. Nếu file nào không có dòng đó, biến `header_done` không bao giờ được set thành `True`, toàn bộ nội dung bị rơi vào nhánh `continue` và `chunk_document()` nhận về `text` rỗng — trả về `chunks = []` mà không báo lỗi gì.

Bug này khó phát hiện vì `build_index()` vẫn chạy bình thường, chỉ in ra "0 chunks" mà không raise exception. Tôi chỉ phát hiện khi chạy `list_chunks()` và thấy collection trống. Fix bằng cách thêm fallback: nếu sau khi duyệt hết các dòng metadata mà `header_done` vẫn là `False`, tự động coi toàn bộ nội dung là body.

**Điều ngạc nhiên:** `paraphrase-multilingual-MiniLM-L12-v2` hiểu tiếng Việt tốt hơn kỳ vọng. Với query như "điều kiện hoàn trả hàng", similarity score vẫn đạt > 0.75 dù cách diễn đạt trong tài liệu gốc khá khác ("khách hàng có quyền yêu cầu đổi/trả").

---

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi:** *"Khách hàng có thể yêu cầu hoàn tiền sau bao nhiêu ngày kể từ ngày mua hàng?"*

**Phân tích:**

Baseline trả lời đúng con số ("30 ngày") nhưng thiếu các điều kiện đi kèm. Điểm Faithfulness cao (không bịa), Relevance tốt, nhưng Completeness thấp vì câu trả lời không đề cập điều kiện hàng còn nguyên vẹn và phải có hóa đơn.

Lỗi nằm ở tầng **Indexing**: chunk chứa "30 ngày" bị cắt trước đoạn liệt kê điều kiện. Retrieval hoạt động đúng — trả về đúng chunk liên quan nhất. Generation cũng đúng — model không bịa thông tin không có trong context. Vấn đề là context đưa vào đã thiếu từ đầu.

Variant fix: tăng `CHUNK_SIZE` từ 400 lên 600 và `CHUNK_OVERLAP` từ 80 lên 120. Chunk mới gộp được cả điều kiện đi kèm. Completeness cải thiện rõ, tổng điểm tăng từ ~3/5 lên ~4.5/5.

Bài học: khi Generation trả về câu trả lời đúng nhưng thiếu, không nên chỉnh prompt trước — hãy kiểm tra chunk bằng `list_chunks()` và `search()` trước tiên.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

**Thử hybrid retrieval (BM25 + embedding):** Kết quả eval cho thấy các câu hỏi chứa tên sản phẩm hoặc mã điều khoản cụ thể đôi khi bị retrieval miss vì embedding ưu tiên ngữ nghĩa hơn keyword chính xác. Tôi sẽ thêm BM25 sparse retrieval song song và dùng Reciprocal Rank Fusion để merge, kỳ vọng cải thiện recall cho các câu hỏi factual dạng tra cứu.

**Tự động hóa eval bằng LLM-as-judge:** Hiện tại chấm điểm thủ công chỉ cover được ~10 câu. Nếu dùng một model nhỏ để tự động chấm Faithfulness và Completeness, có thể chạy eval trên toàn bộ test set và phát hiện regression mỗi khi thay đổi chunk config.

---


