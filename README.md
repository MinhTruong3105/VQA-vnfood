# VQA-vnfood
# Dự Án Cuối Kỳ Môn Học Sâu (Deep Learning)

**Trường:** Đại học Tôn Đức Thắng (TDTU)
**Sinh viên thực hiện:** Nguyễn Thái Khánh Nam - 523H0159

Dự án này bao gồm hai bài toán chính ứng dụng Học Sâu:
1. **Visual Question Answering (VQA)** trên miền dữ liệu Món ăn Việt Nam.
2. **Abstractive Text Summarization** sử dụng BARTpho cho tin tức tiếng Việt.

---

## 🔗 Liên kết tài nguyên (Dataset & Checkpoints)
*Do giới hạn dung lượng, dữ liệu và trọng số mô hình được lưu trữ tại:*
- **Dataset VQA & Summarization:** [Link Google Drive / HuggingFace Dataset]
- **Checkpoints (A1, A2, B2, BARTpho):** [Link HuggingFace Model Hub]
- **Video Demo & Slides:** [Link Google Drive]

---

## 🚀 Bài 1: Visual Question Answering (VQA)
### 1. Giới thiệu Dữ liệu
- Miền chuyên biệt: Món ăn Việt Nam.
- Cấu trúc: Hơn 2000 cặp (ảnh, câu hỏi, câu trả lời) cho tập Train và 50 câu tập Test.
- Các dạng câu hỏi: Yes/No, Nhận dạng, Đếm số lượng, Màu sắc,...

### 2. Cấu trúc Mô hình
Dự án triển khai và so sánh 4 cấu hình:
- **Cấu hình A1:** ViT (Image Encoder) + PhoBERT (Text Encoder) + LSTM (Decoder).
- **Cấu hình A2:** ViT + PhoBERT + Transformer (Decoder).
- **Cấu hình B1 (Zero-shot):** BLIP-2 (`Salesforce/blip2-opt-2.7b`) kết hợp dịch máy Vi-En.
- **Cấu hình B2 (Fine-tuned):** BLIP-2 tinh chỉnh bằng LoRA.

*(Chèn ảnh bảng so sánh hoặc biểu đồ đánh giá vào đây sử dụng `![alt](assets/comparison.png)`)*

### 3. Đánh giá Mô hình
Các mô hình được đánh giá thông qua các độ đo: VQA Accuracy (Exact Match), BLEU, ROUGE-L, METEOR và BERTScore. (Xem chi tiết trong Báo cáo PDF).

---

## 📝 Bài 2: Abstractive Text Summarization
### 1. Đặt vấn đề
Trình bày ngắn gọn lý do chọn bài toán Tóm tắt văn bản tin tức tiếng Việt.

### 2. Giải pháp
- **Mô hình:** `vinai/bartpho-syllable`
- **Dữ liệu:** VnExpress news (~3000 mẫu).
- **Kết quả:** ROUGE-1, ROUGE-2, ROUGE-L đạt được [Nhập số liệu].

---

## 💻 Hướng dẫn chạy Giao diện (Demo)
Giao diện được xây dựng bằng Gradio tích hợp cả 3 mô hình AI.

1. Cài đặt thư viện:
`pip install -r requirements.txt`

2. Tải checkpoints từ link HuggingFace phía trên và đặt vào thư mục gốc.

3. Khởi chạy ứng dụng:
`python app.py`
