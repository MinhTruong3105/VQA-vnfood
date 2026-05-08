

# Cài đặt các thư viện cần thiết
!pip install -q transformers==4.40.0 accelerate bitsandbytes
!pip install -q deep-translator
!pip install -q bert-score rouge-score nltk
!pip install -q Pillow tqdm pandas
!pip install -U transformers bert-score accelerate bitsandbytes rouge-score

print("✅ Đã cài đặt xong tất cả thư viện!")



import os
import pandas as pd
import torch
import json
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ── Đường dẫn (dựa trên lối tắt dataset) ────────────────────────────────────
BASE_DIR    = '/content/drive/MyDrive'
DATA_DIR    = os.path.join(BASE_DIR, 'dataset')
ANNO_DIR    = os.path.join(DATA_DIR, 'annotations')

RESULTS_DIR = os.path.join(BASE_DIR, 'results_B1')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Đường dẫn đến file JSON test
TEST_JSON   = os.path.join(ANNO_DIR, 'test.json')

# Kiểm tra thư mục
if os.path.exists(DATA_DIR):
    print(f" Đã kết nối thành công với thư mục dataset tại: {DATA_DIR}")
else:
    print(f" Không tìm thấy thư mục dataset. Vui lòng kiểm tra lại lối tắt.")

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def load_and_flatten_json(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Không tìm thấy file: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    flat_data = []
    for item in data:
        img_path = item['image_path'] # VD: "Images/test/..."

        for qa in item['qa_pairs']:
            flat_data.append({
                'image_path': img_path,
                'question': qa['question'],
                'answer': qa['answer'],
                'q_type': qa.get('type', 'unknown') # Lấy nhãn loại câu hỏi từ data
            })
    return pd.DataFrame(flat_data)

test_df = load_and_flatten_json(TEST_JSON)

print(f"Tổng số cặp câu hỏi-trả lời trong tập Test: {len(test_df)}")
print("\nMẫu dữ liệu (5 dòng đầu):")
display(test_df.head())
print("\n B1 (Zero-shot) chỉ cần tập Test để đánh giá — không cần Train/Val.")


from deep_translator import GoogleTranslator

class VietnameseTranslator:
    def __init__(self):
        self.vi_to_en = GoogleTranslator(source='vi', target='en')
        self.en_to_vi = GoogleTranslator(source='en', target='vi')
        self._cache_vi2en = {}
        self._cache_en2vi = {}

    def translate_question(self, text_vi: str) -> str:
        if text_vi in self._cache_vi2en: return self._cache_vi2en[text_vi]
        try:
            result = self.vi_to_en.translate(text_vi)
            self._cache_vi2en[text_vi] = result
            return result
        except: return text_vi

    def translate_answer(self, text_en: str) -> str:
        if text_en in self._cache_en2vi: return self._cache_en2vi[text_en]
        try:
            result = self.en_to_vi.translate(text_en)
            self._cache_en2vi[text_en] = result
            return result
        except: return text_en

translator = VietnameseTranslator()
print(" Đã khởi tạo module dịch thuật Việt ↔ Anh.")


from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig

MODEL_NAME = "Salesforce/blip2-opt-2.7b"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

print(f" Đang tải processor và mô hình: {MODEL_NAME} ...")
processor = Blip2Processor.from_pretrained(MODEL_NAME)
blip2_model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
blip2_model.eval()
print(f"\n Đã tải xong mô hình BLIP-2 ở chế độ 8-bit!")


def infer_blip2_zeroshot(image_pil: Image.Image, question_vi: str, return_english: bool = False, max_new_tokens: int = 30) -> str:
    question_en = translator.translate_question(question_vi)
    prompt = f"Question: {question_en}\nAnswer:"
    inputs = processor(images=image_pil, text=prompt, return_tensors="pt").to(device, torch.float16)

    with torch.no_grad():
        generated_ids = blip2_model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=5,
            length_penalty=1.0, repetition_penalty=1.5, early_stopping=True,
        )

    answer_en = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if "Answer:" in answer_en: answer_en = answer_en.split("Answer:")[-1].strip()

    answer_vi = translator.translate_answer(answer_en)
    return (answer_vi, answer_en) if return_english else answer_vi

print(" Kiểm tra inference với 3 mẫu ngẫu nhiên từ tập test:\n")
sample_rows = test_df.sample(3, random_state=99)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (_, row) in zip(axes, sample_rows.iterrows()):
    img_path = os.path.join(DATA_DIR, row['image_path'])
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        img = Image.new('RGB', (224, 224), (200, 200, 200))

    answer_vi, answer_en = infer_blip2_zeroshot(img, row['question'], return_english=True)

    ax.imshow(img)
    ax.axis('off')
    ax.set_title(
        f" {row['question']}\n GT  : {row['answer']}\n🤖 B1  : {answer_vi}\n(EN): {answer_en}",
        fontsize=8, loc='left', wrap=True
    )

plt.suptitle("B1 Zero-shot — BLIP-2 (Salesforce/blip2-opt-2.7b)", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()


import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def evaluate_b1_zeroshot(test_df, data_dir, save_dir, max_new_tokens=20):
    predictions_vi, predictions_en, references_vi = [], [], []
    questions_vi, image_paths, q_types = [], [], []

    print("🔮 Đang chạy inference zero-shot trên tập Test...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="B1 Inference"):
        img_path = os.path.join(data_dir, row['image_path'])

        try: img = Image.open(img_path).convert('RGB')
        except: img = Image.new('RGB', (224, 224), (200, 200, 200))

        pred_vi, pred_en = infer_blip2_zeroshot(img, str(row['question']), return_english=True, max_new_tokens=max_new_tokens)

        predictions_vi.append(pred_vi if pred_vi else "")
        predictions_en.append(pred_en if pred_en else "")
        references_vi.append(str(row['answer']))
        questions_vi.append(str(row['question']))
        image_paths.append(row['image_path'])
        q_types.append(row['q_type'])

    # Lưu DataFrame
    results_df = pd.DataFrame({
        'image': image_paths,
        'question_vi': questions_vi,
        'reference_vi': references_vi,
        'prediction_vi': predictions_vi,
        'prediction_en': predictions_en,
        'q_type': q_types
    })
    results_df.to_csv(os.path.join(save_dir, 'predictions_B1.csv'), index=False, encoding='utf-8-sig')

    # Metrics
    chencherry  = SmoothingFunction()
    nltk_preds  = [p.split() for p in predictions_vi]
    nltk_refs   = [[r.split()] for r in references_vi]

    bleu1 = corpus_bleu(nltk_refs, nltk_preds, weights=(1,0,0,0), smoothing_function=chencherry.method1)
    bleu4 = corpus_bleu(nltk_refs, nltk_preds, weights=(0.25,0.25,0.25,0.25), smoothing_function=chencherry.method1)

    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rl_scores = [rouge.score(ref, pred)['rougeL'].fmeasure for pred, ref in zip(predictions_vi, references_vi)]
    avg_rl = sum(rl_scores) / len(rl_scores) if rl_scores else 0

    meteor_scores = [meteor_score([ref.split()], pred.split()) for pred, ref in zip(predictions_vi, references_vi)]
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0

    def normalize(s): return str(s).lower().strip()
    exact_match = sum(normalize(p) == normalize(r) for p, r in zip(predictions_vi, references_vi)) / len(references_vi)

    P, R, F1 = bert_score_fn(predictions_vi, references_vi, lang="vi", model_type="bert-base-multilingual-cased", verbose=False)
    avg_bert_f1 = F1.mean().item()

    metrics = {
        'model': 'B1 Zero-shot', 'bleu1': round(bleu1*100, 2), 'bleu4': round(bleu4*100, 2),
        'rougeL': round(avg_rl*100, 2), 'meteor': round(avg_meteor*100, 2),
        'vqa_accuracy': round(exact_match*100, 2), 'bertscore_f1': round(avg_bert_f1*100, 2),
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, 'metrics_B1.csv'), index=False)

    print("\n" + "=" * 45)
    for k, v in metrics.items(): print(f"{k:<15} : {v}")
    print("=" * 45)

    return metrics, results_df

b1_metrics, b1_results = evaluate_b1_zeroshot(test_df, DATA_DIR, RESULTS_DIR)


import numpy as np

# Plot Radar & Bar Chart (Giữ nguyên logic của bạn)
metric_names  = ['BLEU-1', 'BLEU-4', 'ROUGE-L', 'METEOR', 'VQA Acc.', 'BERTScore']
metric_values = [b1_metrics['bleu1'], b1_metrics['bleu4'], b1_metrics['rougeL'], b1_metrics['meteor'], b1_metrics['vqa_accuracy'], b1_metrics['bertscore_f1']]
colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#F7B731']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars = axes[0].bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=0.8)
axes[0].set_ylim(0, 100); axes[0].set_ylabel('Score (%)'); axes[0].set_title('B1 Zero-shot — Điểm Metric')
for bar, val in zip(bars, metric_values): axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', ha='center')

N = len(metric_names)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
vals = metric_values + [metric_values[0]]
ax_radar = plt.subplot(122, polar=True)
ax_radar.plot(angles, vals, 'o-', linewidth=2, color='#4ECDC4')
ax_radar.fill(angles, vals, alpha=0.25, color='#4ECDC4')
ax_radar.set_xticks(angles[:-1]); ax_radar.set_xticklabels(metric_names)
ax_radar.set_ylim(0, 100); ax_radar.set_title('B1 Zero-shot — Biểu đồ Radar')

plt.tight_layout()
plt.show()

# Hiển thị mẫu đúng sai
def normalize(s): return str(s).lower().strip()
b1_results['correct'] = b1_results.apply(lambda r: normalize(r['prediction_vi']) == normalize(r['reference_vi']), axis=1)

def plot_samples(rows, title, color):
    fig, axes = plt.subplots(1, min(len(rows), 3), figsize=(15, 5))
    if len(rows) == 1: axes = [axes]
    for ax, (_, row) in zip(axes, rows.iterrows()):
        img_path = os.path.join(DATA_DIR, row['image'])
        try: img = Image.open(img_path).convert('RGB')
        except: img = Image.new('RGB', (224, 224), (200, 200, 200))
        ax.imshow(img); ax.axis('off')
        ax.set_title(f"❓ {row['question_vi']}\n GT : {row['reference_vi']}\n B1 : {row['prediction_vi']}", fontsize=8, loc='left', color=color)
    plt.suptitle(title, fontsize=13, color=color); plt.tight_layout(); plt.show()

plot_samples(b1_results[b1_results['correct']].head(3), " Dự đoán ĐÚNG", "green")
plot_samples(b1_results[~b1_results['correct']].head(3), " Dự đoán SAI", "red")

type_stats = b1_results.groupby('q_type').agg(
    total=('correct', 'count'),
    correct=('correct', 'sum')
).assign(accuracy=lambda x: x['correct'] / x['total'] * 100).sort_values('accuracy', ascending=False)

print("Accuracy theo loại câu hỏi (Ground Truth từ JSON):")
print("-" * 45)
print(f"{'Loại câu hỏi':<15} {'Tổng':>8} {'Đúng':>8} {'Accuracy':>12}")
print("-" * 45)
for q_type, row in type_stats.iterrows():
    print(f"{q_type:<15} {int(row['total']):>8} {int(row['correct']):>8} {row['accuracy']:>11.1f}%")
print("-" * 45)

# Vẽ biểu đồ
fig, ax = plt.subplots(figsize=(8, 4))
colors_type = ['#2ECC71' if v >= 50 else '#E74C3C' for v in type_stats['accuracy']]
ax.barh(type_stats.index, type_stats['accuracy'], color=colors_type, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Accuracy (%)', fontsize=11)
ax.set_title('B1 Zero-shot: Accuracy theo Loại câu hỏi', fontsize=13)
ax.axvline(x=50, color='gray', linestyle='--', alpha=0.6, label='50%')
for i, (idx, row) in enumerate(type_stats.iterrows()):
    ax.text(row['accuracy'] + 0.5, i, f"{row['accuracy']:.1f}%", va='center', fontsize=9)
ax.set_xlim(0, 110)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# ── Phân tích lỗi theo loại câu hỏi ─────────────────────────────────────────
# Phân loại câu hỏi theo keyword
def classify_question(q):
    q_lower = q.lower()
    if any(kw in q_lower for kw in ['có', 'không', 'phải', 'có phải']):
        return 'Yes/No'
    elif any(kw in q_lower for kw in ['bao nhiêu', 'mấy', 'số lượng']):
        return 'Đếm số'
    elif any(kw in q_lower for kw in ['màu', 'màu sắc']):
        return 'Màu sắc'
    elif any(kw in q_lower for kw in ['gì', 'là gì', 'đây là']):
        return 'Nhận dạng'
    elif any(kw in q_lower for kw in ['ở đâu', 'vị trí', 'bên']):
        return 'Không gian'
    else:
        return 'Khác'

b1_results['q_type'] = b1_results['question_vi'].apply(classify_question)

# Tính accuracy theo loại câu hỏi
type_stats = b1_results.groupby('q_type').agg(
    total=('correct', 'count'),
    correct=('correct', 'sum')
).assign(accuracy=lambda x: x['correct'] / x['total'] * 100).sort_values('accuracy', ascending=False)

print(" Accuracy theo loại câu hỏi (B1 Zero-shot):")
print("-" * 45)
print(f"{'Loại câu hỏi':<15} {'Tổng':>8} {'Đúng':>8} {'Accuracy':>12}")
print("-" * 45)
for q_type, row in type_stats.iterrows():
    print(f"{q_type:<15} {int(row['total']):>8} {int(row['correct']):>8} {row['accuracy']:>11.1f}%")
print("-" * 45)

# Vẽ biểu đồ
fig, ax = plt.subplots(figsize=(8, 4))
colors_type = ['#2ECC71' if v >= 50 else '#E74C3C' for v in type_stats['accuracy']]
ax.barh(type_stats.index, type_stats['accuracy'], color=colors_type, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Accuracy (%)', fontsize=11)
ax.set_title('B1 Zero-shot: Accuracy theo Loại câu hỏi', fontsize=13)
ax.axvline(x=50, color='gray', linestyle='--', alpha=0.6, label='50%')
for i, (idx, row) in enumerate(type_stats.iterrows()):
    ax.text(row['accuracy'] + 0.5, i, f"{row['accuracy']:.1f}%", va='center', fontsize=9)
ax.set_xlim(0, 110)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_by_qtype_B1.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── Bảng so sánh tổng hợp (placeholder — cập nhật sau khi có B2) ─────────────
comparison_data = {
    'Cấu hình':        ['A1 (LSTM Dec.)', 'A2 (Transformer Dec.)', 'B1 (Zero-shot)', 'B2 (Fine-tuned)'],
    'Mô tả':           [
        'ViT + PhoBERT + LSTM Decoder',
        'ViT + PhoBERT + Transformer Decoder',
        'BLIP-2 opt-2.7b (zero-shot)',
        'BLIP-2 opt-2.7b (fine-tuned)'
    ],
    'BLEU-1':          ['—', '—', b1_metrics['bleu1'], '—'],
    'ROUGE-L':         ['—', '—', b1_metrics['rougeL'], '—'],
    'METEOR':          ['—', '—', b1_metrics['meteor'], '—'],
    'VQA Accuracy':    ['—', '—', b1_metrics['vqa_accuracy'], '—'],
    'BERTScore F1':    ['—', '—', b1_metrics['bertscore_f1'], '—'],
}

comparison_df = pd.DataFrame(comparison_data)
print(" Bảng so sánh tổng hợp A1 / A2 / B1 / B2:")
print("   (Điền kết quả A1, A2, B2 vào cột tương ứng sau khi chạy xong các notebook)")
display(comparison_df)

# Lưu bảng so sánh
comparison_df.to_csv(
    os.path.join(RESULTS_DIR, 'comparison_table_all_models.csv'),
    index=False, encoding='utf-8-sig'
)
print("\n Đã lưu bảng so sánh.")


import io
from google.colab import files

def show_image(image_pil, title=""):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_pil)
    if title:
        ax.set_title(title, fontsize=12, pad=8)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

print("=" * 60)
print("     HỆ THỐNG HỎI-ĐÁP TRỰC QUAN (VQA) — MÔ HÌNH B1")
print("     BLIP-2 Zero-shot + Dịch Việt↔Anh")
print("=" * 60)
print("  Lệnh đặc biệt:")
print("      new   → upload ảnh mới")
print("      thoat → kết thúc chương trình")
print("=" * 60)

current_image = None

while True:
    if current_image is None:
        print("\n Tải lên ảnh để bắt đầu:")
        uploaded = files.upload()
        if not uploaded:
            print("  Không có file nào được tải lên. Thoát.")
            break
        filename = list(uploaded.keys())[0]
        try:
            current_image = Image.open(io.BytesIO(uploaded[filename])).convert('RGB')
        except Exception as e:
            print(f" Không thể đọc ảnh: {e}")
            continue
        show_image(current_image, title=f"📷 {filename}")

    question = input("\n Câu hỏi tiếng Việt (new = ảnh mới | thoat = thoát): ").strip()

    if question.lower() in ['thoat', 'exit', 'quit']:
        print("\n Đã dừng chương trình.")
        break

    if question.lower() == 'new':
        current_image = None
        print("-" * 60)
        continue

    if question == '':
        print("Vui lòng nhập câu hỏi.")
        continue

    answer_vi, answer_en = infer_blip2_zeroshot(
        current_image, question, return_english=True
    )
    q_en = translator.translate_question(question)

    print("-" * 60)
    print(f"   Câu hỏi (EN) : {q_en}")
    print(f"   Trả lời (EN) : {answer_en}")
    print(f"  🇻🇳 Trả lời (VI) : {answer_vi}")
    print("-" * 60)
