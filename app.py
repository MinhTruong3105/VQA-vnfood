!pip install deep-translator
!pip install deep-translator gradio peft transformers bitsandbytes accelerate

import os
import re
import math
import pickle
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import (
    AutoModel, AutoTokenizer, 
    Blip2Processor, Blip2ForConditionalGeneration,
    BitsAndBytesConfig
)
from peft import PeftModel
from deep_translator import GoogleTranslator
import gradio as gr
from PIL import Image

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ THIẾT BỊ
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Đang chạy trên thiết bị: {device}")

MODEL_DIR = "/kaggle/input/datasets/truongminh3105/models"
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.pkl")
MODEL_A1_PATH = os.path.join(MODEL_DIR, "vqa_model_A1_vit_phobert.pth")
MODEL_A2_PATH = os.path.join(MODEL_DIR, "vqa_model_A2_parallel.pth")
# Thư mục chứa adapter_config.json và adapter_model.safetensors của BLIP-2
ADAPTER_DIR = MODEL_DIR 

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict

# ---------------------------------------------------------
# MODULE DỊCH THUẬT (Cho BLIP-2 vì model gốc dùng tiếng Anh)
# ---------------------------------------------------------
class VietnameseTranslator:
    def __init__(self):
        self.vi_to_en = GoogleTranslator(source='vi', target='en')
        self.en_to_vi = GoogleTranslator(source='en', target='vi')

    def translate_vi2en(self, text):
        try: return self.vi_to_en.translate(text)
        except: return text

    def translate_en2vi(self, text):
        try: return self.en_to_vi.translate(text)
        except: return text

translator = VietnameseTranslator()

# ---------------------------------------------------------
# CLASS VOCABULARY (Dùng cho A1 và A2)
# ---------------------------------------------------------
class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx = 4
    def __len__(self): return len(self.itos)
    @staticmethod
    def tokenizer(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = self.idx
                self.itos[self.idx] = word
                self.idx += 1

# Tải Vocab
try:
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
except:
    vocab = Vocabulary() # Dummy

vqa_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ==============================================================================
# 2. KIẾN TRÚC MÔ HÌNH (A1, A2, B2)
# ==============================================================================

# ==============================================================================
# 3. KHỞI TẠO VÀ TẢI TRỌNG SỐ
# ==============================================================================

# --- MODEL B2 (BLIP-2 + LoRA) ---
blip2_model, blip2_processor = None, None
try:
    BLIP2_NAME = "Salesforce/blip2-opt-2.7b"
    print(" Đang tải BLIP-2 base (8-bit)...")
    
    q_config = BitsAndBytesConfig(load_in_8bit=True)
    blip2_processor = Blip2Processor.from_pretrained(BLIP2_NAME)
    
    base_blip2 = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_NAME, quantization_config=q_config, device_map="auto"
    )
    
    # Nạp LoRA adapter
    blip2_model = PeftModel.from_pretrained(base_blip2, ADAPTER_DIR)
    blip2_model.eval()
    print(" Đã tải xong BLIP-2 + LoRA Adapter.")
except Exception as e:
    print(f" Lỗi tải BLIP-2: {e}")

# --- MODEL A1 & A2 (Tải như cũ) ---
# [Giả định model_A1 và model_A2 đã được khởi tạo và load state_dict]

# ==============================================================================
# 4. HÀM INFERENCE
# ==============================================================================

def infer_vqa_b2(image_pil, question_vi):
    """Pipeline: Vi -> En -> BLIP-2 -> En -> Vi"""
    if image_pil is None or not question_vi: return "Thiếu dữ liệu."
    if blip2_model is None: return "Model B2 chưa được load."

    # 1. Dịch câu hỏi sang tiếng Anh
    question_en = translator.translate_vi2en(question_vi)
    prompt = f"Question: {question_en}\nAnswer:"

    # 2. Xử lý ảnh và text qua processor
    inputs = blip2_processor(images=image_pil, text=prompt, return_tensors="pt").to(device, torch.float16)

    # 3. Generate
    with torch.no_grad():
        generated_ids = blip2_model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5
        )
    
    # 4. Decode và dịch lại tiếng Việt
    answer_en = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if "Answer:" in answer_en:
        answer_en = answer_en.split("Answer:")[-1].strip()
        
    answer_vi = translator.en_to_vi.translate(answer_en)
    return f"Tiếng Anh: {answer_en}\nTiếng Việt: {answer_vi}"

# [Hàm infer_vqa_a1 và infer_vqa_a2 giữ nguyên]

# ==============================================================================
# 5. GIAO DIỆN GRADIO UI CẬP NHẬT
# ==============================================================================

with gr.Blocks(theme=gr.themes.Soft(), title="VQA Multimodal System") as app:
    gr.Markdown("<h1 style='text-align: center;'>Hệ thống VQA Đa Mô Hình</h1>")
    
    with gr.Tabs():
        # TAB 1: BLIP-2 FINE-TUNED
        with gr.TabItem("1. VQA B2 (BLIP-2 + LoRA)"):
            gr.Markdown("### Kiến trúc Large Vision-Language Model (SFT với LoRA)")
            with gr.Row():
                with gr.Column():
                    img_b2 = gr.Image(type="pil", label="Tải ảnh")
                    q_b2 = gr.Textbox(label="Câu hỏi (Tiếng Việt)")
                    btn_b2 = gr.Button("Dự đoán B2", variant="primary")
                with gr.Column():
                    out_b2 = gr.Textbox(label="Kết quả BLIP-2", lines=5)
            btn_b2.click(fn=infer_vqa_b2, inputs=[img_b2, q_b2], outputs=out_b2)

        # TAB 2: VQA A1
        with gr.TabItem("2. VQA A1 (LSTM Decoder)"):
            # [Giao diện VQA A1 cũ...]
            with gr.Row():
                with gr.Column():
                    img_a1 = gr.Image(type="pil", label="Tải ảnh")
                    q_a1 = gr.Textbox(label="Câu hỏi (Tiếng Việt)")
                    btn_a1 = gr.Button("Dự đoán A1", variant="primary")
                with gr.Column():
                    out_a1 = gr.Textbox(label="Kết quả A1", lines=5)
            btn_a1.click(fn=infer_vqa_a1, inputs=[img_a1, q_a1], outputs=out_a1)

        # TAB 3: VQA A2
        with gr.TabItem("3. VQA A2 (Transformer Decoder)"):
            # [Giao diện VQA A2 cũ...]
            with gr.Row():
                with gr.Column():
                    img_a2 = gr.Image(type="pil", label="Tải ảnh")
                    q_a2 = gr.Textbox(label="Câu hỏi (Tiếng Việt)")
                    btn_a2 = gr.Button("Dự đoán A2", variant="primary")
                with gr.Column():
                    out_a2 = gr.Textbox(label="Kết quả A2", lines=5)
            btn_a2.click(fn=infer_vqa_a2, inputs=[img_a2, q_a2], outputs=out_a2)

if __name__ == "__main__":
    app.launch(debug=True, share=True)