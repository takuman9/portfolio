import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

# ベクトル化モデルのロード
model = SentenceTransformer('all-MiniLM-L6-v2')

# ディレクトリ設定
data_dir = "data"
embed_dir = "embed"
text_data_dir = "text_data"  # テキストを保存するディレクトリ
os.makedirs(embed_dir, exist_ok=True)
os.makedirs(text_data_dir, exist_ok=True)

# 全PDFファイルのテキストを結合
combined_text = ""

# PDFファイルの処理
for file_name in os.listdir(data_dir):
    if file_name.endswith(".pdf"):
        file_path = os.path.join(data_dir, file_name)
        reader = PdfReader(file_path)
        
        # 各PDFファイルの全ページのテキストを結合
        print(f"Processing file: {file_name}")
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            print(f"----- Page {i} -----")
            print(page_text)  # 各ページのテキストをデバッグ出力
            combined_text += page_text + "\n"  # 各ページのテキストを改行で区切って結合

# 結合されたテキストを保存
combined_text_file_path = os.path.join(text_data_dir, "combined_text.txt")
try:
    with open(combined_text_file_path, "w", encoding="utf-8") as text_file:
        text_file.write(combined_text)
    print(f"Saved combined text to {combined_text_file_path}")
except Exception as e:
    print(f"Error saving combined text: {e}")

# 結合されたテキストをベクトル化
try:
    embeddings = model.encode(combined_text, convert_to_numpy=True)
except Exception as e:
    print(f"Error during embedding for combined text: {e}")

# ベクトルを保存
combined_embed_file_path = os.path.join(embed_dir, "combined_embeddings.npy")
try:
    np.save(combined_embed_file_path, embeddings)
    print(f"Saved embeddings for combined text to {combined_embed_file_path}")
except Exception as e:
    print(f"Error saving embeddings for combined text: {e}")