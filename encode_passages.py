"""
编码所有passages并保存到磁盘
只需要运行一次，之后可以直接加载使用
"""
import os
import torch
import pickle
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np

# 设置使用GPU 6和7
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'bert-base-uncased'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"Available GPUs: {torch.cuda.device_count()}")

# 加载tokenizer和模型
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
passage_encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

# 加载数据
print("Loading passages...")
with open(f'{BASE_DIR}/processed/passages.pkl', 'rb') as f:
    all_passages = pickle.load(f)

# 获取训练数据中实际需要的passage IDs
print("Extracting required passage IDs from training data...")
import csv
required_pids = set()
with open(f'{BASE_DIR}/processed/train_triples.tsv', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)  # 跳过header
    for row in reader:
        if len(row) >= 3:
            required_pids.add(row[1])  # pos_pid
            required_pids.add(row[2])  # neg_pid

print(f"Required passages: {len(required_pids)}")

# 只编码训练需要的passages
passages = {pid: all_passages[pid] for pid in required_pids if pid in all_passages}
print(f"Total passages to encode: {len(passages)}")

# 准备passage文本
passage_texts = []
passage_ids = []

for pid, (pid_raw, text, title) in passages.items():
    passage_texts.append(f"{title} {text}")
    passage_ids.append(pid)

# 编码所有passages
print("\nEncoding passages (this will take a while)...")
batch_size = 128  # 可以根据显存调整
all_embeddings = []

with torch.no_grad():
    for i in tqdm(range(0, len(passage_texts), batch_size)):
        batch_texts = passage_texts[i:i+batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(DEVICE)
        attention_mask = inputs['attention_mask'].to(DEVICE)

        # Encode
        outputs = passage_encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

        all_embeddings.append(embeddings.cpu())

        # 每1000批打印进度
        if (i // batch_size) % 1000 == 0:
            print(f"Processed {i}/{len(passage_texts)} passages")

# 合并所有embeddings
passage_embeddings = torch.cat(all_embeddings, dim=0)

print(f"\nEncoded shape: {passage_embeddings.shape}")
print(f"Memory usage: {passage_embeddings.element_size() * passage_embeddings.nelement() / 1024 / 1024:.2f} MB")

# 保存到文件
output_path = f'{BASE_DIR}/processed/passage_embeddings.pt'
print(f"\nSaving embeddings to {output_path}...")

# 保存embeddings和对应的IDs
torch.save({
    'embeddings': passage_embeddings,
    'passage_ids': passage_ids
}, output_path)
print(passage_ids)
print("✓ Embeddings saved successfully!")
print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

# 同时保存一个numpy格式用于快速加载
numpy_path = f'{BASE_DIR}/processed/passage_embeddings.npy'
print(f"\nSaving to numpy format: {numpy_path}...")
np.save(numpy_path, passage_embeddings.numpy())
print("✓ Numpy format saved!")

print("\n=== Encoding Complete ===")
print(f"Total passages encoded: {len(passage_ids)}")
print(f"Embedding dimension: {passage_embeddings.shape[1]}")
print("\nYou can now use these embeddings for training and testing!")
