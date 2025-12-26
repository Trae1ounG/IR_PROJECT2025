"""
简化版训练脚本 - 更容易理解和使用
升级：支持BGE-M3模型和难负样本挖掘
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import pickle
import csv
import random


class SimpleBiEncoderDataset(Dataset):
    """简化的训练数据集 - 支持难负样本挖掘"""
    def __init__(self, triples_path, passages, qa_mapping, tokenizer, max_length=256,
                 num_hard_negatives=2, use_bge=False):
        self.passages = passages
        self.qa_mapping = qa_mapping
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_hard_negatives = num_hard_negatives
        self.use_bge = use_bge  # 是否使用BGE模型

        # 加载训练三元组
        self.triples = []
        with open(triples_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # 跳过header
            for row in reader:
                if len(row) >= 3:
                    qid, pos_pid, neg_pid = row
                    self.triples.append((int(qid), pos_pid, neg_pid))

        # 难负样本池 (每个query的难负样本列表)
        self.hard_negatives_pool = self._build_hard_negatives_pool()

    def _build_hard_negatives_pool(self):
        """构建难负样本池：为每个query收集多个随机负样本作为难负样本候选"""
        hard_neg_pool = {}
        all_pids = list(self.passages.keys())

        for qid, pos_pid, neg_pid in self.triples:
            if qid not in hard_neg_pool:
                # 随机选择一些负样本作为难负样本候选
                # 排除正样本
                candidates = [pid for pid in all_pids if pid != pos_pid]
                # 随机选择一些作为难负样本
                hard_neg_pool[qid] = random.sample(candidates, min(20, len(candidates)))

        return hard_neg_pool

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        qid, pos_pid, neg_pid = self.triples[idx]

        # 获取question
        question, _ = self.qa_mapping[qid]

        # BGE模型需要添加instruction前缀
        if self.use_bge:
            question = f"query: {question}"

        # 获取positive passage
        _, pos_text, pos_title = self.passages[pos_pid]
        pos_passage = f"{pos_title} {pos_text}"

        # BGE模型需要添加instruction前缀
        if self.use_bge:
            pos_passage = f"passage: {pos_passage}"

        # 获取原始negative passage
        _, neg_text, neg_title = self.passages[neg_pid]
        neg_passage = f"{neg_title} {neg_text}"

        if self.use_bge:
            neg_passage = f"passage: {neg_passage}"

        # 获取难负样本
        hard_neg_pids = self.hard_negatives_pool.get(qid, [])
        if len(hard_neg_pids) > 0:
            # 随机选择num_hard_negatives个难负样本
            selected_hard_negs = random.sample(hard_neg_pids,
                                               min(self.num_hard_negatives, len(hard_neg_pids)))
        else:
            selected_hard_negs = []

        # Tokenize question
        question_inputs = self.tokenizer(
            question,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize positive passage
        pos_inputs = self.tokenizer(
            pos_passage,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize negative passage (原始负样本)
        neg_inputs = self.tokenizer(
            neg_passage,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize hard negative passages (难负样本)
        hard_neg_inputs_list = []
        for hard_neg_pid in selected_hard_negs:
            _, hard_neg_text, hard_neg_title = self.passages[hard_neg_pid]
            hard_neg_passage = f"{hard_neg_title} {hard_neg_text}"

            if self.use_bge:
                hard_neg_passage = f"passage: {hard_neg_passage}"

            hard_neg_inputs = self.tokenizer(
                hard_neg_passage,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            hard_neg_inputs_list.append({
                'input_ids': hard_neg_inputs['input_ids'].squeeze(0),
                'attention_mask': hard_neg_inputs['attention_mask'].squeeze(0)
            })

        result = {
            'question_ids': question_inputs['input_ids'].squeeze(0),
            'question_mask': question_inputs['attention_mask'].squeeze(0),
            'pos_ids': pos_inputs['input_ids'].squeeze(0),
            'pos_mask': pos_inputs['attention_mask'].squeeze(0),
            'neg_ids': neg_inputs['input_ids'].squeeze(0),
            'neg_mask': neg_inputs['attention_mask'].squeeze(0),
        }

        # 添加难负样本
        for i, hard_neg_inputs in enumerate(hard_neg_inputs_list):
            result[f'hard_neg_{i}_ids'] = hard_neg_inputs['input_ids']
            result[f'hard_neg_{i}_mask'] = hard_neg_inputs['attention_mask']

        # 记录难负样本数量
        result['num_hard_negatives'] = len(hard_neg_inputs_list)

        return result


def collate_fn(batch):
    """整理batch - 支持难负样本"""
    question_ids = torch.stack([item['question_ids'] for item in batch])
    question_mask = torch.stack([item['question_mask'] for item in batch])
    pos_ids = torch.stack([item['pos_ids'] for item in batch])
    pos_mask = torch.stack([item['pos_mask'] for item in batch])
    neg_ids = torch.stack([item['neg_ids'] for item in batch])
    neg_mask = torch.stack([item['neg_mask'] for item in batch])

    result = {
        'question_ids': question_ids,
        'question_mask': question_mask,
        'pos_ids': pos_ids,
        'pos_mask': pos_mask,
        'neg_ids': neg_ids,
        'neg_mask': neg_mask,
    }

    # 处理难负样本
    max_hard_negatives = max([item.get('num_hard_negatives', 0) for item in batch])

    for i in range(max_hard_negatives):
        key_ids = f'hard_neg_{i}_ids'
        key_mask = f'hard_neg_{i}_mask'

        # 收集该位置的所有难负样本
        hard_neg_ids_list = []
        hard_neg_mask_list = []

        for item in batch:
            if i < item.get('num_hard_negatives', 0):
                hard_neg_ids_list.append(item[key_ids])
                hard_neg_mask_list.append(item[key_mask])
            else:
                # 用第0个负样本填充（保证batch维度一致）
                hard_neg_ids_list.append(item['neg_ids'])
                hard_neg_mask_list.append(item['neg_mask'])

        result[key_ids] = torch.stack(hard_neg_ids_list)
        result[key_mask] = torch.stack(hard_neg_mask_list)

    result['num_hard_negatives'] = max_hard_negatives

    return result


class BiEncoder(nn.Module):
    """双编码器模型 - 支持BGE-M3"""
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.question_encoder = AutoModel.from_pretrained(model_name)
        self.passage_encoder = AutoModel.from_pretrained(model_name)

    def forward(self, question_ids, question_mask, passage_ids, passage_mask):
        q_outputs = self.question_encoder(input_ids=question_ids, attention_mask=question_mask)
        p_outputs = self.passage_encoder(input_ids=passage_ids, attention_mask=passage_mask)

        q_vec = q_outputs.last_hidden_state[:, 0, :]  # [CLS]
        p_vec = p_outputs.last_hidden_state[:, 0, :]

        return q_vec, p_vec


def contrastive_loss_with_hard_negatives(q_vec, pos_vec, neg_vec_list, temperature=0.05):
    """对比学习损失 - 支持多个难负样本"""
    # 归一化
    q_vec = nn.functional.normalize(q_vec, dim=-1)
    pos_vec = nn.functional.normalize(pos_vec, dim=-1)

    # 计算正样本相似度
    pos_sim = torch.sum(q_vec * pos_vec, dim=-1, keepdim=True) / temperature  # [batch_size, 1]

    # 计算所有负样本的相似度
    neg_sims = []
    for neg_vec in neg_vec_list:
        neg_vec = nn.functional.normalize(neg_vec, dim=-1)
        neg_sim = torch.sum(q_vec * neg_vec, dim=-1, keepdim=True) / temperature  # [batch_size, 1]
        neg_sims.append(neg_sim)

    # 拼接所有相似度分数 [batch_size, 1 + num_negatives]
    all_sims = torch.cat([pos_sim] + neg_sims, dim=1)

    # 标签：正样本在第0列
    labels = torch.zeros(all_sims.size(0), dtype=torch.long).to(all_sims.device)

    # 使用交叉熵损失
    loss = nn.functional.cross_entropy(all_sims, labels)

    return loss


def compute_hit_at_k(model, tokenizer, questions, answers, passages, device, k_list=[1, 10, 100],
                     use_bge=False):
    """计算Hit@k"""
    model.eval()

    # 1. 编码所有passages
    passage_texts = []
    passage_ids = []
    for pid, (pid_raw, text, title) in passages.items():
        passage_text = f"{title} {text}"
        if use_bge:
            passage_text = f"passage: {passage_text}"
        passage_texts.append(passage_text)
        passage_ids.append(pid)

    print("Encoding passages...")
    passage_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(passage_texts), 32)):
            batch = passage_texts[i:i+32]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model.passage_encoder(**inputs)
            embs = outputs.last_hidden_state[:, 0, :]
            embs = nn.functional.normalize(embs, dim=-1)
            passage_embs.append(embs.cpu())

    passage_embs = torch.cat(passage_embs, dim=0)

    # 2. 对每个问题检索
    hits = {k: 0 for k in k_list}

    print("Retrieving...")
    for question, ans_list in tqdm(zip(questions, answers), total=len(questions)):
        # BGE需要添加前缀
        if use_bge:
            question = f"query: {question}"

        # 编码query
        with torch.no_grad():
            inputs = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model.question_encoder(**inputs)
            q_emb = outputs.last_hidden_state[:, 0, :]
            q_emb = nn.functional.normalize(q_emb, dim=-1)

        # 计算相似度
        scores = torch.matmul(q_emb, passage_embs.t().to(device)).squeeze(0)
        _, top_indices = torch.topk(scores, k=100)

        # 检查命中
        for k in k_list:
            top_k_indices = top_indices[:k]
            for idx in top_k_indices:
                pid = passage_ids[idx]
                _, text, title = passages[pid]
                passage = f"{title} {text}".lower()

                hit = False
                for ans in ans_list:
                    if str(ans).lower() in passage:
                        hit = True
                        break

                if hit:
                    hits[k] += 1
                    break

    # 计算准确率
    n = len(questions)
    return {f'hit_at_{k}': hits[k] / n for k in k_list}


def main():
    # 配置
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ========== 模型配置 ==========
    # 可选: 'bert-base-uncased', 'BAAI/bge-m3', 'BAAI/bge-large-en-v1.5', 'intfloat/e5-large-v2'
    MODEL_NAME = '/netcache/huggingface/models/bge-m3'
    USE_BGE = 'bge' in MODEL_NAME.lower() or 'e5' in MODEL_NAME.lower()  # 自动检测是否使用BGE/E5

    # 设置使用GPU 6和7
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,7'

    BATCH_SIZE = 8  # BGE-M3较大，减小batch size
    EPOCHS = 3  # 增加训练轮数
    LR = 2e-5
    MAX_LENGTH = 512 if USE_BGE else 256  # BGE支持更长序列
    NUM_HARD_NEGATIVES = 2  # 每个样本的难负样本数量

    # 设备配置（使用cuda:0，实际对应物理GPU 6）
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Use BGE instruction: {USE_BGE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LR}")
    print(f"Max Length: {MAX_LENGTH}")
    print(f"Number of Hard Negatives: {NUM_HARD_NEGATIVES}")
    print("="*60)

    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")

    # 加载tokenizer
    print(f"\nLoading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 加载数据
    print("Loading data...")
    with open(f'{BASE_DIR}/processed/passages.pkl', 'rb') as f:
        passages = pickle.load(f)
    print(f"Loaded {len(passages)} passages")

    with open(f'{BASE_DIR}/processed/train_triples_qa.pkl', 'rb') as f:
        qa_mapping = pickle.load(f)
    print(f"Loaded {len(qa_mapping)} QA pairs")

    # 创建数据集和加载器
    print("\nCreating dataset with hard negative mining...")
    train_dataset = SimpleBiEncoderDataset(
        triples_path=f'{BASE_DIR}/processed/train_triples.tsv',
        passages=passages,
        qa_mapping=qa_mapping,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        num_hard_negatives=NUM_HARD_NEGATIVES,
        use_bge=USE_BGE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    print(f"Dataset size: {len(train_dataset)}")

    # 创建模型
    print(f"\nCreating model: {MODEL_NAME}...")
    model = BiEncoder(MODEL_NAME).to(DEVICE)

    # 优化器 - BGE推荐使用较小的学习率
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # 训练循环
    output_dir = f'{BASE_DIR}/output/retriever'
    os.makedirs(output_dir, exist_ok=True)

    best_hit_at_10 = 0

    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        # 训练
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Training")
        for batch in progress_bar:
            question_ids = batch['question_ids'].to(DEVICE)
            question_mask = batch['question_mask'].to(DEVICE)
            pos_ids = batch['pos_ids'].to(DEVICE)
            pos_mask = batch['pos_mask'].to(DEVICE)
            neg_ids = batch['neg_ids'].to(DEVICE)
            neg_mask = batch['neg_mask'].to(DEVICE)

            # 前向传播 - 编码正样本和原始负样本
            q_vec, pos_vec = model(question_ids, question_mask, pos_ids, pos_mask)
            _, neg_vec = model(question_ids, question_mask, neg_ids, neg_mask)

            # 收集所有负样本（原始负样本 + 难负样本）
            neg_vec_list = [neg_vec]

            num_hard_negs = batch.get('num_hard_negatives', 0)
            for i in range(num_hard_negs):
                hard_neg_ids = batch[f'hard_neg_{i}_ids'].to(DEVICE)
                hard_neg_mask = batch[f'hard_neg_{i}_mask'].to(DEVICE)
                _, hard_neg_vec = model(question_ids, question_mask, hard_neg_ids, hard_neg_mask)
                neg_vec_list.append(hard_neg_vec)

            # 计算损失（使用难负样本）
            loss = contrastive_loss_with_hard_negatives(q_vec, pos_vec, neg_vec_list)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        print(f"Average loss: {avg_loss:.4f}")

        # 验证（使用验证集，不使用测试集）
        print("\nValidating on dev set...")

        # 优先使用dev目录下的验证集
        dev_questions_path = f'{BASE_DIR}/processed/dev/dev_questions.pkl'
        dev_answers_path = f'{BASE_DIR}/processed/dev/dev_answers.pkl'

        try:
            with open(dev_questions_path, 'rb') as f:
                dev_questions = pickle.load(f)
            with open(dev_answers_path, 'rb') as f:
                dev_answers = pickle.load(f)
            print(f"Using dev set from processed/dev/ ({len(dev_questions)} samples)")
        except FileNotFoundError:
            # 如果没有dev目录，警告用户
            print("⚠️  Warning: Dev set not found in processed/dev/")
            print("   Please run: python preprocess_data.py")
            print("   Skipping validation for this epoch.")
            continue

        metrics = compute_hit_at_k(model, tokenizer, dev_questions, dev_answers, passages, DEVICE, use_bge=USE_BGE)
        print(f"Validation - Hit@1: {metrics['hit_at_1']:.4f}, Hit@10: {metrics['hit_at_10']:.4f}, Hit@100: {metrics['hit_at_100']:.4f}")

        # 保存最佳模型
        if metrics['hit_at_10'] > best_hit_at_10:
            best_hit_at_10 = metrics['hit_at_10']
            torch.save(model.state_dict(), f'{output_dir}/best_model.pt')
            print(f"✓ Saved best model with Hit@10: {best_hit_at_10:.4f}")

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best Hit@10: {best_hit_at_10:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
