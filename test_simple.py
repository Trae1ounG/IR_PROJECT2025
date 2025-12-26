"""
简化版测试脚本 - 在测试集上评估并生成结果文件
升级：支持BGE-M3模型
"""
import os
import sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle
import json
import csv


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


def check_answer_in_passage(answers, passage_text):
    """检查答案是否在passage中"""
    passage_lower = passage_text.lower()
    for ans in answers:
        if str(ans).lower() in passage_lower:
            return True
    return False


def evaluate_and_save_results(model, tokenizer, questions, answers, passages, device, output_dir,
                               use_bge=False):
    """评估并保存结果"""

    # 1. 编码所有passages
    print("Encoding passages...")
    passage_texts = []
    passage_ids = []

    for pid, (pid_raw, text, title) in passages.items():
        passage_text = f"{title} {text}"
        # BGE需要添加前缀
        if use_bge:
            passage_text = f"passage: {passage_text}"
        passage_texts.append(passage_text)
        passage_ids.append(pid)
    BATCH_SIZE=4
    passage_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(passage_texts), BATCH_SIZE)):
            batch = passage_texts[i:i+BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512 if use_bge else 256, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model.passage_encoder(**inputs)
            embs = outputs.last_hidden_state[:, 0, :]
            embs = nn.functional.normalize(embs, dim=-1)
            passage_embs.append(embs.cpu())

    passage_embs = torch.cat(passage_embs, dim=0)
    print(f"Encoded {len(passage_embs)} passages")

    # 2. 对每个问题检索并计算Hit@k
    print("\nRetrieving and evaluating...")

    results = []
    hit_at_1_count = 0
    hit_at_10_count = 0
    hit_at_100_count = 0

    for qid, (question, ans_list) in enumerate(tqdm(zip(questions, answers), total=len(questions))):
        # BGE需要添加前缀
        if use_bge:
            question_input = f"query: {question}"
        else:
            question_input = question

        # 编码query
        with torch.no_grad():
            inputs = tokenizer(question_input, padding=True, truncation=True, max_length=128, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model.question_encoder(**inputs)
            q_emb = outputs.last_hidden_state[:, 0, :]
            q_emb = nn.functional.normalize(q_emb, dim=-1)

        # 计算相似度并获取top-100
        scores = torch.matmul(q_emb, passage_embs.t().to(device)).squeeze(0)
        top_scores, top_indices = torch.topk(scores, k=100)

        # 转换为passage IDs
        top_pids = [passage_ids[idx.item()] for idx in top_indices]

        # 检查Hit@k
        hit_1 = 0
        hit_10 = 0
        hit_100 = 0

        # 检查前100个结果
        for rank, idx in enumerate(top_indices, start=1):
            pid = passage_ids[idx.item()]
            _, text, title = passages[pid]
            passage = f"{title} {text}"

            if check_answer_in_passage(ans_list, passage):
                if rank <= 1:
                    hit_1 = 1
                if rank <= 10:
                    hit_10 = 1
                if rank <= 100:
                    hit_100 = 1
                break  # 找到第一个命中的就停止

        hit_at_1_count += hit_1
        hit_at_10_count += hit_10
        hit_at_100_count += hit_100

        # 保存结果
        results.append({
            'id': qid,
            'question': question,
            'answers': ans_list,
            'hit_at_1': hit_1,
            'hit_at_10': hit_10,
            'hit_at_100': hit_100
        })

    # 3. 计算整体指标
    n = len(questions)
    metrics = {
        'hit_at_1': hit_at_1_count / n,
        'hit_at_10': hit_at_10_count / n,
        'hit_at_100': hit_at_100_count / n,
    }

    # 4. 保存结果
    os.makedirs(output_dir, exist_ok=True)

    # 保存指标
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_dir}/metrics.json")

    # 保存Hit@k结果（提交格式）
    with open(f'{output_dir}/hit_at_k.txt', 'w') as f:
        f.write("id\thit_at_1\thit_at_10\thit_at_100\n")
        for r in results:
            f.write(f"{r['id']}\t{r['hit_at_1']}\t{r['hit_at_10']}\t{r['hit_at_100']}\n")
    print(f"Saved hit@k results to {output_dir}/hit_at_k.txt")

    # 保存详细结果
    with open(f'{output_dir}/detailed_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed results to {output_dir}/detailed_results.json")

    # 打印结果
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Hit@1:   {metrics['hit_at_1']:.4f}")
    print(f"Hit@10:  {metrics['hit_at_10']:.4f}")
    print(f"Hit@100: {metrics['hit_at_100']:.4f}")
    print("="*50)

    return metrics


def main():
    # 配置
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ========== 模型配置 ==========
    # 可选: 'bert-base-uncased', 'BAAI/bge-m3', 'BAAI/bge-large-en-v1.5', 'intfloat/e5-large-v2'
    MODEL_NAME = '/netcache/huggingface/models/bge-m3'
    USE_BGE = 'bge' in MODEL_NAME.lower() or 'e5' in MODEL_NAME.lower()

    MODEL_PATH = f'{BASE_DIR}/output/retriever/best_model.pt'
    DEVICE = 'cuda:7' if torch.cuda.is_available() else 'cpu'  # 使用6号GPU

    print("="*60)
    print("Testing Configuration")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Use BGE instruction: {USE_BGE}")
    print(f"Device: {DEVICE}")
    print(f"Model Path: {MODEL_PATH}")
    print("="*60)

    # 加载tokenizer
    print(f"\nLoading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    model = BiEncoder(MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 加载数据
    print("Loading data...")
    with open(f'{BASE_DIR}/processed/passages.pkl', 'rb') as f:
        passages = pickle.load(f)
    print(f"Loaded {len(passages)} passages")

    with open(f'{BASE_DIR}/processed/test_questions.pkl', 'rb') as f:
        test_questions = pickle.load(f)
    with open(f'{BASE_DIR}/processed/test_answers.pkl', 'rb') as f:
        test_answers = pickle.load(f)
    print(f"Loaded {len(test_questions)} test questions")

    # 评估
    output_dir = f'{BASE_DIR}/output/test_results'
    metrics = evaluate_and_save_results(
        model=model,
        tokenizer=tokenizer,
        questions=test_questions,
        answers=test_answers,
        passages=passages,
        device=DEVICE,
        output_dir=output_dir,
        use_bge=USE_BGE
    )

    print("\nEvaluation completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
