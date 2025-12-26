#!/usr/bin/env python3
"""
数据预处理脚本：将WebQ数据集转换为训练所需的格式
"""
import json
import csv
import pickle
from typing import List, Tuple, Dict
import os
from tqdm import tqdm


def load_json(file_path: str) -> List[Dict]:
    """加载JSON格式的数据"""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_corpus(corpus_path: str) -> Dict[str, Tuple[str, str, str]]:
    """
    加载语料库
    返回: {passage_id: (id, text, title)}
    """
    print(f"Loading corpus from {corpus_path}...")
    passages = {}

    with open(corpus_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)  # 修改：不要忽略引号
        next(reader)  # 跳过header

        for row in tqdm(reader):
            if len(row) >= 3:
                pid, text, title = row[0], row[1], row[2]
                passages[pid] = (pid, text, title)

    print(f"Loaded {len(passages)} passages")
    return passages


def process_webq_train(json_path: str, passages: Dict[str, Tuple[str, str, str]],
                       output_path: str, max_samples: int = None):
    """
    处理WebQ训练数据，生成训练三元组
    格式: (question, positive_passage_id, negative_passage_id)
    """
    print(f"Processing WebQ train data from {json_path}...")

    data = load_json(json_path)
    triples = []
    qa_mapping = {}

    for idx, item in enumerate(tqdm(data)):
        question = item.get('question', '')
        answers = item.get('answers', [])

        if not answers or not question:
            continue

        # 使用已有的positive_ctxs和negative_ctxs
        positive_ctxs = item.get('positive_ctxs', [])
        negative_ctxs = item.get('negative_ctxs', [])

        # 如果没有正样本，跳过
        if not positive_ctxs:
            continue

        # 获取正样本PID
        for pos_ctx in positive_ctxs[:2]:  # 每个问题最多2个正样本
            pos_pid = pos_ctx.get('psg_id')

            if not pos_pid or pos_pid not in passages:
                continue

            # 获取负样本PID
            if negative_ctxs:
                neg_ctx = negative_ctxs[0]
                neg_pid = neg_ctx.get('psg_id')
            else:
                # 如果没有负样本，随机选择一个
                import random
                passage_keys = list(passages.keys())
                neg_pid = random.choice(passage_keys)

            if not neg_pid or neg_pid not in passages:
                continue

            triples.append({
                'qid': idx,
                'pos_pid': pos_pid,
                'neg_pid': neg_pid
            })

            qa_mapping[idx] = (question, answers)

        if max_samples and len(triples) >= max_samples:
            break

    # 保存为TSV格式
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['qid', 'pos_pid', 'neg_pid'])  # header
        for triple in triples:
            writer.writerow([triple['qid'], triple['pos_pid'], triple['neg_pid']])

    print(f"Saved {len(triples)} training triples to {output_path}")

    # 同时保存问题和答案映射
    qa_path = output_path.replace('.tsv', '_qa.pkl')
    with open(qa_path, 'wb') as f:
        pickle.dump(qa_mapping, f)

    return triples


def process_webq_test(csv_path: str, output_dir: str):
    """
    处理WebQ测试数据
    """
    print(f"Processing WebQ test data from {csv_path}...")

    questions = []
    answers_list = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')

        for row in tqdm(reader):
            if len(row) >= 2:
                question = row[0]
                answers = eval(row[1]) if isinstance(row[1], str) else row[1]
                questions.append(question)
                answers_list.append(answers)

    os.makedirs(output_dir, exist_ok=True)

    # 保存问题列表
    questions_path = os.path.join(output_dir, 'test_questions.pkl')
    with open(questions_path, 'wb') as f:
        pickle.dump(questions, f)

    # 保存答案列表
    answers_path = os.path.join(output_dir, 'test_answers.pkl')
    with open(answers_path, 'wb') as f:
        pickle.dump(answers_list, f)

    # 同时保存为文本格式方便查看
    txt_path = os.path.join(output_dir, 'test_questions.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        for i, (q, a) in enumerate(zip(questions, answers_list)):
            f.write(f"{i}\t{q}\t{a}\n")

    print(f"Saved {len(questions)} test questions to {output_dir}")

    return questions, answers_list


def process_webq_dev_json(json_path: str, output_dir: str):
    """
    处理WebQ验证数据（从JSON格式）
    """
    print(f"Processing WebQ dev data from {json_path}...")

    data = load_json(json_path)
    questions = []
    answers_list = []

    for item in tqdm(data):
        question = item.get('question', '')
        answers = item.get('answers', [])

        if question and answers:
            questions.append(question)
            answers_list.append(answers)

    os.makedirs(output_dir, exist_ok=True)

    # 保存问题列表
    questions_path = os.path.join(output_dir, 'dev_questions.pkl')
    with open(questions_path, 'wb') as f:
        pickle.dump(questions, f)

    # 保存答案列表
    answers_path = os.path.join(output_dir, 'dev_answers.pkl')
    with open(answers_path, 'wb') as f:
        pickle.dump(answers_list, f)

    print(f"Saved {len(questions)} dev questions to {output_dir}")

    return questions, answers_list


if __name__ == '__main__':
    # 设置路径（自动检测脚本所在目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir
    corpus_path = os.path.join(base_dir, 'corpus/wiki_webq_corpus.tsv')
    train_json = os.path.join(base_dir, 'datas/webq-train.json')
    dev_json = os.path.join(base_dir, 'datas/webq-dev.json')
    test_csv = os.path.join(base_dir, 'datas/webq-test.csv')

    output_dir = os.path.join(base_dir, 'processed')

    print(f"Working directory: {base_dir}")

    # 1. 加载语料库
    passages = load_corpus(corpus_path)

    # 保存passage映射（用于后续检索）
    passage_output = os.path.join(output_dir, 'passages.pkl')
    os.makedirs(os.path.dirname(passage_output), exist_ok=True)
    with open(passage_output, 'wb') as f:
        pickle.dump(passages, f)
    print(f"Saved passages to {passage_output}")

    # 2. 处理训练数据
    train_output = os.path.join(output_dir, 'train_triples.tsv')
    process_webq_train(train_json, passages, train_output, max_samples=None)

    # 3. 处理验证数据（使用webq-dev.json）
    dev_output_dir = os.path.join(output_dir, 'dev')
    os.makedirs(dev_output_dir, exist_ok=True)
    process_webq_dev_json(dev_json, dev_output_dir)

    # 4. 处理测试数据（仅用于最终评估）
    process_webq_test(test_csv, output_dir)

    print("\n" + "="*60)
    print("数据预处理完成！")
    print(f"输出目录: {output_dir}")
    print("="*60)
    print("\n数据集说明:")
    print(f"  - 训练集: {output_dir}/train_triples.tsv")
    print(f"  - 验证集: {output_dir}/dev/ (用于模型选择和超参数调优)")
    print(f"  - 测试集: {output_dir}/test_*.pkl (仅用于最终评估，训练时不可使用)")
    print("\n⚠️  重要提示:")
    print("  训练时只能使用验证集(dev)进行验证，严禁使用测试集(test)！")
    print("="*60)
