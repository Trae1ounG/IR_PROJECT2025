"""
检索数据集类：用于训练和验证
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import random
from typing import Dict, List, Tuple


class RetrievalDataset(Dataset):
    """
    检索训练数据集
    """
    def __init__(self, data_path: str, passages: Dict, tokenizer,
                 max_length: int = 512, is_train: bool = True):
        """
        Args:
            data_path: 数据文件路径
            passages: passage字典 {pid: (pid, text, title)}
            tokenizer: BERT tokenizer
            max_length: 最大序列长度
            is_train: 是否为训练模式
        """
        self.passages = passages
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

        # 加载数据
        self.data = []
        self.load_data(data_path)

    def load_data(self, data_path: str):
        """加载数据"""
        print(f"Loading data from {data_path}...")

        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                # {qid: (question, answers)}
                for qid, (question, answers) in data.items():
                    self.data.append({
                        'qid': qid,
                        'question': question,
                        'answers': answers
                    })
            else:
                self.data = data

        elif data_path.endswith('.tsv'):
            import csv
            with open(data_path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                next(reader)  # 跳过header

                for row in reader:
                    if len(row) >= 3:
                        qid, pos_pid, neg_pid = row[0], row[1], row[2]
                        self.data.append({
                            'qid': int(qid),
                            'pos_pid': pos_pid,
                            'neg_pid': neg_pid
                        })

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.is_train and 'pos_pid' in item:
            # 训练模式：返回query-positive pair和query-negative pair
            pos_pid = item['pos_pid']
            neg_pid = item['neg_pid']

            # 从passages中获取文本
            if pos_pid in self.passages and neg_pid in self.passages:
                _, pos_text, pos_title = self.passages[pos_pid]
                _, neg_text, neg_title = self.passages[neg_pid]

                # 组合title和text
                pos_passage = f"{pos_title} {pos_text}"
                neg_passage = f"{neg_title} {neg_text}"

                return {
                    'qid': item['qid'],
                    'pos_pid': pos_pid,
                    'neg_pid': neg_pid,
                    'pos_passage': pos_passage,
                    'neg_passage': neg_passage
                }
        else:
            # 测试/验证模式
            question = item['question']
            answers = item['answers']

            return {
                'qid': item['qid'],
                'question': question,
                'answers': answers
            }


class RetrievalCollator:
    """
    数据整理函数
    """
    def __init__(self, tokenizer, max_length: int = 512, is_train: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

    def __call__(self, batch: list):
        """
        整理batch数据
        """
        if self.is_train:
            # 训练模式
            queries = []
            pos_passages = []
            neg_passages = []

            for item in batch:
                # 获取对应的question（这里简化处理，实际需要从qid映射）
                # 为简化，我们假设每个item都有对应的question
                queries.append("query")  # 实际应该从映射中获取
                pos_passages.append(item['pos_passage'])
                neg_passages.append(item['neg_passage'])

            # Tokenize
            query_inputs = self.tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            pos_inputs = self.tokenizer(
                pos_passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            neg_inputs = self.tokenizer(
                neg_passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            return {
                'query_input_ids': query_inputs['input_ids'],
                'query_attention_mask': query_inputs['attention_mask'],
                'pos_input_ids': pos_inputs['input_ids'],
                'pos_attention_mask': pos_inputs['attention_mask'],
                'neg_input_ids': neg_inputs['input_ids'],
                'neg_attention_mask': neg_inputs['attention_mask'],
            }
        else:
            # 测试/验证模式
            questions = [item['question'] for item in batch]
            qids = [item['qid'] for item in batch]
            answers = [item['answers'] for item in batch]

            query_inputs = self.tokenizer(
                questions,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            return {
                'query_input_ids': query_inputs['input_ids'],
                'query_attention_mask': query_inputs['attention_mask'],
                'qids': qids,
                'answers': answers
            }


def create_data_loader(data_path: str, passages: Dict, tokenizer,
                       batch_size: int = 16, max_length: int = 512,
                       is_train: bool = True, num_workers: int = 0):
    """
    创建数据加载器
    """
    dataset = RetrievalDataset(
        data_path=data_path,
        passages=passages,
        tokenizer=tokenizer,
        max_length=max_length,
        is_train=is_train
    )

    collator = RetrievalCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        is_train=is_train
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        collate_fn=collator,
        num_workers=num_workers
    )

    return data_loader


if __name__ == '__main__':
    from transformers import AutoTokenizer

    # 测试
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # 模拟passages
    passages = {
        '1': ('1', 'test passage 1', 'Test 1'),
        '2': ('2', 'test passage 2', 'Test 2'),
    }

    # 创建数据集
    dataset = RetrievalDataset(
        data_path='test.pkl',
        passages=passages,
        tokenizer=tokenizer
    )

    print(f"Dataset size: {len(dataset)}")
