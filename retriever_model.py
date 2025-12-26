"""
基于BERT的双编码器检索模型
使用两个独立的BERT编码器分别编码query和passage
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BiEncoder(nn.Module):
    """
    双编码器模型：分别编码query和passage
    """
    def __init__(self, model_name: str = 'bert-base-uncased', dropout: float = 0.1):
        super(BiEncoder, self).__init__()

        self.question_encoder = AutoModel.from_pretrained(model_name)
        self.passage_encoder = AutoModel.from_pretrained(model_name)

        # 可以选择共享参数或不共享
        # self.passage_encoder = self.question_encoder

        self.dropout = nn.Dropout(dropout)

        # 获取隐藏层维度
        config = self.question_encoder.config
        self.hidden_size = config.hidden_size

    def forward(self, query_input_ids, query_attention_mask,
                passage_input_ids, passage_attention_mask):
        """
        前向传播
        Args:
            query_input_ids: [batch_size, seq_len]
            query_attention_mask: [batch_size, seq_len]
            passage_input_ids: [batch_size, seq_len]
            passage_attention_mask: [batch_size, seq_len]
        Returns:
            query_vec: [batch_size, hidden_size]
            passage_vec: [batch_size, hidden_size]
        """
        # 编码query
        query_outputs = self.question_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask
        )
        # 使用[CLS] token的表示
        query_vec = query_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        query_vec = self.dropout(query_vec)

        # 编码passage
        passage_outputs = self.passage_encoder(
            input_ids=passage_input_ids,
            attention_mask=passage_attention_mask
        )
        passage_vec = passage_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        passage_vec = self.dropout(passage_vec)

        return query_vec, passage_vec


class InfoNCELoss(nn.Module):
    """
    InfoNCE损失函数（对比学习损失）
    """
    def __init__(self, temperature: float = 0.05):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, query_vecs, passage_vecs):
        """
        计算InfoNCE损失
        Args:
            query_vecs: [batch_size, hidden_size]
            passage_vecs: [batch_size, hidden_size]
        Returns:
            loss: scalar
        """
        batch_size = query_vecs.size(0)

        # 归一化
        query_vecs = nn.functional.normalize(query_vecs, dim=-1)
        passage_vecs = nn.functional.normalize(passage_vecs, dim=-1)

        # 计算相似度矩阵 [batch_size, batch_size]
        similarity_matrix = torch.matmul(query_vecs, passage_vecs.t()) / self.temperature

        # 正样本在对角线上
        labels = torch.arange(batch_size).to(query_vecs.device)

        # 使用交叉熵损失
        loss = nn.functional.cross_entropy(similarity_matrix, labels)

        return loss


class Retriever:
    """
    检索器类：用于编码和检索
    """
    def __init__(self, model_path: str, tokenizer_name: str = 'bert-base-uncased',
                 max_length: int = 512, device: str = 'cuda'):
        self.device = device
        self.max_length = max_length

        # 加载模型
        self.model = BiEncoder(model_name=tokenizer_name).to(device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")

        self.model.eval()

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def encode_queries(self, queries: list, batch_size: int = 32):
        """
        编码查询
        Args:
            queries: list of strings
            batch_size: batch size
        Returns:
            embeddings: [num_queries, hidden_size]
        """
        self.model.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_queries,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)

                # 编码
                outputs = self.model.question_encoder(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                embeddings = nn.functional.normalize(embeddings, dim=-1)

                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def encode_passages(self, passages: list, batch_size: int = 32):
        """
        编码passage
        Args:
            passages: list of strings
            batch_size: batch size
        Returns:
            embeddings: [num_passages, hidden_size]
        """
        self.model.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(passages), batch_size):
                batch_passages = passages[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_passages,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)

                # 编码
                outputs = self.model.passage_encoder(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                embeddings = nn.functional.normalize(embeddings, dim=-1)

                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def retrieve(self, query: str, passage_embeddings: torch.Tensor,
                 top_k: int = 100, passage_texts: list = None):
        """
        检索top-k passages
        Args:
            query: query string
            passage_embeddings: [num_passages, hidden_size]
            top_k: 返回top-k结果
            passage_texts: passage文本列表（可选）
        Returns:
            top_indices: top-k的索引
            top_scores: top-k的相似度分数
        """
        # 编码query
        query_embedding = self.encode_queries([query])[0]  # [hidden_size]

        # 计算相似度
        scores = torch.matmul(query_embedding, passage_embeddings.t())  # [num_passages]

        # 获取top-k
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(passage_embeddings)))

        return top_indices.numpy(), top_scores.numpy()


def save_model(model, path):
    """保存模型"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path):
    """加载模型"""
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model


if __name__ == '__main__':
    import os
    # 测试代码
    print("Testing BiEncoder...")

    model = BiEncoder()
    query_input_ids = torch.randint(0, 1000, (4, 128))
    query_attention_mask = torch.ones(4, 128)
    passage_input_ids = torch.randint(0, 1000, (4, 128))
    passage_attention_mask = torch.ones(4, 128)

    query_vec, passage_vec = model(query_input_ids, query_attention_mask,
                                   passage_input_ids, passage_attention_mask)

    print(f"Query vector shape: {query_vec.shape}")
    print(f"Passage vector shape: {passage_vec.shape}")

    # 测试损失
    loss_fn = InfoNCELoss()
    loss = loss_fn(query_vec, passage_vec)
    print(f"Loss: {loss.item()}")
