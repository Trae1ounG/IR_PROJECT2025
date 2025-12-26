# IR 2025 大作业 - WebQ检索系统

## 项目概述

本项目实现了基于双编码器（Bi-Encoder）的检索系统，在WebQ数据集上进行开放域问答的文档检索任务。

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- transformers 4.0+
- CUDA（推荐）

### 安装依赖

```bash
pip install torch transformers tqdm
```

## 数据准备

数据目录结构：
```
data/IR_2025_Project/
├── corpus/
│   └── wiki_webq_corpus.tsv      # Wikipedia语料库
└── datas/
    ├── webq-train.json            # 训练集
    ├── webq-test.csv              # 测试集
    └── webq-test.txt              # 测试集（txt格式）
```

## 快速开始

### 方法1：一键运行（推荐）

```bash
cd /home/tanyuqiao/IR_Project-main/data/IR_2025_Project
bash run.sh
```

这个脚本会自动完成：
1. 数据预处理
2. 模型训练
3. 测试评估

### 方法2：分步运行

#### Step 1: 数据预处理

```bash
python preprocess_data.py
```

这会生成：
- `processed/passages.pkl` - 语料库
- `processed/train_triples.tsv` - 训练三元组
- `processed/train_triples_qa.pkl` - 问题-答案映射
- `processed/test_questions.pkl` - 测试问题
- `processed/test_answers.pkl` - 测试答案
- `processed/dev/` - 验证集

#### Step 2: 训练模型

```bash
python train_retriever.py \
    --train_data ./processed/train_triples.tsv \
    --dev_data ./processed/dev \
    --passages_path ./processed/passages.pkl \
    --qa_mapping_path ./processed/train_triples_qa.pkl \
    --model_name bert-base-uncased \
    --batch_size 16 \
    --epochs 10 \
    --lr 2e-5 \
    --output_dir ./output/retriever
```

主要参数：
- `--model_name`: 预训练模型名称（默认bert-base-uncased）
- `--batch_size`: 批次大小（默认16）
- `--epochs`: 训练轮数（默认10）
- `--lr`: 学习率（默认2e-5）
- `--output_dir`: 输出目录

训练过程中会：
- 在验证集上评估Hit@1, Hit@10, Hit@100
- 保存最佳模型到 `output/retriever/best_model.pt`
- 每5个epoch保存checkpoint

#### Step 3: 测试评估

```bash
python test_retriever.py \
    --model_path ./output/retriever/best_model.pt \
    --model_name bert-base-uncased \
    --test_data_dir ./processed \
    --passages_path ./processed/passages.pkl \
    --output_dir ./output/test_results
```

这会生成：
- `output/test_results/metrics.json` - 整体指标
- `output/test_results/hit_at_k.txt` - Hit@k结果（提交格式）
- `output/test_results/detailed_results.json` - 详细结果

## 文件说明

### 核心文件

- `preprocess_data.py` - 数据预处理脚本
- `retriever_model.py` - 双编码器模型定义
- `retriever_dataset.py` - 数据集类
- `train_retriever.py` - 训练脚本
- `test_retriever.py` - 测试评估脚本
- `run.sh` - 一键运行脚本

### 工具文件

- `log.py` - 日志配置
- `util/retriever_utils.py` - 检索工具函数
- `cal_hit_multi.py` - Hit@k计算（参考实现）

## 模型架构

本项目采用双编码器架构：

```
Query -> BERT Encoder -> Query Vector
Passage -> BERT Encoder -> Passage Vector

Similarity = dot(Query_Vector, Passage_Vector)
```

### 损失函数

使用InfoNCE损失（对比学习）：

```
loss = -log(exp(sim(q, p+)) / sum(exp(sim(q, p))))
```

## 评价指标

- **Hit@1**: 前1个结果中是否包含答案
- **Hit@10**: 前10个结果中是否包含答案
- **Hit@100**: 前100个结果中是否包含答案

## 结果输出

测试完成后，结果保存在 `output/test_results/` 目录：

### 1. metrics.json
```json
{
  "hit_at_1": 0.XXXX,
  "hit_at_10": 0.XXXX,
  "hit_at_100": 0.XXXX
}
```

### 2. hit_at_k.txt（提交格式）
```
id	hit_at_1	hit_at_10	hit_at_100
0	1	1	1
1	0	1	1
...
```

### 3. detailed_results.json
每个查询的详细结果，包括top-10检索的文档和答案匹配情况。

## 实验报告要点

根据作业要求，实验报告需包含：

### 1. 实现方案
- 双编码器架构说明
- BERT预训练模型选择
- 训练策略

### 2. 代码说明
- 主要类和函数的功能
- 运行方法

### 3. 技术原理
- BERT编码器原理
- InfoNCE损失函数
- 相似度计算方法

### 4. 实验步骤
- 数据预处理
- 模型训练（在训练集上训练，验证集上调优）
- 测试评估（**未在测试集上训练**）

### 5. 结果汇报
- Hit@1, Hit@10, Hit@100的具体数值
- 与baseline的对比（如果有）

## 提交材料

根据作业要求，提交材料应包含：

1. **源代码** - 完整的代码
2. **可执行程序** - 运行脚本
3. **结果文件** - `hit_at_k.txt` 等
4. **实验报告** - PDF格式

### 提交方式

```bash
# 打包（不要包含中间文件）
tar -czf IR_Project_提交.tar.gz \
    *.py \
    run.sh \
    README.md \
    output/test_results/ \
    实验报告.pdf
```

发送至：`konglingdi24@mails.ucas.ac.cn`

邮件标题：`IR大作业_[组长姓名]`

截止日期：2025年12月31日 24点

## 常见问题

### Q1: 训练时显存不足
A: 减小batch_size或使用gradient_accumulation

### Q2: 数据预处理很慢
A: 这是正常的，第一次运行需要处理大量数据

### Q3: Hit@k结果不理想
A: 可以尝试：
- 增加训练epoch数
- 调整学习率
- 使用更大的预训练模型（如bert-large）
- 添加hard negative mining

## 改进方向（加分项）

1. **Hard Negative Mining**: 使用难负样本提升训练效果
2. **模型蒸馏**: 使用更大的教师模型
3. **数据增强**: 对query进行back-translation等增强
4. **集成学习**: 结合多个检索器的结果
5. **重排序**: 使用交叉编码器对检索结果重排序

## 联系方式

如有问题，请联系课程助教。

## 许可证

本项目仅用于学术研究和教育目的。
