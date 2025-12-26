# 快速开始指南

## 一、环境准备

### 1. 检查Python环境
```bash
python --version  # 应该是3.7+
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 二、运行方式

### 方式1：一键运行（最简单）

```bash
cd /home/tanyuqiao/IR_Project-main/data/IR_2025_Project
bash run.sh
```

这个脚本会自动完成：
- ✓ 数据预处理
- ✓ 模型训练
- ✓ 测试评估
- ✓ 生成结果文件

### 方式2：分步运行（方便调试）

#### Step 1: 数据预处理
```bash
cd /home/tanyuqiao/IR_Project-main/data/IR_2025_Project
python preprocess_data.py
```

**预期输出**：
```
Loading corpus from ./corpus/wiki_webq_corpus.tsv...
Loaded 5180525 passages
Processing WebQ train data from ./datas/webq-train.json...
Saved 10000 training triples to ./processed/train_triples.tsv
...
```

**检查输出**：
```bash
ls -lh processed/
# 应该看到:
# passages.pkl (约4GB)
# train_triples.tsv
# test_questions.pkl
# test_answers.pkl
# dev/ 目录
```

#### Step 2: 训练模型
```bash
python train_simple.py
```

**预期输出**：
```
Using device: cuda
Loading tokenizer...
Loading data...
Loaded 5180525 passages
Loaded 10000 QA pairs
Creating dataset...
Dataset size: 10000

Epoch 1/5
Training: 100%|████████| 1250/1250 [XX:XX<00:00, loss=X.XXXX]
Validating...
Encoding passages...
Retrieving...
Validation - Hit@1: 0.XXXX, Hit@10: 0.XXXX, Hit@100: 0.XXXX
Saved best model with Hit@10: 0.XXXX

Epoch 2/5
...
```

**检查输出**：
```bash
ls -lh output/retriever/
# 应该看到:
# best_model.pt (约440MB)
# checkpoint_epoch_*.pt (可选)
```

#### Step 3: 测试评估
```bash
python test_simple.py
```

**预期输出**：
```
Using device: cuda
Loading tokenizer...
Loading model from ./output/retriever/best_model.pt...
Loading data...
Loaded 5180525 passages
Loaded 2030 test questions

Encoding passages...
Encoded 5180525 passages

Retrieving and evaluating...
100%|████████| 2030/2030 [XX:XX<00:00]

==================================================
FINAL RESULTS
==================================================
Hit@1:   0.XXXX
Hit@10:  0.XXXX
Hit@100: 0.XXXX
==================================================

Saved metrics to ./output/test_results/metrics.json
Saved hit@k results to ./output/test_results/hit_at_k.txt
Saved detailed results to ./output/test_results/detailed_results.json
```

## 三、查看结果

### 1. 查看整体指标
```bash
cat output/test_results/metrics.json
```

### 2. 查看提交格式结果
```bash
cat output/test_results/hit_at_k.txt
```

### 3. 查看前20个结果
```bash
head -20 output/test_results/hit_at_k.txt
```

## 四、常见问题

### Q1: 数据预处理报错
**错误**: `FileNotFoundError: [Errno 2] No such file or directory`

**解决**: 检查数据文件是否存在
```bash
ls -lh datas/webq-train.json
ls -lh datas/webq-test.csv
ls -lh corpus/wiki_webq_corpus.tsv
```

### Q2: 训练时显存不足（CUDA out of memory）
**解决**: 减小batch_size

在 `train_simple.py` 中修改：
```python
BATCH_SIZE = 4  # 从8改为4或更小
```

### Q3: 编码passages时很慢
**说明**: 这是正常的，第一次需要编码500万+文档

**预计时间**:
- GPU (V100): 约30-60分钟
- GPU (3090): 约60-90分钟
- CPU: 不推荐

### Q4: Hit@K结果不理想
**可能原因**:
1. 训练epoch不够
2. 数据质量问题
3. 模型太小

**改进建议**:
1. 增加epoch数（在train_simple.py中修改 `EPOCHS = 10`）
2. 使用更大的预训练模型（修改 `MODEL_NAME = 'bert-large-uncased'`）
3. 增加训练样本数（在preprocess_data.py中修改 `max_samples=50000`）

### Q5: 想使用部分数据快速测试
**修改**: 在 `preprocess_data.py` 中修改：
```python
process_webq_train(train_json, passages, train_output, max_samples=100)  # 只用100个样本
```

## 五、参数调优建议

### 训练参数（在train_simple.py中）

```python
# 推荐配置1: 快速测试（约30分钟）
BATCH_SIZE = 16
EPOCHS = 3
MAX_LENGTH = 128

# 推荐配置2: 标准训练（约2-3小时）
BATCH_SIZE = 8
EPOCHS = 5
MAX_LENGTH = 256

# 推荐配置3: 高性能（约6-8小时）
BATCH_SIZE = 4
EPOCHS = 10
MAX_LENGTH = 256
MODEL_NAME = 'bert-large-uncased'  # 需要更多显存
```

## 六、提交准备

### 1. 检查结果文件
```bash
ls -lh output/test_results/
```

应该包含：
- `metrics.json` - 整体指标
- `hit_at_k.txt` - **提交用**
- `detailed_results.json` - 详细结果

### 2. 准备提交材料

创建提交目录：
```bash
mkdir -p submission
cp *.py submission/
cp run.sh submission/
cp -r util submission/
cp output/test_results/hit_at_k.txt submission/
```

打包：
```bash
tar -czf IR_Project_提交.tar.gz submission/
```

### 3. 撰写实验报告

参考 `实验报告模板.md` 撰写报告，填写实际结果。

## 七、时间预估

| 任务 | 时间（GPU） | 时间（CPU） |
|------|------------|------------|
| 数据预处理 | 5-10分钟 | 10-20分钟 |
| 模型训练（5 epochs） | 1-2小时 | 8-12小时 |
| 模型测试 | 30-60分钟 | 3-5小时 |
| **总计** | **2-3小时** | **12-18小时** |

**建议**: 强烈建议使用GPU，如果没有GPU，可以使用Google Colab或学校的服务器。

## 八、联系方式

如有问题，请参考：
1. README.md - 完整文档
2. 实验报告模板.md - 报告写作指南
3. 课程助教

---

**祝您顺利完成项目！**
