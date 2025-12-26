# 环境配置说明

## Conda环境

- **环境名称**: `ir2025`
- **Python版本**: 3.9
- **创建时间**: 2025-12-24

## 快速开始

### 1. 激活环境

```bash
# 方法1: 使用设置脚本（推荐）
cd /home/tanyuqiao/IR_Project-main/data/IR_2025_Project
source setup_env.sh

# 方法2: 直接激活conda环境
conda activate ir2025
```

### 2. 验证环境

```bash
python -c "import torch; import transformers; print('环境配置成功！')"
```

## 依赖包列表

主要依赖包（详见 requirements.txt）：

- **torch**: 2.1.0 (PyTorch with CUDA 12.1)
- **transformers**: 4.35.0 (Hugging Face Transformers)
- **numpy**: 1.24.3
- **pandas**: 2.0.3
- **tqdm**: 4.66.1
- **scikit-learn**: 1.3.2

## GPU配置

系统有8个NVIDIA A800-SXM4-80GB GPU可用。

### 推荐使用的GPU

- **GPU 6-7**: 空闲，推荐使用
- **GPU 0-5**: 有其他任务在运行

### 指定GPU运行

在代码中设置设备：

```python
# 使用6号GPU
device = 'cuda:6'

# 使用7号GPU
device = 'cuda:7'
```

或在命令行中指定：

```bash
CUDA_VISIBLE_DEVICES=6 python train_simple.py
```

## 重新安装环境（如果需要）

```bash
# 1. 创建环境
conda create -n ir2025 python=3.9 -y

# 2. 激活环境
conda activate ir2025

# 3. 安装依赖
cd /home/tanyuqiao/IR_Project-main/data/IR_2025_Project
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 常见问题

### Q1: conda命令找不到
```bash
# 初始化conda（如果还没初始化）
conda init bash
source ~/.bashrc
```

### Q2: CUDA不可用
```bash
# 检查CUDA安装
nvidia-smi

# 检查PyTorch CUDA支持
python -c "import torch; print(torch.cuda.is_available())"
```

### Q3: 显存不足
```bash
# 查看GPU使用情况
nvidia-smi

# 使用其他GPU或减小batch_size
```

## 导出环境（用于复现）

```bash
# 导出conda环境
conda env export > environment.yml

# 在其他机器上重建
conda env create -f environment.yml
```

## 项目结构

```
IR_2025_Project/
├── corpus/                  # 语料库
│   └── wiki_webq_corpus.tsv
├── datas/                   # 原始数据
│   ├── webq-train.json
│   ├── webq-test.csv
│   └── webq-test.txt
├── processed/               # 预处理后的数据
│   ├── passages.pkl
│   ├── train_triples.tsv
│   └── ...
├── output/                  # 输出结果
│   ├── retriever/          # 模型
│   └── test_results/       # 测试结果
├── preprocess_data.py      # 数据预处理
├── train_simple.py         # 训练脚本
├── test_simple.py          # 测试脚本
├── run.sh                  # 一键运行
├── requirements.txt        # 依赖列表
└── setup_env.sh            # 环境设置脚本
```

## 运行项目

### 完整流程（一键运行）

```bash
cd /home/tanyuqiao/IR_Project-main/data/IR_2025_Project
source setup_env.sh
bash run.sh
```

### 分步运行

```bash
# 1. 数据预处理
python preprocess_data.py

# 2. 训练模型
python train_simple.py

# 3. 测试评估
python test_simple.py
```

## 提示

1. **首次运行**: 数据预处理和模型下载可能需要较长时间
2. **GPU选择**: 脚本默认使用GPU 6，可在代码中修改
3. **显存管理**: 如果显存不足，减小batch_size或使用梯度累积
4. **结果保存**: 所有结果保存在`output/`目录下

---

**更新日期**: 2025-12-24
