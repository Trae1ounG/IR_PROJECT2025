#!/bin/bash
# 优化版一键运行脚本

set -e

echo "======================================"
echo "WebQ检索系统 - 优化版（多GPU + 预编码）"
echo "======================================"

BASE_DIR="/home/tanyuqiao/IR_Project-main/data/IR_2025_Project"
cd $BASE_DIR

# 激活环境
echo ""
echo "Step 0: 激活conda环境..."
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ir2025

# Step 1: 数据预处理（如果还没做）
if [ ! -f "$BASE_DIR/processed/passages.pkl" ]; then
    echo ""
    echo "======================================"
    echo "Step 1: 数据预处理"
    echo "======================================"
    python preprocess_data.py
else
    echo "✓ 数据预处理已完成"
fi

# Step 2: 预编码passages（如果还没做）
if [ ! -f "$BASE_DIR/processed/passage_embeddings.pt" ]; then
    echo ""
    echo "======================================"
    echo "Step 2: 预编码Passages（一次性操作）"
    echo "======================================"
    echo "这将把所有passages编码并保存，之后训练和测试会快很多"
    python encode_passages.py
else
    echo "✓ Passages已预编码"
fi

# Step 3: 训练模型
echo ""
echo "======================================"
echo "Step 3: 训练检索模型（使用GPU 6和7）"
echo "======================================"
python train_optimized.py

# 检查训练是否成功
if [ ! -f "$BASE_DIR/output/retriever/best_model.pt" ]; then
    echo "Error: 训练失败，找不到best_model.pt"
    exit 1
fi

echo "✓ 训练完成！"

# Step 4: 测试模型
echo ""
echo "======================================"
echo "Step 4: 在测试集上评估（使用GPU 6和7）"
echo "======================================"
python test_optimized.py

echo ""
echo "======================================"
echo "✓ 全部完成！"
echo "======================================"
echo ""
echo "结果保存在: ./output/test_results/"
echo ""
echo "查看结果:"
echo "  - cat ./output/test_results/metrics.json"
echo "  - cat ./output/test_results/hit_at_k.txt"
echo ""
echo "性能优化说明:"
echo "  - 使用GPU 6和7进行并行训练"
echo "  - Passages已预编码，训练和测试速度提升显著"
echo ""
